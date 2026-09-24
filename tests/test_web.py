"""Service web : parties sauvegardées à deux joueurs, jetons, bot, annulation (nouveaux dés), API HTTP."""

import json
import random
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import default_raw_dir, load_catalog  # noqa: E402
from fortyk.engine.actions import EndPhaseAction  # noqa: E402
from fortyk.web.rooms import GameStore, Room, RoomError  # noqa: E402
from fortyk.web.server import App, make_handler  # noqa: E402

HAS_DATA = (default_raw_dir() / "Datasheets.csv").exists()
TOY = {"attacker": None, "defender": None}
HUMANS = {"attacker": {"kind": "human", "name": "Valentin"}, "defender": {"kind": "human", "name": "Paul"}}


def signature(s):
    return (s.battle_round, s.phase, s.scoreboard.totals(), tuple(s.log),
            tuple((m.id, round(m.x, 4), round(m.y, 4), m.wounds, m.alive) for u in s.units.values() for m in u.models))


def play_random(room, tokens, n, rng):
    """Joue ``n`` décisions au hasard (options du moteur) pour les camps humains ; s'arrête en fin de partie."""
    for _ in range(n):
        d = room.engine.decision(room.state)
        if d is None:
            return
        idx = [i for i, o in enumerate(d.options) if not isinstance(o, EndPhaseAction)] if d.kind == "select_unit" else list(range(len(d.options)))
        room.act(tokens[d.side], {"type": "option", "index": rng.choice(idx or [0])})


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class RoomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = GameStore(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def new(self, players=HUMANS, seed=7):
        room = Room.create(self.cat, self.store, lists=TOY, players=players, seed=seed)
        return room, {s: p["token"] for s, p in room.record["players"].items() if p["token"]}

    def test_tokens_turns_and_spectators(self):
        room, tok = self.new()
        snap = room.snapshot(tok["attacker"])
        self.assertEqual(snap["you"], "attacker")
        self.assertEqual(snap["pending"]["kind"], "select_unit")
        self.assertTrue(snap["waiting_for_human"])
        self.assertEqual(room.snapshot(None)["you"], None)  # spectateur
        with self.assertRaises(RoomError):
            room.act(tok["defender"], {"type": "select_unit", "unit_id": "EC1"})  # pas son tour
        with self.assertRaises(RoomError):
            room.act(None, {"type": "select_unit", "unit_id": "SM1"})  # spectateur
        with self.assertRaises(RoomError):
            room.act("faux", {"type": "select_unit", "unit_id": "SM1"})
        room.act(tok["attacker"], {"type": "select_unit", "unit_id": "SM1"})
        snap = room.snapshot(tok["attacker"])
        unit = next(u for u in snap["units"] if u["id"] == "SM1")
        bad = {m["id"]: [10 + i * 1.5, 30] for i, m in enumerate(unit["models"])}
        with self.assertRaisesRegex(RoomError, "zone de déploiement"):
            room.act(tok["attacker"], {"type": "deploy_models", "unit_id": "SM1", "positions": bad})
        good = {m["id"]: [10 + (i % 3) * 1.5, 8 + (i // 3) * 1.5] for i, m in enumerate(unit["models"])}
        room.act(tok["attacker"], {"type": "deploy_models", "unit_id": "SM1", "positions": good})
        self.assertEqual(room.snapshot(tok["defender"])["pending"]["side"], "defender")
        hist = room.history()
        self.assertEqual([h["label"] for h in hist], ["active Intercessor Squad [SM1]", "déploie Intercessor Squad [SM1]"])
        # le fichier est écrit après chaque action et ne dépend pas d'un fichier de liste
        doc = json.loads(Path(self.tmp.name, room.record["id"] + ".json").read_text(encoding="utf-8"))
        self.assertEqual(doc["format"], "fortyk-game/2")
        self.assertEqual(len(doc["history"]), 2)

    def test_undo_rerolls_dice_and_reload_is_identical(self):
        room, tok = self.new(seed=11)
        rng = random.Random(3)
        play_random(room, tok, 60, rng)
        n = len(room.record["history"])
        k = n - 12
        old_rng = room.snaps[k].rng.getstate()
        res = room.undo(tok["defender"], to=k)  # n'importe quel joueur, n'importe quel point
        self.assertEqual((res["undone"], res["to"]), (12, k))
        self.assertEqual(len(room.record["history"]), k)
        self.assertNotEqual(room.state.rng.getstate(), old_rng)  # nouveaux dés à partir d'ici
        self.assertEqual(room.record["reseeds"][-1]["at"], k)
        self.assertEqual(room.record["undos"][-1]["count"], 12)
        self.assertEqual(len(room.record["undos"][-1]["undone"]), 12)  # la branche annulée reste dans le journal
        self.assertEqual(room.snapshot(tok["attacker"])["last_undo"]["by"], "defender")
        play_random(room, tok, 25, rng)
        room.undo(tok["attacker"])  # la dernière action
        play_random(room, tok, 10, rng)
        live = signature(room.state)
        again = Room(self.cat, self.store, self.store.load(room.record["id"]))  # redémarrage du serveur
        self.assertEqual(signature(again.state), live)
        self.assertEqual(again.snapshot(tok["attacker"])["timeline"], 2)

    def test_bot_game_step_mode_and_undo_back_to_human(self):
        room, tok = self.new(players={"attacker": {"kind": "human", "name": "Valentin"}, "defender": {"kind": "bot"}})
        self.assertNotIn("defender", tok)
        t = tok["attacker"]
        room.act(t, {"type": "select_unit", "unit_id": "SM1"})
        room.act(t, {"type": "option", "index": 0})  # déploiement proposé
        snap = room.snapshot(t)
        self.assertEqual(snap["pending"]["kind"], "observe")  # pas à pas : le bot a déployé, on valide
        self.assertTrue(snap["vs_bot"])
        room.act(t, {"type": "continue"})
        room.set_step_mode(t, False)
        rng = random.Random(1)
        play_random(room, tok, 40, rng)
        by = [h["by"] for h in room.record["history"]]
        self.assertIn("bot", by)
        last_human = max(i for i, h in enumerate(room.record["history"]) if h["by"] == "human")
        room.undo(t)
        self.assertEqual(len(room.record["history"]), last_human)
        d = room.engine.decision(room.state)
        self.assertEqual(d.side, "attacker")  # retour à la dernière décision humaine, le bot ne rejoue pas tout seul

    def test_full_game_vs_bot_and_training_export(self):
        from fortyk.training import training_export

        room, tok = self.new(players={"attacker": {"kind": "bot"}, "defender": {"kind": "human", "name": "Paul"}}, seed=21)
        t = tok["defender"]
        room.set_step_mode(t, False)
        n = 0
        while not room.engine.is_over(room.state):
            snap = room.snapshot(t)
            p = snap["pending"]
            if p["kind"] == "move":
                room.act(t, {"type": "declare_advance", "unit_id": p["unit_id"]} if n % 3 == 0 else {"type": "stationary", "unit_id": p["unit_id"]})
            elif p["kind"] == "select_unit" and n % 5 == 4 and any(o.get("end_phase") for o in p["options"]):
                room.act(t, {"type": "end_phase"})
            else:
                room.act(t, {"type": "option", "index": 0})
            n += 1
            self.assertLess(n, 800)
        snap = room.snapshot(t)
        self.assertTrue(snap["game_over"])
        self.assertIsNotNone(snap["result"])
        self.assertEqual(room.record["status"], "finished")
        ex = training_export(self.cat, room.record)
        self.assertEqual(ex["format"], "fortyk-training/1")
        self.assertEqual(len(ex["samples"]), len(room.record["history"]))
        self.assertEqual(ex["result"]["scores"], room.state.scoreboard.totals())
        first = ex["samples"][0]
        self.assertIn("state", first)
        self.assertEqual({u["side"] for u in first["state"]["units"]}, {"attacker", "defender"})
        self.assertTrue(any(s["by"] == "bot" for s in ex["samples"]))
        # une annulation après la fin rouvre la partie
        room.undo(t)
        self.assertEqual(room.record["status"], "active")


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req) as r:
        body = r.read()
        return json.loads(body) if r.headers.get_content_type() == "application/json" else body


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def serve(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        app = App(self.cat, data_dir=tmp.name)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return f"http://127.0.0.1:{httpd.server_address[1]}", app

    def test_two_player_game_over_http(self):
        base, app = self.serve()
        self.assertIn(b"Nouvelle partie", request(base, "/"))
        health = request(base, "/healthz")
        self.assertEqual((health["ok"], health["storage"]), (True, "ok"))
        res = request(base, "/api/games", {"title": "Test", "lists": {"attacker": None, "defender": None}, "players": HUMANS})
        self.assertTrue(res["ok"], res)
        gid, la, ld = res["id"], res["links"]["attacker"], res["links"]["defender"]
        ta, td = la.split("t=")[1], ld.split("t=")[1]
        self.assertIn(b"<canvas", request(base, f"/g/{gid}"))
        self.assertEqual(request(base, f"/api/g/{gid}/layout")["board"], [44.0, 60.0])
        st = request(base, f"/api/g/{gid}/state?since=0&t={ta}")
        self.assertEqual((st["you"], st["pending"]["side"], st["game"]["title"]), ("attacker", "attacker", "Test"))
        self.assertEqual(st["players"]["defender"]["name"], "Paul")
        refused = request(base, f"/api/g/{gid}/action?t={td}", {"type": "option", "index": 0})
        self.assertFalse(refused["ok"])
        self.assertTrue(request(base, f"/api/g/{gid}/action?t={ta}", {"type": "option", "index": 0})["ok"])
        self.assertEqual(len(request(base, f"/api/g/{gid}/history")["history"]), 1)
        self.assertFalse(request(base, f"/api/g/{gid}/undo", {})["ok"])  # spectateur
        und = request(base, f"/api/g/{gid}/undo?t={td}", {"to": None})
        self.assertTrue(und["ok"], und)
        st2 = request(base, f"/api/g/{gid}/state?since=0&t={td}")
        self.assertEqual((st2["history_length"], st2["timeline"], st2["last_undo"]["name"]), (0, 1, "Paul"))
        rec = request(base, f"/api/g/{gid}/record")
        self.assertEqual(rec["format"], "fortyk-game/2")
        self.assertTrue(all("token" not in p for p in rec["players"].values()))  # le journal ne donne pas les liens
        ex = request(base, f"/api/g/{gid}/export?states=0")
        self.assertEqual(ex["format"], "fortyk-training/1")
        games = request(base, "/api/games")["games"]
        self.assertEqual([g["id"] for g in games], [gid])
        self.assertNotIn(ta, json.dumps(games))
        with self.assertRaises(urllib.error.HTTPError):
            request(base, "/api/g/inconnue/state")

    def test_join_with_game_code_and_own_list(self):
        """Le créateur choisit sa liste et son camp ; l'adversaire rejoint avec le code de partie et sa liste."""
        base, app = self.serve()
        res = request(base, "/api/games", {"side": "defender", "name": "Valentin", "list": None, "opponent": {"kind": "human"}})
        self.assertTrue(res["ok"], res)
        self.assertEqual((res["status"], res["join"]["side"]), ("waiting", "attacker"))
        code, gid = res["join"]["code"], res["id"]
        self.assertRegex(code, r"^[A-Z2-9]{3}-[A-Z2-9]{3}$")
        mine = res["links"]["defender"].split("t=")[1]
        # en attente : le créateur voit le code, un spectateur non, la partie est listée sans le code
        st = request(base, f"/api/g/{gid}/state?t={mine}")
        self.assertTrue(st["waiting"])
        self.assertEqual(st["join"]["code"], code)
        self.assertIsNone(request(base, f"/api/g/{gid}/state")["join"])
        self.assertEqual(request(base, f"/api/g/{gid}/layout")["board"], [44.0, 60.0])
        games = request(base, "/api/games")["games"]
        self.assertEqual((games[0]["status"], games[0]["open_side"]), ("waiting", "attacker"))
        self.assertNotIn(code.replace("-", ""), json.dumps(games))
        self.assertNotIn("join", request(base, f"/api/g/{gid}/record"))
        self.assertFalse(request(base, f"/api/g/{gid}/action?t={mine}", {"type": "option", "index": 0})["ok"])
        # mauvais code, puis bon code tapé sans tiret et en minuscules
        self.assertFalse(request(base, "/api/join", {"code": "AAA-AAA", "name": "Paul"})["ok"])
        joined = request(base, "/api/join", {"code": code.replace("-", "").lower(), "name": "Paul", "list": "ec_mercurial_host_2000"})
        self.assertTrue(joined["ok"], joined)
        self.assertEqual(joined["side"], "attacker")
        theirs = joined["link"].split("t=")[1]
        st = request(base, f"/api/g/{gid}/state?since=0&t={theirs}")
        self.assertNotIn("waiting", st)
        self.assertEqual((st["you"], st["players"]["attacker"]["name"], st["players"]["defender"]["name"]), ("attacker", "Paul", "Valentin"))
        self.assertIn("CR1", {u["id"] for u in st["units"]})  # la liste choisie par l'adversaire
        self.assertEqual(st["game"]["title"], "Emperor’s Children vs Emperor’s Children")
        self.assertEqual(request(base, f"/api/g/{gid}/state?t={mine}")["you"], "defender")
        self.assertFalse(request(base, "/api/join", {"code": code, "name": "Intrus"})["ok"])  # plus de place
        # par le lien d'invitation
        res2 = request(base, "/api/games", {"side": "attacker", "name": "Valentin", "opponent": {"kind": "human"}})
        invite = res2["join"]["invite"].split("t=")[1]
        self.assertTrue(request(base, f"/api/g/{res2['id']}/state?t={invite}")["can_join"])
        self.assertFalse(request(base, f"/api/g/{res2['id']}/join?t=faux", {"name": "X"})["ok"])
        ok2 = request(base, f"/api/g/{res2['id']}/join?t={invite}", {"name": "Paul"})
        self.assertTrue(ok2["ok"], ok2)
        self.assertEqual(ok2["side"], "defender")
        # contre le bot : la partie démarre tout de suite, pas de code
        res3 = request(base, "/api/games", {"side": "attacker", "name": "Moi", "opponent": {"kind": "bot", "list": None}})
        self.assertEqual((res3["status"], res3.get("join")), ("active", None))


if __name__ == "__main__":
    unittest.main()
