"""Service web : parties sauvegardées à deux joueurs, jetons, bot, annulation (nouveaux dés), API HTTP."""

import json
import random
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import default_raw_dir, load_catalog  # noqa: E402
from fortyk.engine.actions import Decision, EndPhaseAction, FightAction, ShootAction, StratagemAction  # noqa: E402
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

    def to_movement(self):
        """Déploiement automatique des deux camps, puis phases de scouts terminées : premier mouvement."""
        room, tok = self.new()
        room.set_settings(tok["defender"], {"auto_deploy": True})
        out = room.set_settings(tok["attacker"], {"auto_deploy": True})
        self.assertTrue(out["auto_deploy"] and out["auto_actions"] and out["step_mode"])
        self.assertGreater(len(room.record["history"]), 0)
        self.assertTrue(all(h["by"] == "auto" for h in room.record["history"]))
        for _ in range(10):
            if room.state.phase == "movement":
                break
            d = room.engine.decision(room.state)
            room.act(tok[d.side], {"type": "end_phase"})
        self.assertEqual(room.state.phase, "movement")
        return room, tok

    def test_direct_move_without_selecting_and_live_check(self):
        room, tok = self.to_movement()
        d = room.engine.decision(room.state)
        self.assertEqual((d.kind, d.side), ("select_unit", "defender"))
        t = tok["defender"]
        u = room.state.unit("EC1")
        good = {m.id: [m.x + 4, m.y] for m in u.models}
        bad = {m.id: [m.x, m.y - 9] for m in u.models}
        move = {"type": "model_move", "unit_id": "EC1", "kind": "normal"}
        # vérification à blanc pendant le glissement : rien n'est joué
        n, sig = len(room.record["history"]), signature(room.state)
        self.assertEqual(room.check(t, {**move, "positions": good}), {"ok": True, "error": None, "model_id": None})
        res = room.check(t, {**move, "positions": bad})
        self.assertFalse(res["ok"])
        self.assertEqual(res["model_id"], "EC1.1")
        self.assertIn('plus que 7"', res["error"])
        self.assertFalse(room.check(tok["attacker"], {**move, "positions": good})["ok"])  # pas son tour
        self.assertEqual((len(room.record["history"]), signature(room.state)), (n, sig))
        # action refusée : la sélection implicite est défaite
        with self.assertRaisesRegex(RoomError, "plus que 7"):
            room.act(t, {**move, "positions": bad})
        self.assertEqual((len(room.record["history"]), signature(room.state)), (n, sig))
        self.assertEqual(room.engine.decision(room.state).kind, "select_unit")
        # glisser l'unité directement : sélection implicite + mouvement, en une action
        room.act(t, {**move, "positions": good})
        tail = room.record["history"][n:]
        self.assertEqual([(h["decision"], h.get("implicit")) for h in tail], [("select_unit", True), ("move", None)])
        self.assertAlmostEqual(room.state.unit("EC1").models[0].x, u.models[0].x + 4, places=3)
        self.assertTrue(room.history()[-2]["implicit"])
        # une seule annulation défait les deux
        room.undo(t)
        self.assertEqual((len(room.record["history"]), room.state.unit("EC1").models[0].x), (n, u.models[0].x))

    def test_auto_actions_follow_player_settings(self):
        room, tok = self.new()
        shoot = Decision("shoot", "attacker", [ShootAction("SM1", "EC1"), ShootAction("SM1", None)], unit_id="SM1")
        two = Decision("shoot", "attacker", [ShootAction("SM1", "EC1"), ShootAction("SM1", "EC2"), ShootAction("SM1", None)], unit_id="SM1")
        fight = Decision("fight", "attacker", [FightAction("SM1", "EC1")], unit_id="SM1")
        self.assertEqual(room._auto_choice(shoot), ShootAction("SM1", "EC1"))  # une seule cible possible
        self.assertIsNone(room._auto_choice(two))  # un vrai choix : on demande
        self.assertEqual(room._auto_choice(fight), FightAction("SM1", "EC1"))
        room.set_settings(tok["attacker"], {"auto_actions": False})
        self.assertIsNone(room._auto_choice(shoot))
        self.assertIsNone(room._auto_choice(fight))
        self.assertFalse(room.snapshot(tok["attacker"])["settings"]["auto_actions"])
        self.assertTrue(room.snapshot(tok["defender"])["settings"]["auto_actions"])  # réglage par joueur
        with self.assertRaises(RoomError):
            room.set_settings(None, {"auto_actions": True})  # spectateur
        # les réglages survivent au rechargement
        again = Room(self.cat, self.store, self.store.load(room.record["id"]))
        self.assertFalse(again.setting("attacker", "auto_actions"))

    def test_frames_unit_details_and_long_poll(self):
        room, tok = self.to_movement()
        n = len(room.record["history"])
        f = room.frames(0)
        self.assertEqual((f["start"], len(f["frames"])), (0, n))
        first = f["frames"][1]  # déploiement de la première unité
        self.assertEqual(first["label"], room.record["history"][1]["label"])
        self.assertTrue(first["models"])
        self.assertTrue(all(v[5] == 1 for v in first["models"].values()))  # posées sur la table
        self.assertEqual(room.frames(n)["frames"], [])
        det = room.unit_details("SM1")
        self.assertEqual(det["name"], "Intercessor Squad")
        self.assertEqual(det["profiles"][0]["T"], 4)
        self.assertTrue(any(w["kind"] == "ranged" for w in det["weapons"]))
        with self.assertRaises(RoomError):
            room.unit_details("ZZ9")
        # long-polling : le client qui attend est réveillé dès qu'une action est jouée
        v = room.version
        woke = []
        th = threading.Thread(target=lambda: (room.wait_change(v, timeout=10), woke.append(room.version)))
        th.start()
        room.act(tok["defender"], {"type": "select_unit", "unit_id": "EC1"})
        th.join(5)
        self.assertEqual(len(woke), 1)
        self.assertGreater(woke[0], v)
        t0 = time.monotonic()
        room.wait_change(v, timeout=5)  # version déjà dépassée : retour immédiat
        self.assertLess(time.monotonic() - t0, 1)

    def test_stratagem_windows_settings_and_protocol(self):
        from fortyk.training import build_initial_state
        from fortyk.web.rooms import decode_action
        from fortyk.web.serialize import action_label, decision_to_json

        room, tok = self.new()
        self.assertEqual((room.record["config"]["rev"], room.record["config"]["stratagems"]), (2, True))
        self.assertTrue(room.state.stratagems)
        old = dict(room.record["config"])
        del old["rev"], old["stratagems"]  # partie enregistrée avant les stratagèmes : rejouée sans
        legacy = build_initial_state(self.cat, old)
        self.assertEqual((legacy.rev, legacy.stratagems), (1, False))
        use = StratagemAction("fire_overwatch", "SM1", target_id="EC1")
        d = Decision("stratagem", "attacker", [use, StratagemAction(None)], window="fire_overwatch", note="…")
        self.assertIsNone(room._auto_choice(d))  # fenêtres affichées par défaut
        self.assertEqual(decode_action(d, {"type": "stratagem", "stratagem": "fire_overwatch", "unit_id": "SM1", "target_id": "EC1"}), use)
        self.assertEqual(decode_action(d, {"type": "stratagem", "stratagem": None}), StratagemAction(None))
        with self.assertRaises(ValueError):
            decode_action(d, {"type": "stratagem", "stratagem": "explosives"})
        js = decision_to_json(room.state, d)
        self.assertEqual(js["window"], "fire_overwatch")
        self.assertEqual((js["options"][0]["cost"], js["options"][0]["name"]), (1, "Fire Overwatch"))
        self.assertIn("expected_damage", js["options"][0])
        self.assertTrue(js["options"][1]["pass"])
        self.assertEqual(action_label(room.state, d, StratagemAction(None)), "pas de Fire Overwatch")
        self.assertIn("Fire Overwatch (1 CP)", action_label(room.state, d, use))
        room.set_settings(tok["attacker"], {"stratagems": False})
        self.assertEqual(room._auto_choice(d), StratagemAction(None))  # coupées : on passe sans demander
        snap = room.snapshot(tok["attacker"])
        self.assertFalse(snap["settings"]["stratagems"])
        self.assertTrue(snap["stratagems"])
        self.assertEqual(snap["stratagems_used"], [])

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

    def test_fluidity_endpoints(self):
        base, app = self.serve()
        res = request(base, "/api/games", {"lists": {"attacker": None, "defender": None}, "players": HUMANS})
        gid = res["id"]
        ta = res["links"]["attacker"].split("t=")[1]
        out = request(base, f"/api/g/{gid}/settings?t={ta}", {"auto_deploy": True, "step_mode": False})
        self.assertTrue(out["ok"] and out["auto_deploy"] and not out["step_mode"])
        st = request(base, f"/api/g/{gid}/state?since=0&t={ta}")
        self.assertEqual(st["settings"], {"step_mode": False, "auto_actions": True, "auto_deploy": True, "stratagems": True})
        self.assertEqual(st["pending"]["side"], "defender")  # l'attaquant a été déployé d'office
        self.assertIn("cp", st)
        # long-polling : sans changement, la réponse arrive après le délai demandé
        t0 = time.monotonic()
        st2 = request(base, f"/api/g/{gid}/state?since=0&t={ta}&wait={st['version']}&timeout=0.4")
        self.assertGreaterEqual(time.monotonic() - t0, 0.35)
        self.assertEqual(st2["version"], st["version"])
        chk = request(base, f"/api/g/{gid}/check?t={ta}", {"type": "deploy_models", "unit_id": "SM1", "positions": {}})
        self.assertFalse(chk["ok"])
        fr = request(base, f"/api/g/{gid}/frames?from=0")
        self.assertEqual(len(fr["frames"]), st["history_length"])
        self.assertEqual(request(base, f"/api/g/{gid}/unit?id=SM3")["name"], "Redemptor Dreadnought")
        with self.assertRaises(urllib.error.HTTPError):
            request(base, f"/api/g/{gid}/unit?id=nope")

    def test_delete_game(self):
        base, app = self.serve()
        res = request(base, "/api/games", {"lists": {"attacker": None, "defender": None}, "players": HUMANS})
        gid = res["id"]
        ta, td = res["links"]["attacker"].split("t=")[1], res["links"]["defender"].split("t=")[1]
        self.assertTrue(request(base, f"/api/g/{gid}/action?t={ta}", {"type": "option", "index": 0})["ok"])
        self.assertFalse(request(base, f"/api/g/{gid}/delete", {})["ok"])  # spectateur
        out = request(base, f"/api/g/{gid}/delete?t={td}", {})
        self.assertTrue(out["ok"], out)
        self.assertEqual(request(base, "/api/games")["games"], [])
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request(base, f"/api/g/{gid}/state?t={ta}")
        self.assertEqual(ctx.exception.code, 410)
        body = json.loads(ctx.exception.read())
        self.assertTrue(body["deleted"])
        self.assertIn("Paul", body["error"])
        refused = request(base, f"/api/g/{gid}/action?t={ta}", {"type": "option", "index": 0})
        self.assertFalse(refused["ok"])
        self.assertTrue(refused.get("deleted"))
        self.assertTrue((app.store.directory / "deleted" / f"{gid}.json").exists())  # récupérable à la main
        self.assertFalse((app.store.directory / f"{gid}.json").exists())
        # partie en attente : seul le créateur peut l'annuler
        w = request(base, "/api/games", {"side": "attacker", "name": "Valentin", "opponent": {"kind": "human"}})
        invite = w["join"]["invite"].split("t=")[1]
        self.assertFalse(request(base, f"/api/g/{w['id']}/delete?t={invite}", {})["ok"])
        self.assertTrue(request(base, f"/api/g/{w['id']}/delete?t={w['links']['attacker'].split('t=')[1]}", {})["ok"])
        self.assertFalse(request(base, "/api/join", {"code": w["join"]["code"], "name": "Paul"})["ok"])

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
