"""Bibliothèque de listes : coller, vérifier, enregistrer, jouer par son nom (API et navigateur)."""

import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import default_raw_dir, load_catalog  # noqa: E402
from fortyk.data import list_library as lib  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LIST_TEXT = (ROOT / "data" / "lists" / "ec_mercurial_host_2000.txt").read_text(encoding="utf-8")
HAS_DATA = (default_raw_dir() / "Datasheets.csv").exists()


def post(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as r:
        return json.load(r)


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class ListLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def test_slug_and_save_and_reload(self):
        self.assertEqual(lib.slugify("Mercurial Host – 2000 pts !"), "mercurial_host_2000_pts")
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            name, al = lib.save_list(LIST_TEXT, self.cat, directory=d)
            self.assertEqual(name, "ec_mercurial_host_2000")
            name2, _ = lib.save_list(LIST_TEXT, self.cat, directory=d)
            self.assertEqual(name2, "ec_mercurial_host_2000_2")  # pas d'écrasement silencieux
            listed = lib.saved_lists(self.cat, directory=d)
            self.assertEqual([x["name"] for x in listed], [name, name2])
            self.assertEqual(listed[0]["points"], 2000)
            again = lib.load_named_list(name, self.cat, directory=d)
            self.assertEqual(again.points, al.points)
            report = lib.list_report(al)
            self.assertEqual(report["points"], 2000)
            self.assertIn("Chaos Rhino", [u["name"] for u in report["units"]])
            self.assertGreater(report["coverage"]["joué"], 10)

    def test_garbage_is_refused(self):
        with self.assertRaises(ValueError):
            lib.resolve_text("bonjour", self.cat)

    def test_list_against_toy_roster(self):
        from fortyk.toy import new_state_from_lists

        s = new_state_from_lists("ec_mercurial_host_2000", None, self.cat, seed=1)
        self.assertEqual(s.units["EC3"].side, "defender")  # toy model côté défenseur (Daemon Prince)
        self.assertIn("CR1", s.units)
        self.assertEqual(set(s.army_lists), {"attacker"})

    def test_http_preview_save_and_new_game(self):
        from fortyk.web.server import App, make_handler

        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as games, mock.patch.object(lib, "LISTS_DIR", Path(tmp)):
            app = App(self.cat, data_dir=games)
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            th = threading.Thread(target=httpd.serve_forever, daemon=True)
            th.start()
            try:
                bad = post(base, "/api/lists/preview", {"text": "rien"})
                self.assertFalse(bad["ok"])
                pre = post(base, "/api/lists/preview", {"text": LIST_TEXT})
                self.assertTrue(pre["ok"])
                self.assertEqual(pre["report"]["points"], 2000)
                self.assertEqual(list(Path(tmp).glob("*.txt")), [])  # l'aperçu n'enregistre rien
                saved = post(base, "/api/lists/save", {"text": LIST_TEXT, "name": "ma liste EC"})
                self.assertEqual(saved["name"], "ma_liste_ec")
                with urllib.request.urlopen(base + "/api/lists") as r:
                    lists = json.load(r)["lists"]
                names = [x["name"] for x in lists]
                self.assertIn("ma_liste_ec", names)  # listes enregistrées + listes du dépôt
                self.assertIn("ec_mercurial_host_2000", names)
                self.assertTrue(next(x for x in lists if x["name"] == "ec_mercurial_host_2000").get("builtin"))
                new = post(base, "/api/games", {"lists": {"attacker": "ma_liste_ec", "defender": "ma_liste_ec"},
                                                "players": {"attacker": {"kind": "bot"}, "defender": {"kind": "human", "name": "Moi"}}})
                self.assertTrue(new["ok"], new)
                tok = new["links"]["defender"].split("t=")[1]
                with urllib.request.urlopen(base + f"/api/g/{new['id']}/state?since=0&t={tok}") as r:
                    st = json.load(r)
                ids = {u["id"] for u in st["units"]}
                self.assertIn("CR1", ids)
                self.assertEqual(st["you"], "defender")
                rhino = next(u for u in st["units"] if u["id"] == "CR1")
                self.assertGreater(rhino["models"][0]["hx"], 2.0)  # coque rectangulaire envoyée à l'interface
                # la liste est gardée en texte dans la partie : elle ne dépend plus du fichier
                doc = json.loads(Path(games, new["id"] + ".json").read_text(encoding="utf-8"))
                self.assertIn("Chaos Rhino", doc["config"]["lists"]["attacker"]["text"])
                wrong = post(base, "/api/games", {"list": "inconnue", "opponent": {"kind": "human"}})
                self.assertFalse(wrong["ok"])
            finally:
                httpd.shutdown()
                httpd.server_close()


if __name__ == "__main__":
    unittest.main()
