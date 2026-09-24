"""Import de liste d'armée (export texte NewRecruit), résolution contre Wahapedia, unités jouables."""

import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fortyk.data import default_raw_dir, load_catalog  # noqa: E402
from fortyk.data.army_list import parse_army_list, resolve_army_list  # noqa: E402

HAS_DATA = (default_raw_dir() / "Datasheets.csv").exists()
LIST_PATH = ROOT / "data" / "lists" / "ec_mercurial_host_2000.txt"
LIST_TEXT = LIST_PATH.read_text(encoding="utf-8")


class ParseTests(unittest.TestCase):
    def test_header_and_entries(self):
        doc = parse_army_list(LIST_TEXT)
        self.assertEqual(doc.faction, "Chaos - Emperor's Children")
        self.assertEqual(doc.detachments, ["Mercurial Host", "Spectacle of Slaughter"])
        self.assertEqual(doc.detachment_rules, ["Quicksilver Grace"])
        self.assertEqual(doc.force_disposition, "Reconnaissance")
        self.assertEqual((doc.total_points, doc.number_of_units), (2000, 14))
        self.assertEqual(doc.warlord, "Fulgrim")
        self.assertEqual(len(doc.header_enhancements), 3)
        self.assertIn("Bring It Down", doc.secondaries)
        self.assertTrue(doc.source.startswith("newrecruit.eu"))
        self.assertEqual(len(doc.entries), 14)
        keys = [e.key for e in doc.entries]
        self.assertIn("Infractors[2]", keys)
        self.assertIn("Flawless Blades[3]", keys)
        fulgrim = doc.entries[0]
        self.assertEqual((fulgrim.label, fulgrim.name, fulgrim.count, fulgrim.points, fulgrim.tags), ("Char1", "Fulgrim", 1, 340, ["Warlord"]))
        exultant = doc.entries[1]
        self.assertEqual(exultant.leading, "Infractors[1]")
        inf = next(e for e in doc.entries if e.key == "Infractors[1]")
        self.assertEqual([(g.count, g.model_name) for g in inf.groups], [(1, "Obsessionist"), (4, "Infractor")])
        self.assertEqual(inf.attached_to, "Lord Exultant[1]")
        defiler = next(e for e in doc.entries if e.name == "Defiler")
        self.assertIn((2, "Excruciator cannon"), defiler.groups[0].items)
        spawn = next(e for e in doc.entries if e.name == "Chaos Spawn")
        self.assertEqual((spawn.groups[0].count, spawn.groups[0].items), (2, [(1, "Hideous Mutations")]))
        fb = next(e for e in doc.entries if e.key == "Flawless Blades[1]")
        self.assertEqual((fb.enhancement, fb.enhancement_points, fb.attached_to), ("Beguiling Grotesquerie", 15, "Lucius the Eternal"))

    def test_mixed_loadouts_in_one_line(self):
        doc = parse_army_list("5x Tormentors (95 pts)\n• 1x Obsessionist: Bolt pistol, Power sword\n• 4x Tormentor: 3 with Boltgun, Close combat weapon, 1 with Meltagun, Close combat weapon\n")
        e = doc.entries[0]
        self.assertEqual([(g.count, g.model_name, g.items) for g in e.groups],
                         [(1, "Obsessionist", [(1, "Bolt pistol"), (1, "Power sword")]),
                          (3, "Tormentor", [(1, "Boltgun"), (1, "Close combat weapon")]),
                          (1, "Tormentor", [(1, "Meltagun"), (1, "Close combat weapon")])])


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class ResolveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()
        cls.al = resolve_army_list(parse_army_list(LIST_TEXT), cls.cat)

    def test_valid_list_points_and_rules(self):
        al = self.al
        self.assertEqual(al.issues, [])
        self.assertEqual((al.faction_id, al.points, al.dp), ("EC", 2000, 3))
        self.assertEqual([d.name for d in al.detachments], ["Mercurial Host", "Spectacle of Slaughter"])
        # toutes les règles des détachements pris s'appliquent (V11)
        self.assertEqual([a.name for a in al.active_rules], ["Quicksilver Grace", "Entitled to Victory"])
        self.assertEqual(al.warlord.key, "Fulgrim[1]")
        self.assertEqual(al.unit("Defiler[1]").points_computed, 330)  # 300 + 2 × 15 (heavy reaper autocannons)
        self.assertEqual(al.unit("Flawless Blades[1]").points_computed, 110)  # 95 + 15 (amélioration)
        # 10 stratagèmes core V11 + 6 Mercurial Host + 3 Spectacle of Slaughter
        self.assertEqual(len(al.stratagems), 19)
        self.assertEqual(sum(1 for s in al.stratagems if s.is_core), 10)
        overwatch = next(s for s in al.stratagems if s.name == "FIRE OVERWATCH")
        self.assertTrue(overwatch.is_reactive)
        self.assertIn("snap shooting", overwatch.effect)

    def test_models_weapons_and_leaders(self):
        al = self.al
        inf = al.unit("Infractors[1]")
        self.assertEqual(inf.leaders, ["Lord Exultant[1]"])
        self.assertEqual(al.unit("Lord Exultant[1]").leading, "Infractors[1]")
        self.assertEqual(al.unit("Lucius the Eternal[1]").leading, "Flawless Blades[1]")
        self.assertEqual([m.name for m in inf.models], ["Obsessionist"] + ["Infractor"] * 4)
        self.assertEqual(sorted(w.name for w in inf.models[0].weapons), ["Bolt pistol", "Power sword"])
        defiler = al.unit("Defiler[1]").models[0]
        self.assertEqual(sum(1 for w in defiler.weapons if w.name == "Excruciator cannon"), 2)
        self.assertEqual({w.name for w in defiler.weapons if w.group == "Shearing claws"}, {"Shearing claws – strike", "Shearing claws – sweep"})
        fulgrim = al.unit("Fulgrim[1]").models[0]
        self.assertNotIn("Warlord", [w.name for w in fulgrim.weapons])
        d = al.to_dict()
        self.assertEqual(d["points"], {"listed": 2000, "computed": 2000})
        self.assertEqual(len(d["units"]), 14)

    def test_errors_are_reported(self):
        bad = LIST_TEXT.replace("1x Maulerfiend (120 pts)", "1x Maulerfiend (125 pts)")
        bad = bad.replace("Maulerfiend fists, Lasher tendrils", "Maulerfiend fists, Lasher tendrils, Laser banana")
        bad = bad.replace("Leading Flawless Blades[1]", "Leading Infractors[2]")
        bad = bad.replace("3x Flawless Blades (110 pts)\n  Enhancement: Beguiling Grotesquerie (+15 pts)\n  3 with Blissblade, Bolt pistol\n  Attached to Lucius the Eternal\n",
                          "3x Flawless Blades (110 pts)\n  Enhancement: Beguiling Grotesquerie (+15 pts)\n  3 with Blissblade, Bolt pistol\n")
        bad = bad.replace("+ DETACHMENT: Mercurial Host, Spectacle of Slaughter (Quicksilver Grace)", "+ DETACHMENT: Mercurial Host (Quicksilver Grace)")
        al = resolve_army_list(parse_army_list(bad), self.cat)
        text = "\n".join(al.issues)
        self.assertIn("125 pts dans la liste, 120 pts recalculés", text)
        self.assertIn("Laser banana", text)
        self.assertIn("Lucius the Eternal[1] ne peut pas mener Infractors", text)
        self.assertIn("détachement absent", text)  # Beguiling Grotesquerie vient de Spectacle of Slaughter
        self.assertIn("recalculés", text)


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class ImportedArmyInEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        from fortyk.toy import new_state_from_lists

        self.s = new_state_from_lists(LIST_PATH, LIST_PATH, self.cat, seed=5)

    def test_units_built_with_leaders_and_gear(self):
        import math

        s = self.s
        att = [u for u in s.units.values() if u.side == "attacker"]
        self.assertEqual(len(att), 11)  # 14 entrées dont 3 personnages attachés
        self.assertEqual(len(s.units), 22)
        self.assertEqual(len({u.id for u in s.units.values()}), 22)  # identifiants uniques entre les deux camps
        inf = s.units["INF1"]
        self.assertEqual(inf.name, "Infractors + Lord Exultant [INF1]")
        self.assertEqual(inf.strength, 6)
        self.assertTrue(inf.models[0].is_leader and inf.models[0].name == "Lord Exultant")
        self.assertEqual(inf.models[0].profile.wounds, 5)
        self.assertTrue(inf.has_keyword("Character"))  # mot-clé du personnage
        self.assertTrue(inf.has_ability("Perfectionists") and inf.has_ability("Excessive Assault"))
        self.assertEqual(inf.points, 175)
        self.assertTrue(s.units["FUL1"].is_warlord)
        rhino = s.units["CR1"].models[0]  # coque rectangulaire (data/hulls.json), axe long vers l'adversaire
        self.assertEqual((rhino.radius, round(rhino.hx * 2 * 25.4), round(rhino.hy * 2 * 25.4)), (0.0, 117, 88))
        self.assertAlmostEqual(rhino.angle, math.pi / 2)
        self.assertEqual(s.units["FB1"].enhancements, ("Beguiling Grotesquerie",))
        # les blessures vont aux gardes du corps d'abord : le personnage tombe en dernier
        inf.allocate_damage([2] * 4)  # les 4 Infractors ; l'Obsessionist (chef d'escouade) reste
        self.assertEqual([m.alive for m in inf.models], [True, True, False, False, False, False])
        inf.allocate_damage([2] * 2)  # l'Obsessionist, puis enfin le Lord Exultant
        self.assertEqual([m.alive for m in inf.models], [True, False, False, False, False, False])
        self.assertEqual(inf.models[0].wounds, 3)

    def test_duplicated_weapons_fire_twice(self):
        from fortyk.engine.combat import build_shooting_profiles
        from fortyk.engine.movement import apply_translation

        s = self.s
        for u in s.units.values():
            c = u.centroid
            apply_translation(u, 22 - c[0] + (0 if u.side == "attacker" else 0), (8 if u.side == "attacker" else 52) - c[1])
        dfl, target = s.units["DEF1"], s.units["DEF2"]
        for u, (x, y) in ((dfl, (41, 20)), (target, (41, 40))):
            c = u.centroid
            apply_translation(u, x - c[0], y - c[1])
            u.reset_turn_flags()
        profiles, _ = build_shooting_profiles(s, dfl, target)
        counts = {p.label.split(" ×")[0]: p.count for p in profiles}
        self.assertEqual(counts.get("Excruciator cannon"), 2)
        self.assertEqual(counts.get("Heavy reaper autocannon"), 2)

    def test_full_random_game_between_lists(self):
        from fortyk.engine import Engine
        from fortyk.engine.actions import EndPhaseAction

        s, eng, rng = self.s, Engine(), random.Random(2)
        eng.start(s)
        n = 0
        while not eng.is_over(s):
            d = eng.decision(s)
            opts = [o for o in d.options if not isinstance(o, EndPhaseAction)] if d.kind == "select_unit" else d.options
            eng.step(s, rng.choice(opts or d.options))
            n += 1
        self.assertGreater(n, 100)
        self.assertEqual(s.battle_round, 5)


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class CoverageTests(unittest.TestCase):
    def test_registry_matches_engine_code(self):
        """Chaque règle déclarée « jouée » est bien référencée dans le code du moteur."""
        from fortyk.engine.coverage import IMPLEMENTED_ABILITIES, IMPLEMENTED_WEAPON_KEYWORDS

        src = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "fortyk" / "engine").glob("*.py") if p.name != "coverage.py")
        for name in IMPLEMENTED_ABILITIES:
            if name == "Leader":
                self.assertIn("leaders", src)
                continue
            self.assertIn(f'"{name}"', src, name)
        combat = (ROOT / "fortyk" / "engine" / "combat.py").read_text(encoding="utf-8")
        for kw in IMPLEMENTED_WEAPON_KEYWORDS:
            self.assertTrue(re.search(rf'(has|keyword)\("{re.escape(kw)}"\)', combat), kw)

    def test_coverage_of_the_list(self):
        from fortyk.engine.coverage import coverage

        al = resolve_army_list(parse_army_list(LIST_TEXT), load_catalog())
        status = {(it.category.split(" (")[0], it.name): it.status for it in coverage(al)}
        self.assertEqual(status[("capacité", "Deadly Demise")], "joué")
        self.assertEqual(status[("capacité", "Scouts")], "joué")
        self.assertEqual(status[("capacité", "Leader")], "partiel")
        self.assertEqual(status[("mot-clé d'arme", "precision")], "non joué")
        self.assertEqual(status[("mot-clé d'arme", "sustained hits")], "joué")
        self.assertEqual(status[("règle de détachement", "Quicksilver Grace")], "non joué")
        self.assertEqual(status[("structure", "Transport")], "partiel")
        self.assertEqual(status[("structure", "Empreinte sans socle (coque)")], "partiel")


if __name__ == "__main__":
    unittest.main()
