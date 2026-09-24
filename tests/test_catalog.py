"""Tests d'intégration du catalogue sur l'export Wahapedia réel.

Sautés si les CSV ne sont pas présents (``python3 scripts/fetch_wahapedia.py``).
Les valeurs attendues viennent de la fiche Intercessor Squad V11 (Space Marines,
Faction Pack 1.2) et servent de garde-fou : un changement de règles ou de format
d'export fera échouer ces tests, ce qui est le but.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import CatalogError, DiceExpr, default_raw_dir, load_catalog  # noqa: E402

RAW_DIR = default_raw_dir()
HAS_DATA = (RAW_DIR / "Datasheets.csv").exists()


@unittest.skipUnless(HAS_DATA, f"CSV Wahapedia absents de {RAW_DIR}")
class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def test_global_shape(self):
        self.assertGreater(len(self.cat), 1500)
        self.assertGreaterEqual(len(self.cat.factions), 20)
        self.assertEqual(self.cat.warnings, [], "lignes ignorées par le parseur :\n" + "\n".join(self.cat.warnings))
        for ds in self.cat:
            self.assertIn(ds.faction_id, self.cat.factions, ds.name)

    def test_every_weapon_parsed(self):
        rangeless = []
        for ds in self.cat:
            for w in ds.weapons:
                self.assertIn(w.kind, ("ranged", "melee"))
                self.assertIsInstance(w.attacks, DiceExpr)
                self.assertIsInstance(w.damage, DiceExpr)
                if w.is_melee:
                    self.assertIsNone(w.range_in)
                elif w.range_in is None:
                    rangeless.append(f"{ds.name} / {w.name}")
                if w.skill is None and not w.is_melee:
                    self.assertTrue(w.has("torrent"), f"{ds.name} / {w.name} : BS N/A sans Torrent")
        # Seule exception connue : le missile Deathstrike (portée « N/A », tiré via sa capacité).
        self.assertTrue(all("Deathstrike" in r for r in rangeless), rangeless)
        self.assertLessEqual(len(rangeless), 4, rangeless)

    def test_torrent_data_quality(self):
        # Wahapedia laisse parfois un BS sur une arme Torrent (Venerable Dreadnought / Heavy flamer).
        # Le moteur devra ignorer BS dès que Torrent est présent ; on surveille juste que ça reste marginal.
        odd = [f"{ds.name} / {w.name}" for ds in self.cat for w in ds.weapons if w.has("torrent") and w.skill is not None]
        self.assertLessEqual(len(odd), 5, odd)

    def test_intercessor_profile(self):
        ds = self.cat.get("Intercessor Squad")
        self.assertEqual(ds.faction_id, "SM")
        self.assertFalse(ds.virtual)
        (m,) = ds.models
        self.assertEqual((m.move_in, m.toughness, m.save, m.invuln, m.wounds, m.leadership, m.oc), (6.0, 4, 3, None, 2, 6, 2))
        self.assertEqual(m.base.raw, "32mm")
        self.assertAlmostEqual(m.base_radius_in, 16 / 25.4)

    def test_intercessor_weapons(self):
        ds = self.cat.get("intercessor squad")  # insensible à la casse
        rifle = ds.weapon("Bolt rifle")
        self.assertEqual((rifle.range_in, rifle.attacks, rifle.skill, rifle.strength, rifle.ap, rifle.damage), (24.0, DiceExpr.fixed(2), 3, 4, -1, DiceExpr.fixed(1)))
        self.assertEqual(sorted(k.name for k in rifle.keywords), ["assault", "heavy"])
        self.assertTrue(rifle.has("assault"))
        self.assertFalse(rifle.has("pistol"))
        flamer = ds.weapon("Hand flamer")
        self.assertIsNone(flamer.skill)
        self.assertEqual(flamer.attacks, DiceExpr(1, 6, 0))
        ccw = ds.weapon("Close combat weapon")
        self.assertTrue(ccw.is_melee)
        self.assertEqual((ccw.attacks.flat, ccw.skill, ccw.strength, ccw.ap), (3, 3, 4, 0))
        plasma = [w for w in ds.weapons if w.group == "Plasma pistol"]
        self.assertEqual([w.profile_index for w in plasma], [1, 2])
        self.assertTrue(plasma[1].has("hazardous"))

    def test_intercessor_keywords_costs_abilities(self):
        ds = self.cat.get("Intercessor Squad")
        self.assertEqual(ds.faction_keywords, ("Adeptus Astartes",))
        for kw in ("Infantry", "Battleline", "Imperium", "Grenades"):
            self.assertTrue(ds.has_keyword(kw), kw)
        self.assertEqual(ds.cost_for(5), 80)
        self.assertEqual(ds.cost_for(10), 150)
        self.assertEqual(ds.min_cost, 80)
        self.assertEqual(ds.composition, ("1 Intercessor Sergeant", "4-9 Intercessors"))
        names = [a.name for a in ds.abilities]
        self.assertIn("Oath of Moment", names)  # capacité de faction résolue via Abilities.csv
        self.assertIsNotNone(ds.ability("Hail of Bolts"))
        self.assertIn("Captain", [self.cat.datasheets[i].name for i in ds.leader_ids])

    def test_tiered_costs(self):
        ds = self.cat.get("Leman Russ Commander", faction="AM")
        self.assertEqual(ds.cost_for(1, copy_index=1), 195)
        self.assertEqual(ds.cost_for(1, copy_index=2), 195)
        self.assertEqual(ds.cost_for(1, copy_index=3), 210)
        self.assertEqual(ds.min_cost, 195)
        self.assertTrue(any(c.kind == "wargear" for c in ds.costs))
        self.assertTrue(all(c.models is None for c in ds.wargear_costs))

    def test_cost_context(self):
        ds = self.cat.get("Inquisitor")
        contexts = {c.context for c in ds.costs}
        self.assertEqual(len(contexts), 2)
        self.assertEqual(ds.cost_for(1, context="Assigned Agent"), 65)

    def test_ambiguity_requires_faction(self):
        with self.assertRaises(CatalogError):
            self.cat.get("Leman Russ Commander")
        self.assertEqual(self.cat.get("Leman Russ Commander", "Astra Militarum").faction_id, "AM")
        with self.assertRaises(CatalogError):
            self.cat.get("Fiche Imaginaire")

    def test_find_and_by_faction(self):
        hits = self.cat.find(r"^Intercessor", faction="SM")
        self.assertIn("Intercessor Squad", [d.name for d in hits])
        ec = self.cat.by_faction("EC")
        self.assertGreater(len(ec), 15)
        self.assertIn("Fulgrim", [d.name for d in ec])
        infractors = self.cat.get("Infractors")
        self.assertEqual(infractors.faction_id, "EC")
        self.assertEqual(infractors.models[0].base.raw, "32mm")
        self.assertEqual(infractors.faction_keywords, ("Emperor’s Children",))  # mot-clé vide de l'export filtré

    def test_no_empty_keywords(self):
        for ds in self.cat:
            for k in ds.keyword_entries:
                self.assertTrue(k.name, f"{ds.name} : mot-clé vide")

    def test_core_abilities_resolved(self):
        deep_strike = [ds.name for ds in self.cat if any(a.name == "Deep Strike" and a.is_core for a in ds.abilities)]
        self.assertGreater(len(deep_strike), 100)
        scouts = [a for ds in self.cat for a in ds.core_abilities if a.name == "Scouts"]
        self.assertTrue(all(a.parameter.endswith('"') for a in scouts))


if __name__ == "__main__":
    unittest.main()
