"""Tests du parseur de caractéristiques (sans dépendance aux CSV).

Exécution : ``python3 -m unittest discover -s tests`` (ou ``pytest``).
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data.parse import (  # noqa: E402
    DiceExpr,
    ParseError,
    parse_base_size,
    parse_dice,
    parse_inches,
    parse_int,
    parse_save,
    parse_skill,
    parse_weapon_keywords,
    strip_html,
)


class DiceTests(unittest.TestCase):
    def test_fixed(self):
        d = parse_dice("3")
        self.assertEqual(d, DiceExpr.fixed(3))
        self.assertTrue(d.is_fixed)
        self.assertEqual((d.min, d.max, d.mean), (3, 3, 3.0))

    def test_dice_forms(self):
        self.assertEqual(parse_dice("D6"), DiceExpr(1, 6, 0))
        self.assertEqual(parse_dice("d3"), DiceExpr(1, 3, 0))
        self.assertEqual(parse_dice("2D6"), DiceExpr(2, 6, 0))
        self.assertEqual(parse_dice("D6+1"), DiceExpr(1, 6, 1))
        self.assertEqual(parse_dice("2D6+3"), DiceExpr(2, 6, 3))
        self.assertEqual(parse_dice("D6 "), DiceExpr(1, 6, 0))  # espace parasite de l'export

    def test_stats(self):
        d = parse_dice("2D6+3")
        self.assertEqual((d.min, d.max), (5, 15))
        self.assertAlmostEqual(d.mean, 10.0)
        self.assertAlmostEqual(parse_dice("D3").mean, 2.0)
        self.assertAlmostEqual(parse_dice("D6+1").mean, 4.5)

    def test_roll_within_bounds(self):
        rng = random.Random(42)
        d = parse_dice("2D6+3")
        for _ in range(200):
            self.assertTrue(d.min <= d.roll(rng) <= d.max)
        self.assertEqual(parse_dice("4").roll(rng), 4)

    def test_na(self):
        self.assertIsNone(parse_dice("-"))
        self.assertIsNone(parse_dice(""))
        self.assertIsNone(parse_dice("N/A"))

    def test_str_roundtrip(self):
        for s in ("3", "D6", "2D6", "D6+1", "2D6+3", "D3"):
            self.assertEqual(str(parse_dice(s)), s)

    def test_error(self):
        with self.assertRaises(ParseError):
            parse_dice("beaucoup")


class ScalarTests(unittest.TestCase):
    def test_int(self):
        self.assertEqual(parse_int("-2"), -2)
        self.assertEqual(parse_int("-0"), 0)
        self.assertEqual(parse_int("7 "), 7)
        self.assertIsNone(parse_int("-"))

    def test_skill(self):
        self.assertEqual(parse_skill("3"), 3)
        self.assertEqual(parse_skill("3+"), 3)
        self.assertIsNone(parse_skill("N/A"))  # Torrent
        self.assertIsNone(parse_skill("-"))

    def test_save(self):
        self.assertEqual(parse_save("3+"), 3)
        self.assertEqual(parse_save("4"), 4)
        self.assertEqual(parse_save("4*"), 4)
        self.assertIsNone(parse_save("-"))

    def test_inches(self):
        self.assertEqual(parse_inches('6"'), 6.0)
        self.assertEqual(parse_inches("24"), 24.0)
        self.assertEqual(parse_inches('20+"'), 20.0)
        self.assertIsNone(parse_inches("Melee"))
        self.assertIsNone(parse_inches("-"))
        self.assertIsNone(parse_inches(""))


class BaseSizeTests(unittest.TestCase):
    def test_round(self):
        b = parse_base_size("32mm")
        self.assertEqual((b.shape, b.width_mm, b.depth_mm, b.flying), ("round", 32.0, 32.0, False))
        self.assertAlmostEqual(b.radius_mm, 16.0)
        self.assertAlmostEqual(b.radius_in, 16 / 25.4)
        self.assertEqual(parse_base_size("28.5mm").width_mm, 28.5)

    def test_oval_and_flying(self):
        b = parse_base_size("120 x 92mm flying base")
        self.assertEqual((b.shape, b.width_mm, b.depth_mm, b.flying), ("oval", 120.0, 92.0, True))
        self.assertAlmostEqual(b.radius_mm, 60.0)
        self.assertTrue(parse_base_size("60mm flying base").flying)
        self.assertFalse(parse_base_size("75 x 42mm").flying)

    def test_unknown(self):
        self.assertIsNone(parse_base_size("Use model"))
        self.assertIsNone(parse_base_size("No official base size"))
        self.assertIsNone(parse_base_size(""))


class WeaponKeywordTests(unittest.TestCase):
    def names(self, text):
        return [k.name for k in parse_weapon_keywords(text)]

    def test_simple_list(self):
        self.assertEqual(self.names("assault, heavy"), ["assault", "heavy"])
        self.assertEqual(self.names(""), [])
        self.assertEqual(self.names("ignores cover, pistol, torrent"), ["ignores cover", "pistol", "torrent"])

    def test_case_and_typos_normalised(self):
        self.assertEqual(self.names("IGNORES COvER"), ["ignores cover"])
        self.assertEqual(self.names("TwIN-lINkED"), ["twin-linked"])
        self.assertEqual(self.names("pISTOl"), ["pistol"])

    def test_numeric_value(self):
        (k,) = parse_weapon_keywords("rapid fire 2")
        self.assertEqual((k.name, k.value, k.plus), ("rapid fire", 2, False))
        (k,) = parse_weapon_keywords("melta 2")
        self.assertEqual((k.name, k.value), ("melta", 2))

    def test_anti(self):
        (k,) = parse_weapon_keywords("anti-infantry 4+")
        self.assertEqual((k.name, k.target, k.value, k.plus), ("anti", "infantry", 4, True))
        self.assertTrue(k.known)
        self.assertEqual(str(k), "anti-infantry 4+")
        (k,) = parse_weapon_keywords("ANTI-MONSTER/VEHICLE 4+")
        self.assertEqual(k.target, "monster/vehicle")

    def test_dice_value(self):
        (k,) = parse_weapon_keywords("sustained hits d3")
        self.assertEqual((k.name, k.value), ("sustained hits", DiceExpr(1, 3, 0)))
        (k,) = parse_weapon_keywords("rapid fire d6+3")
        self.assertEqual(k.value, DiceExpr(1, 6, 3))

    def test_condition(self):
        (k,) = parse_weapon_keywords("LETHAL HITS: non-MONSTER/VEHICLE")
        self.assertEqual((k.name, k.condition), ("lethal hits", "non-monster/vehicle"))
        (k,) = parse_weapon_keywords("sustained hits 2: monster/vehicle")
        self.assertEqual((k.name, k.value, k.condition), ("sustained hits", 2, "monster/vehicle"))

    def test_unknown_kept(self):
        (k,) = parse_weapon_keywords("c'tan power")
        self.assertEqual(k.name, "c'tan power")
        self.assertFalse(k.known)
        (k,) = parse_weapon_keywords("cleave 2")
        self.assertEqual((k.name, k.value, k.known), ("cleave", 2, True))


class HtmlTests(unittest.TestCase):
    def test_strip(self):
        html = "<b>Every model</b> is equipped with: bolt pistol;<br>bolt rifle &amp; more."
        self.assertEqual(strip_html(html), "Every model is equipped with: bolt pistol;\nbolt rifle & more.")

    def test_lists(self):
        html = "Replace with:<ul style='x'><li>1 chainsword</li><li>1 hand flamer</li></ul>"
        self.assertEqual(strip_html(html), "Replace with:\n1 chainsword\n1 hand flamer")


if __name__ == "__main__":
    unittest.main()
