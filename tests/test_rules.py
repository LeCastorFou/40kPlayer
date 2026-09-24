"""Tests des fonctions de règles élémentaires (table de blessure, sauvegardes, plafonds)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.engine.rules import DEFAULT_RULES, clamp_modifier, clamp_roll, save_needed, wound_roll_needed  # noqa: E402


class WoundTableTests(unittest.TestCase):
    def test_table(self):
        self.assertEqual(wound_roll_needed(8, 4), 2)  # S ≥ 2T
        self.assertEqual(wound_roll_needed(9, 4), 2)
        self.assertEqual(wound_roll_needed(5, 4), 3)  # S > T
        self.assertEqual(wound_roll_needed(7, 4), 3)
        self.assertEqual(wound_roll_needed(4, 4), 4)  # S = T
        self.assertEqual(wound_roll_needed(3, 4), 5)  # S < T
        self.assertEqual(wound_roll_needed(5, 9), 5)
        self.assertEqual(wound_roll_needed(2, 4), 6)  # S ≤ T/2
        self.assertEqual(wound_roll_needed(4, 9), 6)

    def test_bolt_rifle_vs_marine(self):
        self.assertEqual(wound_roll_needed(4, 4), 4)  # bolt rifle S4 sur T4 : 4+


class SaveTests(unittest.TestCase):
    def test_armour_and_ap(self):
        self.assertEqual(save_needed(3, 0), 3)
        self.assertEqual(save_needed(3, -1), 4)  # bolt rifle AP-1 sur 3+
        self.assertEqual(save_needed(3, -3), 6)
        self.assertIsNone(save_needed(3, -4))  # 7+ : pas de sauvegarde
        self.assertEqual(save_needed(2, 0), 2)  # jamais mieux que 2+
        self.assertEqual(save_needed(5, 0), 5)

    def test_invulnerable(self):
        self.assertEqual(save_needed(3, -4, invuln=4), 4)  # l'invul sauve la mise
        self.assertEqual(save_needed(2, -1, invuln=4), 3)  # l'armure reste meilleure
        self.assertEqual(save_needed(3, 0, invuln=5), 3)


class ClampTests(unittest.TestCase):
    def test_modifier_cap(self):
        self.assertEqual(clamp_modifier(3), 1)
        self.assertEqual(clamp_modifier(-2), -1)
        self.assertEqual(clamp_modifier(0), 0)
        self.assertEqual(clamp_modifier(1 - 1), 0)  # Heavy +1 et couvert -1 s'annulent

    def test_roll_bounds(self):
        self.assertEqual(clamp_roll(1), 2)
        self.assertEqual(clamp_roll(7), 6)
        self.assertEqual(clamp_roll(4), 4)

    def test_objective_radius(self):
        self.assertAlmostEqual(DEFAULT_RULES.objective_control_radius_in, 3 + 20 / 25.4)
        self.assertEqual(DEFAULT_RULES.engagement_range_in, 2.0)


if __name__ == "__main__":
    unittest.main()
