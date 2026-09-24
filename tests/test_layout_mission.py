"""Carte Layout A et mission Unstoppable Force."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.engine.geometry import Disk, is_visible, mm_to_in  # noqa: E402
from fortyk.engine.layout import LAYOUTS_DIR, load_layout  # noqa: E402
from fortyk.engine.mission import Scoreboard, ScoringEvent, UnstoppableForce, controller, level_of_control  # noqa: E402

R32 = mm_to_in(32) / 2


class LayoutATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L = load_layout("layout_a")

    def test_shape(self):
        L = self.L
        self.assertEqual(L.board, (44.0, 60.0))
        self.assertEqual(len(L.objectives), 5)
        self.assertEqual(len(L.terrain), 17)
        self.assertEqual(L.symmetry_report(), [])
        self.assertEqual({o.kind for o in L.objectives}, {"home", "central", "expansion"})

    def test_zones(self):
        L = self.L
        self.assertTrue(L.in_deployment_zone((5, 19), "attacker"))  # partie profonde (20")
        self.assertFalse(L.in_deployment_zone((30, 15), "attacker"))  # partie à 12"
        self.assertTrue(L.in_deployment_zone((30, 11), "attacker"))
        self.assertTrue(L.in_deployment_zone((40, 41), "defender"))
        self.assertFalse(L.in_deployment_zone((5, 45), "defender"))
        self.assertTrue(L.in_deployment_zone((5, 49), "defender"))
        self.assertAlmostEqual(L.deployment_zones["attacker"].area, 22 * 20 + 22 * 12)
        self.assertAlmostEqual(L.deployment_zones["attacker"].area, L.deployment_zones["defender"].area)

    def test_objectives_sit_on_ruins(self):
        L = self.L
        for o in L.objectives:
            piece = L.piece_at(o.center)
            self.assertIsNotNone(piece, o.id)
            self.assertTrue(piece.is_area)
        self.assertEqual(L.home_objective("attacker").id, "home_attacker")
        self.assertEqual(L.territory_of(L.objective("expansion_attacker").center), "attacker")
        self.assertEqual(L.territory_of(L.objective("expansion_defender").center), "defender")
        self.assertIsNone(L.territory_of(L.center))  # sur la ligne

    def test_terrain_conversion_and_los(self):
        L = self.L
        terrain = L.terrain_objects()
        # toutes les empreintes bloquent la vue (validé : le prince derrière la barricade légère T8m est invisible)
        self.assertTrue(all(t.opaque and t.see_through_from_inside for t in terrain))
        self.assertEqual(len(terrain), 17)
        self.assertEqual(sum(1 for pc in L.terrain if pc.density == "dense"), 11)  # 7 ruines denses + 4 barricades denses
        # Deux socles de part et d'autre de la ruine centrale ne se voient pas ; de part et d'autre de T2 (légère) non plus.
        a, b = Disk(22, 22, R32), Disk(22, 38, R32)
        self.assertFalse(is_visible(a, b, terrain))
        c, d = Disk(29.75, 6, R32), Disk(29.75, 16, R32)  # à travers T2 (légère)
        self.assertFalse(is_visible(c, d, terrain))
        inside = Disk(22, 30, R32)  # dans la ruine centrale : voit dehors
        self.assertTrue(is_visible(inside, a, terrain))
        toe = Disk(29.75, 8.5, R32)  # touche T2 (y ≥ 9) : voit à travers
        self.assertTrue(is_visible(toe, d, terrain))


class MissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.L = load_layout("layout_a")
        cls.M = UnstoppableForce()

    def test_control(self):
        obj = self.L.objective("central")
        near = [(Disk(22 + 3.5, 30, R32), 2), (Disk(22, 30 + 2, R32), 2), (Disk(22, 30 + 10, R32), 2)]
        self.assertEqual(level_of_control(near, obj), 4)  # la troisième est trop loin
        self.assertEqual(controller({"attacker": 4, "defender": 2}), "attacker")
        self.assertIsNone(controller({"attacker": 2, "defender": 2}))
        self.assertIsNone(controller({}))

    def test_round_one_only_kills(self):
        self.assertEqual(self.M.score_command_phase("attacker", 1, {"central"}, self.L), [])
        ev = self.M.score_end_of_turn("attacker", 1, {"central"}, set(), 1, self.L)
        self.assertEqual([e.vp for e in ev], [3])

    def test_round_two_scoring(self):
        ev = self.M.score_command_phase("attacker", 2, {"central", "expansion_attacker", "home_attacker"}, self.L)
        self.assertEqual(sum(e.vp for e in ev), 8)  # le home ne compte pas
        ev = self.M.score_end_of_turn("attacker", 2, {"central", "home_attacker"}, {"home_attacker"}, 0, self.L)
        self.assertEqual([e.vp for e in ev], [3])  # objectif nouvellement pris
        ev = self.M.score_end_of_turn("attacker", 2, {"home_attacker"}, set(), 0, self.L)
        self.assertEqual(ev, [])  # prendre son home ne rapporte rien

    def test_final_round_moves_objective_scoring(self):
        self.assertEqual(self.M.score_command_phase("defender", 5, {"central"}, self.L), [])
        ev = self.M.score_end_of_turn("defender", 5, {"central", "expansion_defender"}, {"central"}, 1, self.L)
        self.assertEqual(sorted(e.vp for e in ev), [3, 3, 8])

    def test_end_of_battle(self):
        self.assertEqual([e.vp for e in self.M.score_end_of_battle("attacker", {"central"}, self.L)], [5])
        self.assertEqual(self.M.score_end_of_battle("attacker", {"expansion_attacker"}, self.L), [])

    def test_cap(self):
        sb = Scoreboard()
        r3 = self.M.score_command_phase("attacker", 3, {"central", "expansion_attacker", "expansion_defender"}, self.L)
        sb.add_all(r3)  # 12
        sb.add_all(self.M.score_end_of_turn("attacker", 3, {"central", "expansion_attacker", "expansion_defender"}, set(), 2, self.L))  # 3 + 3 → plafonné
        self.assertEqual(sb.round_total("attacker", 3), 15)
        sb.add(ScoringEvent("attacker", 5, "end_battle", "central", 5))
        self.assertEqual(sb.total("attacker"), 20)  # la fin de bataille n'est pas plafonnée
        sb.add(ScoringEvent("defender", 3, "command", "x", 4))
        self.assertEqual(sb.leader(), "attacker")
        self.assertEqual(sb.totals(), {"attacker": 20, "defender": 4})

    def test_max_per_round(self):
        self.assertEqual(self.M.max_per_round(self.L, "attacker"), 4 * 4 + 3 + 3)


@unittest.skipUnless((LAYOUTS_DIR / "layout_a.json").exists(), "layout absent")
class RenderTests(unittest.TestCase):
    def test_render_if_pillow(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow absent")
        import tempfile

        L = load_layout("layout_a")
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "a.png"
            L.render(out, scale=4)
            self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
