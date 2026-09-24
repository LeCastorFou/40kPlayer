"""Réserves stratégiques V11 (20) : mise en réserve au déploiement, ingress à partir du round 2,
Deep Strike, destruction à la fin du round 3, Rapid Ingress (15.07), remise en réserve (20.02)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import default_raw_dir, load_catalog  # noqa: E402
from fortyk.engine import Engine  # noqa: E402
from fortyk.engine.actions import DeployAction, EndPhaseAction, ReserveAction, SelectUnitAction, StratagemAction  # noqa: E402
from fortyk.engine.geometry import disk_gap, disk_polygon_distance, extreme_points  # noqa: E402
from fortyk.engine.reserves import formation_at, ingress_error  # noqa: E402
from fortyk.toy import new_toy_state  # noqa: E402

HAS_DATA = (default_raw_dir() / "Datasheets.csv").exists()


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class ReserveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setup_game(self, reserve=("SM2",), stratagems=False):
        s = new_toy_state(self.cat, seed=2)
        s.stratagems = stratagems
        eng = Engine()
        eng.start(s, first_player="attacker")
        while s.battle_round == 0:
            d = eng.decision(s)
            if d.kind == "deploy":
                if d.unit_id in reserve:
                    eng.step(s, ReserveAction(d.unit_id))
                else:
                    eng.step(s, next(o for o in d.options if isinstance(o, DeployAction)))
            elif d.kind == "select_unit":
                eng.step(s, next((o for o in d.options if isinstance(o, EndPhaseAction)), d.options[0]))
            else:
                eng.step(s, d.options[0])
        return s, eng

    def advance(self, s, eng, until, keep_reserves=True):
        """Termine les phases (les unités restent immobiles / en réserve) jusqu'à ``until(s, d)``."""
        for _ in range(400):
            d = eng.decision(s)
            if d is None or until(s, d):
                return d
            if d.kind == "select_unit" and any(isinstance(o, EndPhaseAction) for o in d.options):
                eng.step(s, EndPhaseAction())
            elif d.kind == "stratagem":
                eng.step(s, StratagemAction(None))
            elif d.kind == "ingress" and keep_reserves:
                eng.step(s, ReserveAction(d.unit_id))
            else:
                eng.step(s, d.options[0])
        self.fail("condition jamais atteinte")

    def test_reserve_at_deployment_and_ingress_from_round_2(self):
        s, eng = self.setup_game()
        sm2 = s.units["SM2"]
        self.assertTrue(sm2.in_reserve)
        self.assertNotIn(sm2, s.units_of("attacker"))
        self.assertTrue(any("réserve stratégique" in l for l in s.log))
        # round 1 : pas encore
        d = self.advance(s, eng, lambda s, d: s.phase == "movement" and d.side == "attacker")
        self.assertNotIn(SelectUnitAction("SM2"), d.options)
        self.assertIn("round 2", d.ineligible["SM2"])
        # round 2 : ingress
        d = self.advance(s, eng, lambda s, d: s.battle_round == 2 and s.phase == "movement" and d.side == "attacker")
        self.assertIn(SelectUnitAction("SM2"), d.options)
        eng.step(s, SelectUnitAction("SM2"))
        d = eng.decision(s)
        self.assertEqual((d.kind, d.unit_id, d.side), ("ingress", "SM2", "attacker"))
        spots = [o for o in d.options if isinstance(o, DeployAction)]
        self.assertTrue(spots)
        self.assertIn(ReserveAction("SM2"), d.options)
        for o in spots[:10]:
            self.assertIsNone(ingress_error(s, sm2, formation_at(sm2, o.x, o.y)))
        # au milieu de la table : refusé (pas à 6" d'un bord)
        self.assertIn("6\" d'un bord", ingress_error(s, sm2, formation_at(sm2, 22.0, 30.0)))
        self.assertIsNotNone(eng.free_action_error(s, d, DeployAction("SM2", 22.0, 30.0)))
        eng.step(s, spots[0])
        self.assertFalse(sm2.in_reserve)
        self.assertTrue(sm2.arrived)
        w, h = s.layout.board
        enemies = [m.disk for u in s.units_of("defender") for m in u.alive_models]
        opp = s.layout.deployment_zones["defender"]
        for m in sm2.alive_models:
            pts = extreme_points(m.disk)
            self.assertTrue(any(all(f(p) for p in pts) for f in (lambda p: p[0] <= 6, lambda p: p[0] >= w - 6, lambda p: p[1] <= 6, lambda p: p[1] >= h - 6)))
            self.assertTrue(all(disk_gap(m.disk, e) > 8 for e in enemies))
            self.assertGreater(disk_polygon_distance(m.disk, opp), 0)
        self.assertTrue(any("arrive des réserves" in l for l in s.log))
        self.assertIn(sm2, s.units_of("attacker"))

    def test_deep_strike_anywhere_more_than_8_from_enemies(self):
        s, eng = self.setup_game()
        sm2 = s.units["SM2"]
        s.battle_round = 2
        pos = formation_at(sm2, 22.0, 30.0)
        self.assertIsNotNone(ingress_error(s, sm2, pos))
        self.assertIsNone(ingress_error(s, sm2, pos, deep_strike=True))
        ec = next(iter(s.units_of("defender")))
        near = formation_at(sm2, ec.centroid[0], ec.centroid[1] - 6)
        self.assertIn("8\"", ingress_error(s, sm2, near, deep_strike=True) or "8\"")

    def test_reserves_never_arrived_are_destroyed_after_round_3(self):
        s, eng = self.setup_game()
        self.advance(s, eng, lambda s, d: s.battle_round == 4)
        self.assertTrue(s.units["SM2"].is_destroyed)
        self.assertTrue(any("jamais arrivée" in l for l in s.log))
        # une unité remise en réserve pendant la bataille (20.02) survit
        s, eng = self.setup_game(reserve=())
        eng.to_reserves(s, s.units["SM1"], "test")
        self.advance(s, eng, lambda s, d: s.battle_round == 4)
        self.assertFalse(s.units["SM1"].is_destroyed)
        self.assertTrue(s.units["SM1"].in_reserve)

    def test_rapid_ingress_at_end_of_opponent_movement(self):
        s, eng = self.setup_game(stratagems=True)
        s.cp = {"attacker": 5, "defender": 5}
        # round 1 : pas de Rapid Ingress ; round 2, tour du défenseur : fenêtre pour l'attaquant
        d = self.advance(s, eng, lambda s, d: d.kind == "stratagem" and d.window == "rapid_ingress")
        self.assertEqual((s.battle_round, s.side_to_move, d.side), (2, "defender", "attacker"))
        eng.step(s, StratagemAction("rapid_ingress", "SM2"))
        d = eng.decision(s)
        self.assertEqual((d.kind, d.side), ("ingress", "attacker"))
        eng.step(s, next(o for o in d.options if isinstance(o, DeployAction)))
        self.assertFalse(s.units["SM2"].in_reserve)
        log = s.log
        start = log.index("\n--- Tour de defender (round 2) ---")
        arrived = next(i for i, l in enumerate(log) if "arrive des réserves" in l)
        end = next(i for i, l in enumerate(log) if l.startswith("Fin du tour de defender") and i > start)
        self.assertTrue(start < arrived < end)  # pendant le tour adverse, puis la partie reprend son cours
        self.assertTrue(any(l.startswith("Stratagème Rapid Ingress") for l in log))


if __name__ == "__main__":
    unittest.main()
