"""Transports (capacité, embarquer, débarquer, transport détruit), Scouts et coques de véhicule."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import default_raw_dir, load_catalog  # noqa: E402
from fortyk.engine import Engine  # noqa: E402
from fortyk.engine.actions import (  # noqa: E402
    DeployAction, DisembarkAction, DisembarkModelsAction, EmbarkAction, EndPhaseAction, ModelMoveAction, MoveAction, SelectUnitAction,
)
from fortyk.engine.army import UnitSpec  # noqa: E402
from fortyk.engine.geometry import disk_gap  # noqa: E402
from fortyk.engine.movement import FormationMove, MoveKind, apply_translation, check_model_positions, move_distance, pose_disk  # noqa: E402
from fortyk.engine.transport import (  # noqa: E402
    _parse_capacity, capacity_of, check_disembark_positions, disembark_candidates, embark_error, scout_distance,
)
from fortyk.toy import new_toy_state  # noqa: E402

HAS_DATA = (default_raw_dir() / "Datasheets.csv").exists()


def place(unit, x, y):
    c = unit.centroid
    apply_translation(unit, x - c[0], y - c[1])
    unit.reset_turn_flags()


class CapacityParsingTests(unittest.TestCase):
    def test_keywords_exclusions_costs(self):
        c = _parse_capacity("This model has a transport capacity of 12 Emperor’s Children Infantry models (excluding TERMiNATOR and Flawless Blade models).")
        self.assertEqual((c.capacity, c.kinds, c.excluded), (12, (("Emperor’s Children", "Infantry"),), ("TERMiNATOR", "Flawless Blade")))
        c = _parse_capacity("This model has a transport capacity of 12 Adeptus Astartes Infantry models. Each Jump Pack, Wulfen, Gravis or Terminator model "
                            "takes up the space of 2 models and each Centurion model takes up the space of 3 models.")
        self.assertEqual(dict(c.costs), {"Jump Pack": 2, "Wulfen": 2, "Gravis": 2, "Terminator": 2, "Centurion": 3})
        c = _parse_capacity("This model has a transport capacity of 6 models. It can only transport Scout Squad, Scout Sniper Squad and Sergeant Telion models.")
        self.assertEqual((c.capacity, c.kinds, c.only), (6, (), ("Scout Squad", "Scout Sniper Squad", "Sergeant Telion")))
        c = _parse_capacity("This model has a transport capacity of 7 Tacticus or Phobos Infantry models. It cannot transport Jump Pack models.")
        self.assertEqual((c.kinds, c.excluded), ((("Tacticus", "Infantry"), ("Phobos", "Infantry")), ("Jump Pack",)))
        c = _parse_capacity("This model has a transport capacity of 12 ADEPTUS ASTARTES INFANTRY models. It cannot transport JUMP PACK, WULFEN, PHOBOS, GRAVIS, "
                            "CENTURION, TERMINATOR or TACTICUS models (excluding TACTICUS CHARACTER models that began the battle attached to a non-TACTICUS unit).")
        self.assertIn("TACTICUS", c.excluded)
        self.assertNotIn("TACTICUS CHARACTER", c.excluded)
        self.assertTrue(c.unparsed)


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def new(self, attacker=None, defender=None, seed=3):
        rosters = {
            "attacker": attacker or (UnitSpec("EC1", "Infractors", 5, faction="EC"), UnitSpec("CR1", "Chaos Rhino", 1, faction="EC")),
            "defender": defender or (UnitSpec("SM1", "Intercessor Squad", 5, faction="SM"),),
        }
        return new_toy_state(self.cat, seed=seed, rosters=rosters)

    def test_rhino_is_a_rectangle_hull(self):
        s = self.new()
        rhino = s.units["CR1"].models[0]
        self.assertEqual(rhino.radius, 0.0)
        self.assertAlmostEqual(rhino.hx * 2 * 25.4, 117, places=3)
        self.assertAlmostEqual(rhino.angle, math.pi / 2)
        # V11 (03.01, 17.02) : pivoter autour de l'axe central ne compte pas dans la distance
        start = rhino.disk
        self.assertAlmostEqual(move_distance(start, start._replace(a=start.a + math.pi / 2)), 0.0)
        self.assertAlmostEqual(move_distance(start, start._replace(x=start.x + 3, a=start.a + 1.0)), 3.0)

    def test_capacity_and_exclusions(self):
        s = self.new(attacker=(UnitSpec("EC1", "Infractors", 5, faction="EC"), UnitSpec("CR1", "Chaos Rhino", 1, faction="EC"),
                               UnitSpec("FB1", "Flawless Blades", 3, faction="EC"), UnitSpec("EC2", "Infractors", 10, faction="EC")))
        cr = s.units["CR1"]
        self.assertIsNone(embark_error(s, s.units["EC1"], cr))
        self.assertIn("Flawless Blade", embark_error(s, s.units["FB1"], cr))
        s.units["EC2"].embarked_in = "CR1"  # 10 places prises
        self.assertIn("plus assez de place", embark_error(s, s.units["EC1"], cr))

    def test_start_embarked_then_disembark_before_transport_moves(self):
        s = self.new()
        eng = Engine()
        eng.start(s, first_player="attacker")
        # déploiement : le Rhino d'abord, puis les Infractors à bord
        while s.phase == "deployment":
            d = eng.decision(s)
            if d.kind == "select_unit":
                eng.step(s, next(o for o in d.options if isinstance(o, SelectUnitAction) and o.unit_id in ("CR1", "SM1")) if any(
                    isinstance(o, SelectUnitAction) and o.unit_id in ("CR1", "SM1") for o in d.options) else d.options[0])
            elif d.kind == "deploy" and d.unit_id == "EC1":
                emb = [o for o in d.options if isinstance(o, EmbarkAction)]
                self.assertEqual([o.transport_id for o in emb], ["CR1"])
                eng.step(s, emb[0])
            elif d.kind == "deploy":
                eng.step(s, d.options[len(d.options) // 2])
            else:
                eng.step(s, d.options[0])
            if s.battle_round >= 1:
                break
        ec1 = s.units["EC1"]
        self.assertEqual(ec1.embarked_in, "CR1")
        self.assertNotIn(ec1, s.units_of("attacker"))
        self.assertEqual(s.passengers("CR1"), [ec1])
        # tour de l'attaquant : les Infractors débarquent avant que le Rhino ne bouge
        d = eng.decision(s)
        while not (d.kind == "select_unit" and d.phase == "movement"):
            eng.step(s, d.options[0])
            d = eng.decision(s)
        self.assertIn(SelectUnitAction("EC1"), d.options)
        eng.step(s, SelectUnitAction("EC1"))
        d = eng.decision(s)
        self.assertEqual(d.kind, "disembark")
        spots = [o for o in d.options if getattr(o, "x", 0) is not None]
        self.assertTrue(any(isinstance(o, DisembarkModelsAction) for o in spots))  # couronnes autour de la coque
        self.assertTrue(any(isinstance(o, DisembarkAction) for o in spots))  # formations
        eng.step(s, spots[0])
        self.assertIsNone(ec1.embarked_in)
        rhino = s.units["CR1"].models[0].disk
        for m in ec1.alive_models:  # entièrement à 3" de la coque
            self.assertLessEqual(disk_gap(m.disk, rhino) + 2 * m.radius, 3.0 + 1e-6)
        d = eng.decision(s)
        self.assertEqual((d.kind, d.unit_id), ("move", "EC1"))  # le Rhino n'a pas bougé : l'unité agit normalement
        self.assertFalse(ec1.no_charge)

    def test_disembark_after_transport_moved_blocks_move_and_charge(self):
        s = self.new()
        place(s.units["CR1"], 22, 10)
        place(s.units["SM1"], 22, 50)
        s.units["EC1"].embarked_in = "CR1"
        eng = Engine()
        eng.stop_at = {"end_turn"}  # les drapeaux « ce tour » sont remis à zéro au tour suivant
        eng.start_at(s, "movement", "attacker")
        eng.step(s, SelectUnitAction("CR1"))
        d = eng.decision(s)
        fwd = next(o for o in d.options if isinstance(o, MoveAction) and o.move.kind == MoveKind.NORMAL and o.move.dy > 5)
        eng.step(s, fwd)
        self.assertFalse(s.units["CR1"].remained_stationary)
        eng.step(s, SelectUnitAction("EC1"))
        d = eng.decision(s)
        self.assertIn("a bougé", d.note)
        eng.step(s, next(o for o in d.options if getattr(o, "x", 0) is not None))
        ec1 = s.units["EC1"]
        self.assertTrue(ec1.no_charge)
        self.assertIsNotNone(eng.charge_ineligibility(s, ec1))
        after = s.log[next(i for i, l in enumerate(s.log) if "débarquement rapide" in l) + 1:]
        self.assertFalse(any(l.startswith("Mouvement : Infractors") for l in after))  # ne bouge plus
        self.assertTrue(any("EC1 : a débarqué ce tour" in l for l in after))  # ni ne charge

    def test_no_disembark_after_transport_advanced(self):
        from fortyk.engine.actions import DeclareAdvanceAction

        s = self.new(attacker=(UnitSpec("EC1", "Infractors", 5, faction="EC"), UnitSpec("CR1", "Chaos Rhino", 1, faction="EC"),
                               UnitSpec("TO1", "Tormentors", 5, faction="EC")))
        place(s.units["CR1"], 22, 10)
        place(s.units["TO1"], 8, 10)
        place(s.units["SM1"], 22, 50)
        s.units["EC1"].embarked_in = "CR1"
        eng = Engine()
        eng.start_at(s, "movement", "attacker")
        eng.step(s, SelectUnitAction("CR1"))
        eng.step(s, DeclareAdvanceAction("CR1"))
        d = eng.decision(s)
        eng.step(s, next(o for o in d.options if o.move.kind == MoveKind.NORMAL))
        d = eng.decision(s)
        self.assertNotIn(SelectUnitAction("EC1"), d.options)
        self.assertIn("Advance", d.ineligible["EC1"])

    def test_embark_after_moving_next_to_transport(self):
        s = self.new()
        place(s.units["CR1"], 22, 10)
        place(s.units["EC1"], 22, 16)
        place(s.units["SM1"], 22, 50)
        eng = Engine()
        eng.start_at(s, "movement", "attacker")
        eng.step(s, SelectUnitAction("EC1"))
        d = eng.decision(s)
        back = min((o for o in d.options if isinstance(o, MoveAction) and o.move.kind == MoveKind.NORMAL),
                   key=lambda o: min(disk_gap(m.disk.translated(o.move.dx, o.move.dy), s.units["CR1"].models[0].disk) for m in s.units["EC1"].models))
        eng.step(s, back)
        d = eng.decision(s)
        self.assertEqual(d.kind, "embark")
        eng.step(s, EmbarkAction("EC1", "CR1"))
        self.assertEqual(s.units["EC1"].embarked_in, "CR1")
        self.assertNotIn("EC1", [u.id for u in s.units_of("attacker")])

    def test_destroyed_transport_emergency_disembark(self):
        s = self.new()
        place(s.units["CR1"], 22, 30)
        place(s.units["SM1"], 22, 50)
        ec1 = s.units["EC1"]
        ec1.embarked_in = "CR1"
        eng = Engine()
        eng.start_at(s, "shooting", "defender")
        rhino = s.units["CR1"]
        for m in rhino.models:
            m.alive, m.wounds = False, 0
        eng.on_unit_destroyed(s, rhino)
        self.assertIsNone(ec1.embarked_in)
        self.assertTrue(ec1.no_charge)
        self.assertTrue(ec1.battle_shocked)  # V11 18.05
        wreck = rhino.models[0].disk
        for m in ec1.alive_models:
            self.assertLessEqual(disk_gap(m.disk, wreck) + 2 * m.radius, 6.0 + 1e-6)
        line = next(l for l in s.log if "Transport détruit" in l)
        self.assertIn("jets de danger", line)
        ev = next(e for e in s.events if e["kind"] == "emergency_disembark")
        self.assertEqual(len(ev["rolls"]), 5)  # un jet de danger par figurine
        self.assertEqual(ev["mortal_wounds"], sum(1 for r in ev["rolls"] if r <= 2))

    def test_combat_disembark_when_no_room_at_3(self):
        import dataclasses

        from fortyk.engine.rules import DEFAULT_RULES

        rules = dataclasses.replace(DEFAULT_RULES, disembark_range_in=0.5)  # aucune figurine ne tient « entièrement à 0,5" »
        rosters = {"attacker": (UnitSpec("EC1", "Infractors", 5, faction="EC"), UnitSpec("CR1", "Chaos Rhino", 1, faction="EC")),
                   "defender": (UnitSpec("SM1", "Intercessor Squad", 5, faction="SM"),)}
        s = new_toy_state(self.cat, seed=3, rosters=rosters, rules=rules)
        place(s.units["CR1"], 22, 20)
        place(s.units["SM1"], 22, 50)
        ec1 = s.units["EC1"]
        ec1.embarked_in = "CR1"
        eng = Engine()
        eng.start_at(s, "movement", "attacker")
        eng.step(s, SelectUnitAction("EC1"))
        d = eng.decision(s)
        self.assertEqual((d.kind, d.phase, d.max_distance), ("disembark", "combat", 6.0))
        spots = [o for o in d.options if getattr(o, "x", 0) is not None]
        self.assertTrue(spots)
        eng.step(s, spots[0])
        self.assertTrue(ec1.battle_shocked)
        self.assertTrue(ec1.no_charge)
        self.assertTrue(any("débarquement de combat" in l and "jets de danger" in l for l in s.log))

    def test_rapid_disembark_after_transport_moved(self):
        s = self.new()
        place(s.units["CR1"], 22, 10)
        place(s.units["SM1"], 22, 50)
        s.units["EC1"].embarked_in = "CR1"
        eng = Engine()
        eng.stop_at = {"end_turn"}
        eng.start_at(s, "movement", "attacker")
        eng.step(s, SelectUnitAction("CR1"))
        d = eng.decision(s)
        eng.step(s, next(o for o in d.options if isinstance(o, MoveAction) and o.move.kind == MoveKind.NORMAL))
        eng.step(s, SelectUnitAction("EC1"))
        d = eng.decision(s)
        self.assertEqual((d.phase, d.max_distance), ("rapid", 3.0))
        eng.step(s, next(o for o in d.options if getattr(o, "x", 0) is not None))
        self.assertFalse(s.units["EC1"].battle_shocked)
        self.assertTrue(s.units["EC1"].no_charge)

    def test_ring_placements_hug_the_hull_and_are_legal(self):
        from fortyk.engine.transport import ring_placements

        s = self.new(attacker=(UnitSpec("EC1", "Infractors", 10, faction="EC"), UnitSpec("CR1", "Chaos Rhino", 1, faction="EC")))
        place(s.units["CR1"], 22, 30)
        place(s.units["SM1"], 22, 50)
        ec1 = s.units["EC1"]
        ec1.embarked_in = "CR1"
        rings = ring_placements(s, ec1, s.units["CR1"])
        self.assertGreaterEqual(len(rings), 4)
        for pl in rings:
            self.assertIsNone(check_disembark_positions(s, ec1, s.units["CR1"], pl))

    def test_disembark_candidates_are_legal(self):
        s = self.new()
        place(s.units["CR1"], 22, 30)
        place(s.units["SM1"], 22, 37)  # ennemi tout près : certaines places sont interdites
        ec1 = s.units["EC1"]
        ec1.embarked_in = "CR1"
        cands = disembark_candidates(s, ec1, s.units["CR1"], limit=10**6)
        self.assertTrue(cands)
        c = ec1.centroid
        for x, y in cands[::7]:
            pos = {m.id: (m.x - c[0] + x, m.y - c[1] + y) for m in ec1.alive_models}
            self.assertIsNone(check_disembark_positions(s, ec1, s.units["CR1"], pos), (x, y))


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class ScoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def test_scout_move_before_round_one(self):
        rosters = {"attacker": (UnitSpec("EC1", "Infractors", 5, faction="EC"),), "defender": (UnitSpec("SM1", "Intercessor Squad", 5, faction="SM"),)}
        s = new_toy_state(self.cat, seed=4, rosters=rosters)
        eng = Engine()
        eng.start(s, first_player="attacker")
        while s.phase in ("deployment", "scouts") and eng.decision(s).kind != "scout":
            d = eng.decision(s)
            eng.step(s, d.options[len(d.options) // 2] if d.kind == "deploy" else d.options[0])
        d = eng.decision(s)
        self.assertEqual((d.kind, d.unit_id, d.max_distance), ("scout", "EC1", 6.0))
        self.assertIn('8"', d.note)
        self.assertEqual(s.battle_round, 0)
        moves = [o for o in d.options if o.move.kind == MoveKind.NORMAL]
        self.assertTrue(moves)
        ec1 = s.units["EC1"]
        before = {m.id: m.disk for m in ec1.alive_models}
        eng.step(s, max(moves, key=lambda o: o.move.dy))
        for m in ec1.alive_models:
            self.assertLessEqual(move_distance(before[m.id], m.disk), 6.0 + 1e-6)
            self.assertTrue(all(disk_gap(m.disk, e.disk) > 8.0 for e in s.enemy_models("attacker")))  # V11 24.32 : plus de 8"
        self.assertEqual(s.battle_round, 1)

    def test_dedicated_transport_gets_scouts_when_all_passengers_have_it(self):
        rosters = {"attacker": (UnitSpec("EC1", "Infractors", 5, faction="EC"), UnitSpec("CR1", "Chaos Rhino", 1, faction="EC"),
                                UnitSpec("TO1", "Tormentors", 5, faction="EC")),
                   "defender": (UnitSpec("SM1", "Intercessor Squad", 5, faction="SM"),)}
        s = new_toy_state(self.cat, seed=4, rosters=rosters)
        place(s.units["CR1"], 22, 8)
        place(s.units["SM1"], 22, 52)
        self.assertIsNone(scout_distance(s, s.units["CR1"]))  # vide
        s.units["EC1"].embarked_in = "CR1"
        self.assertEqual(scout_distance(s, s.units["CR1"]), 6.0)
        s.units["TO1"].embarked_in = "CR1"  # les Tormentors n'ont pas Scouts
        self.assertIsNone(scout_distance(s, s.units["CR1"]))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class V11CoreRulesTests(unittest.TestCase):
    """Points des règles de base V11 (PDF) repris dans le moteur."""

    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def test_battle_shock_persists_until_passed(self):
        rosters = {"attacker": (UnitSpec("EC1", "Infractors", 5, faction="EC"),), "defender": (UnitSpec("SM1", "Intercessor Squad", 5, faction="SM"),)}
        s = new_toy_state(self.cat, seed=1, rosters=rosters)
        place(s.units["EC1"], 22, 10)
        place(s.units["SM1"], 22, 50)
        s.units["EC1"].battle_shocked = True  # effectif complet, mais battle-shocked : un test est dû (08.03)
        eng = Engine()
        eng.start_at(s, "command", "attacker")
        ev = [e for e in s.events if e["kind"] == "battle_shock" and e["unit"] == "EC1"]
        self.assertEqual(len(ev), 1)
        self.assertEqual(s.units["EC1"].battle_shocked, not ev[0]["passed"])

    def test_close_quarters_is_pistol(self):
        from fortyk.engine.combat import is_close_quarters

        cq = [w for ds in self.cat for w in ds.weapons if w.has("close-quarters")]
        self.assertTrue(cq)
        self.assertTrue(all(is_close_quarters(w) for w in cq))
