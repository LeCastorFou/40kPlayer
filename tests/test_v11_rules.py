"""Règles de base V11 (PDF) : mouvement, charge multi-cibles, combat, allocation, objectifs-décors,
tir indirect, cohérence de fin de tour, points de commandement."""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import default_raw_dir, load_catalog  # noqa: E402
from fortyk.data.parse import DiceExpr  # noqa: E402
from fortyk.engine import Engine  # noqa: E402
from fortyk.engine.army import TOY_ROSTERS_INFANTRY, UnitSpec, build_unit  # noqa: E402
from fortyk.engine.attack import AttackProfile  # noqa: E402
from fortyk.engine.combat import _resolve_profiles, allocation_groups, build_shooting_profiles, shooting_targets  # noqa: E402
from fortyk.engine.geometry import disk_gap  # noqa: E402
from fortyk.engine.movement import (  # noqa: E402
    MoveKind, apply_translation, auto_charge_move, check_charge_positions, check_model_positions, consolidate, pile_in,
)
from fortyk.toy import new_toy_state  # noqa: E402

HAS_DATA = (default_raw_dir() / "Datasheets.csv").exists()


def place(unit, x, y):
    c = unit.centroid
    apply_translation(unit, x - c[0], y - c[1])
    unit.reset_turn_flags()


def park_all(s):
    """Tout le monde dans un coin, loin de tout ; chaque test place ensuite ses acteurs."""
    spots = [(3.0, 57.0), (9.0, 57.0), (15.0, 57.0), (3.0, 3.0), (9.0, 3.0), (41.0, 3.0)]
    for u, (x, y) in zip(s.units.values(), spots):
        place(u, x, y)


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class MovementV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        self.s = new_toy_state(self.cat, seed=1)
        park_all(self.s)
        self.sm1, self.sm2, self.sm3, self.ec1, self.ec3 = (self.s.units[k] for k in ("SM1", "SM2", "SM3", "EC1", "EC3"))

    def test_models_move_through_friends_not_enemies(self):
        """03.01 : on traverse les figurines amies, jamais le socle d'une figurine ennemie."""
        s = self.s
        place(self.sm1, 4.5, 26.0)  # zone dégagée à l'ouest (x 1–7,5 ; y 18–27,5)
        place(self.sm2, 4.5, 20.0)
        # SM2 en travers du chemin, sans contact : sa rangée du haut 3" au nord de celle de SM1
        apply_translation(self.sm2, 0.0, min(m.y for m in self.sm1.alive_models) - 3.0 - min(m.y for m in self.sm2.alive_models))
        self.assertGreater(self.sm1.min_gap_to(self.sm2), 0.0)
        pos = {m.id: (m.x, m.y - 6.0) for m in self.sm1.alive_models}  # 6" vers le nord, à travers SM2
        self.assertIsNone(check_model_positions(s, self.sm1, pos, MoveKind.NORMAL, 6.0))
        # même trajet à travers une unité ennemie : refusé
        c = self.sm2.centroid
        place(self.sm2, 3.0, 57.0)
        place(self.ec1, *c)
        err = check_model_positions(s, self.sm1, pos, MoveKind.NORMAL, 6.0)
        self.assertIn("traverse", err)

    def test_pass_within_engagement_range_but_end_outside(self):
        """09.05 (V11) : pendant un mouvement normal on peut passer à portée d'engagement ; il faut finir hors."""
        s = self.s
        place(self.ec3, 20.0, 22.5)  # Daemon Prince (socle r 1,18")
        place(self.sm3, 15.0, 22.3)  # Redemptor à l'ouest…
        m = self.sm3.models[0]
        # … qui passe au sud du prince à 0,5" de son socle, et finit à l'est
        pos = {m.id: (25.0, 22.3)}
        err = check_model_positions(s, self.sm3, pos, MoveKind.NORMAL, 10.0)
        self.assertIsNotNone(err)  # ligne droite : traverse le socle du prince (M/V contre M/V)
        ok_y = self.ec3.models[0].y + self.ec3.models[0].radius + m.radius + 0.5
        place(self.sm3, 15.0, ok_y)
        self.assertIsNone(check_model_positions(s, self.sm3, {m.id: (25.0, ok_y)}, MoveKind.NORMAL, 10.0))
        # finir à 0,5" (portée d'engagement) : refusé
        err = check_model_positions(s, self.sm3, {m.id: (self.ec3.models[0].x, ok_y)}, MoveKind.NORMAL, 10.0)
        self.assertIn("engagement", err)

    def test_vehicle_passes_over_infantry_in_normal_move_only(self):
        """17.01 : un MONSTER / VEHICLE passe par-dessus les figurines ennemies non M/V en mouvement
        normal / Advance, pas en Fall Back."""
        s = self.s
        place(self.ec1, 22.0, 22.5)
        m = self.sm3.models[0]
        er = s.rules.engagement_range_in
        top = min(x.y - x.radius for x in self.ec1.alive_models)
        bottom = max(x.y + x.radius for x in self.ec1.alive_models)
        start_y, end_y = top - er - m.radius - 0.2, bottom + er + m.radius + 0.2
        reach = end_y - start_y + 0.1  # ~11" : un Advance
        self.sm3.models[0].move_to(22.0, start_y)
        pos = {m.id: (22.0, end_y)}
        self.assertIsNone(check_model_positions(s, self.sm3, pos, MoveKind.ADVANCE, reach))
        self.assertIsNotNone(check_model_positions(s, self.sm3, pos, MoveKind.FALL_BACK, reach))
        # l'infanterie, elle, ne traverse pas
        place(self.sm3, 15.0, 57.0)
        place(self.sm1, 22.0, 0.0)
        dy = start_y - max(x.y for x in self.sm1.alive_models)
        apply_translation(self.sm1, 0.0, dy)
        vec = end_y - min(x.y for x in self.sm1.alive_models)
        pos = {x.id: (x.x, x.y + vec) for x in self.sm1.alive_models}
        self.assertIn("traverse", check_model_positions(s, self.sm1, pos, MoveKind.ADVANCE, vec + 0.1))

    def test_desperate_escape_goes_through_and_costs_hazard_rolls(self):
        """09.07 : Desperate Escape — on traverse les ennemis, un jet de danger par figurine avant de
        bouger, puis un test de battle-shock si l'unité ne l'était pas déjà."""
        s = self.s
        place(self.ec1, 30.0, 5.0)  # zone dégagée du nord-est
        place(self.sm1, 30.0, 0.0)
        ec_top = min(x.y - x.radius for x in self.ec1.alive_models)
        apply_translation(self.sm1, 0.0, ec_top - 0.5 - max(x.y + x.radius for x in self.sm1.alive_models))  # à 0,5" : engagée
        self.assertTrue(s.is_engaged(self.sm1))
        ec_bottom = max(x.y + x.radius for x in self.ec1.alive_models)
        dy = ec_bottom + s.rules.engagement_range_in + 0.2 - min(x.y - x.radius for x in self.sm1.alive_models)  # de l'autre côté d'EC1, hors portée (2")
        pos = {x.id: (x.x, x.y + dy) for x in self.sm1.alive_models}  # à travers EC1, vers le sud
        self.assertIn("traverse", check_model_positions(s, self.sm1, pos, MoveKind.FALL_BACK, dy + 0.1))
        self.assertIsNone(check_model_positions(s, self.sm1, pos, MoveKind.DESPERATE, dy + 0.1))
        eng = Engine()
        s.rng = random.Random(4)
        before = sum(x.wounds for x in self.sm1.alive_models)
        rolls, mw, lost = eng.hazard_rolls(s, self.sm1, len(self.sm1.alive_models))
        self.assertEqual(len(rolls), 5)
        self.assertEqual(mw, sum(1 for r in rolls if r in (1, 2)))  # 1-2 : raté, 1 BM (infanterie)
        self.assertEqual(before - sum(x.wounds for x in self.sm1.alive_models), mw)


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class ChargeAndFightV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        self.s = new_toy_state(self.cat, seed=1, rosters=TOY_ROSTERS_INFANTRY)
        park_all(self.s)
        self.sm1, self.sm2, self.ec1, self.ec2 = (self.s.units[k] for k in ("SM1", "SM2", "EC1", "EC2"))

    def test_multi_target_charge_engages_every_target(self):
        """11.04 : charge sur deux unités — l'unité finit engagée (2") avec chacune, une figurine au
        moins socle à socle (validé avec Valentin), et jamais engagée avec une non-cible."""
        s = self.s
        place(self.ec1, 30.0, 3.0)  # deux unités l'une au-dessus de l'autre, à ~1,2" d'écart
        place(self.ec2, 30.0, 7.0)
        place(self.sm1, 39.0, 5.0)  # à l'est des deux
        ok, msg = auto_charge_move(s, self.sm1, [self.ec1, self.ec2], 12)
        self.assertTrue(ok, msg)
        er = s.rules.engagement_range_in
        self.assertTrue(self.sm1.in_engagement_range_of(self.ec1, s.rules) and self.sm1.in_engagement_range_of(self.ec2, s.rules))
        self.assertLessEqual(min(self.sm1.min_gap_to(self.ec1), self.sm1.min_gap_to(self.ec2)), 0.05)  # socle à socle
        self.assertLessEqual(self.sm1.min_gap_to(self.ec2), er + 1e-6)

    def test_charge_needs_base_contact(self):
        s = self.s
        place(self.ec1, 30.0, 5.0)
        place(self.sm1, 39.0, 5.0)
        gap = self.sm1.min_gap_to(self.ec1)
        # on avance toute l'unité pour finir à 1,5" : engagée mais pas socle à socle → charge invalide
        dx = -(gap - 1.5)
        pos = {m.id: (m.x + dx, m.y) for m in self.sm1.alive_models}
        err = check_charge_positions(s, self.sm1, [self.ec1], pos, 12)
        self.assertIsNotNone(err)
        self.assertIn("socle", err)

    def test_overrun_pile_in_and_objective_consolidation(self):
        """12.03 / 12.08 : pile-in vers l'ennemi le plus proche à 5" ; consolidation vers un objectif-décor à 3"."""
        s = self.s
        place(self.ec1, 30.0, 3.0)
        place(self.sm1, 30.0, 0.0)
        apply_translation(self.sm1, 0.0, max(x.y + x.radius for x in self.ec1.alive_models) + 2.5 - min(x.y - x.radius for x in self.sm1.alive_models))
        self.assertTrue(2.0 < self.sm1.min_gap_to(self.ec1) <= 5.0)  # pas engagée, mais à 5"
        self.assertFalse(s.is_engaged(self.sm1))
        self.assertTrue(pile_in(s, self.sm1))
        self.assertTrue(self.sm1.in_engagement_range_of(self.ec1, s.rules))
        # consolidation : plus d'ennemi à 3", l'objectif home de l'attaquant (ruine T1, x ≤ 21,5) à ~2"
        place(self.ec1, 3.0, 57.0)
        place(self.sm1, 25.5, 11.0)
        home = next(o for o in s.layout.objectives if o.id == "home_attacker")
        self.assertFalse(s.unit_in_objective_range(self.sm1, home))
        self.assertLessEqual(s.unit_objective_distance(self.sm1, home), 3.0)
        mode, new = consolidate(s, self.sm1)
        self.assertEqual((mode, new), ("objective", []))
        self.assertTrue(s.unit_in_objective_range(self.sm1, home))


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class AllocationV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        self.s = new_toy_state(self.cat, seed=1, rosters=TOY_ROSTERS_INFANTRY)
        park_all(self.s)
        self.sm1, self.ec1 = self.s.units["SM1"], self.s.units["EC1"]
        # SM1 menée par un Captain (personnage) : la figurine du personnage rejoint l'unité
        cap = build_unit(self.cat, UnitSpec("X", "Captain", 1, faction="SM"), "attacker")
        leader = cap.models[0]
        leader.is_leader, leader.unit_id, leader.id = True, "SM1", "SM1.6"
        self.sm1.models.insert(0, leader)
        self.sm1.leaders = (cap.datasheet,)
        self.leader = leader
        place(self.sm1, 22.0, 22.5)
        place(self.ec1, 22.0, 20.0)

    def profile(self, n, damage=1, **kw):
        return AttackProfile(attacks=DiceExpr.fixed(1), skill=None, strength=12, ap=-6, damage=DiceExpr.fixed(damage), count=n, **kw)

    def test_groups_order(self):
        groups = allocation_groups(self.sm1)
        self.assertEqual([len(g) for g in groups], [5, 1])  # gardes du corps, puis le personnage en dernier
        self.assertIs(groups[-1][0], self.leader)
        self.leader.wounds -= 1  # un personnage blessé reste après les non-personnages
        self.assertIs(allocation_groups(self.sm1)[-1][0], self.leader)

    def test_characters_last_unless_precision(self):
        s = self.s
        s.rng = random.Random(7)
        r = _resolve_profiles(s, self.ec1, self.sm1, [self.profile(3)], "shooting")
        self.assertGreater(r.unsaved, 0)
        self.assertEqual(self.leader.wounds, self.leader.profile.wounds)  # le Captain n'a rien pris
        # [PRECISION] : l'attaquant choisit le groupe du personnage visible
        s.rng = random.Random(7)
        self.leader.wounds = 1
        r = _resolve_profiles(s, self.ec1, self.sm1, [self.profile(10, precision=True)], "shooting")
        self.assertFalse(self.leader.alive)
        self.assertIn("Precision", r.details[0])

    def test_devastating_wounds_hit_one_model_per_critical(self):
        """24.10 : BM égales à D, sur une figurine au plus par blessure critique (l'excédent est perdu)."""
        s = self.s
        s.rng = random.Random(3)
        p = AttackProfile(attacks=DiceExpr.fixed(1), skill=None, strength=4, ap=0, damage=DiceExpr.fixed(3), count=4,
                          devastating_wounds=True, critical_wound_on=2)
        r = _resolve_profiles(s, self.ec1, self.sm1, [p], "shooting")
        self.assertGreater(r.models_slain, 0)
        self.assertEqual(r.damage, 2 * r.models_slain)  # Intercessors 2 PV : 3 BM → 1 figurine, 1 BM perdue
        self.assertEqual(self.leader.wounds, self.leader.profile.wounds)


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class ObjectivesAndTurnV11Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        self.s = new_toy_state(self.cat, seed=1, rosters=TOY_ROSTERS_INFANTRY)
        park_all(self.s)
        self.sm1, self.ec1 = self.s.units["SM1"], self.s.units["EC1"]

    def test_terrain_objective_is_the_footprint(self):
        """14.01 : l'objectif est l'empreinte de décor désignée par la carte ; un orteil dedans suffit,
        la distance au pion ne compte plus."""
        s = self.s
        central = next(o for o in s.layout.objectives if o.id == "central")
        self.assertEqual(central.terrain_id, "T9")
        x0, y0, x1, y1 = central.rect
        m = self.sm1.models[0]
        m.move_to(x1 + m.radius - 0.05, y1 - 0.5)  # touche le bord est de T9, à ~6" du pion
        self.assertGreater(disk_gap(m.disk, m.disk._replace(x=central.x, y=central.y, r=0.0)), 3.0)
        self.assertTrue(s.model_in_objective_range(m, central))
        m.move_to(x1 + m.radius + 0.1, y1 - 0.5)  # 0,1" dehors
        self.assertFalse(s.model_in_objective_range(m, central))

    def test_cp_and_turn_counter_each_command_phase(self):
        s = self.s
        place(self.sm1, 22.0, 10.0)
        place(self.ec1, 22.0, 50.0)
        eng = Engine()
        eng.start_at(s, "command", "attacker")
        self.assertEqual(s.cp, {"attacker": 1, "defender": 1})  # 08.02 : les deux joueurs gagnent 1 CP
        self.assertEqual(s.turn_counter, 1)

    def test_coherency_restored_at_end_of_turn(self):
        """03.03 : en fin de tour, une unité hors cohérence retire des figurines jusqu'à l'être."""
        s = self.s
        place(self.sm1, 22.0, 22.5)
        straggler = self.sm1.models[-1]
        straggler.move_to(30.0, 22.5)
        self.assertFalse(self.sm1.coherency_ok(s.rules))
        Engine()._restore_coherency(s)
        self.assertTrue(self.sm1.coherency_ok(s.rules))
        self.assertFalse(straggler.alive)
        self.assertEqual(self.sm1.strength, 4)


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class IndirectFireTests(unittest.TestCase):
    """10.07 : tir indirect — cible non visible, couvert, pas de relance, 1-5 rate (1-3 si l'unité est
    restée immobile et la cible vue par une unité amie)."""

    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def test_indirect_fire_rules(self):
        rosters = {"attacker": (UnitSpec("WW", "Whirlwind", 1, faction="SM"), UnitSpec("SM1", "Intercessor Squad", 5, faction="SM")),
                   "defender": (UnitSpec("EC1", "Infractors", 5, faction="EC"),)}
        try:
            s = new_toy_state(self.cat, seed=1, rosters=rosters)
        except Exception as exc:  # pragma: no cover - fiche absente des CSV
            self.skipTest(f"Whirlwind indisponible : {exc}")
        ww, sm1, ec1 = s.units["WW"], s.units["SM1"], s.units["EC1"]
        if not any(w.has("indirect fire") for m in ww.models for w in m.weapons):
            self.skipTest("pas d'arme [INDIRECT FIRE] sur la fiche")
        place(sm1, 3.0, 57.0)
        place(ww, 22.0, 20.5)  # au nord de la ruine centrale T9 (dense, y 25–35)
        place(ec1, 22.0, 39.0)  # au sud de T9 : aucune ligne de vue depuis le Whirlwind
        self.assertFalse(any(s.los.visible(a.disk, b.disk) for a in ww.alive_models for b in ec1.alive_models))
        self.assertIn(ec1, shooting_targets(s, ww))
        profiles, _ = build_shooting_profiles(s, ww, ec1)
        indirect = [p for p in profiles if p.min_unmodified_hit]
        self.assertTrue(indirect)
        p = indirect[0]
        self.assertEqual(p.min_unmodified_hit, 6)  # personne ne voit la cible
        self.assertEqual(p.reroll_hits, "none")
        self.assertIn("couvert", p.label)
        # une unité amie voit la cible et le Whirlwind n'a pas bougé : 1-3 rate
        place(sm1, 22.0, 45.5)
        self.assertTrue(any(s.los.visible(a.disk, b.disk) for a in sm1.alive_models for b in ec1.alive_models))
        ww.remained_stationary = True
        p = [p for p in build_shooting_profiles(s, ww, ec1)[0] if p.min_unmodified_hit][0]
        self.assertEqual(p.min_unmodified_hit, 4)
        ww.remained_stationary = False
        p = [p for p in build_shooting_profiles(s, ww, ec1)[0] if p.min_unmodified_hit][0]
        self.assertEqual(p.min_unmodified_hit, 6)


if __name__ == "__main__":
    unittest.main()
