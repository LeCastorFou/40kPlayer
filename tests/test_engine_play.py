"""Unités, mouvement, combat depuis le plateau, et parties complètes bot contre bot."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.agents import RandomAgent  # noqa: E402
from fortyk.data import default_raw_dir, load_catalog  # noqa: E402
from fortyk.engine.army import TOY_ROSTERS_INFANTRY, UnitSpec, build_unit, formation_offsets  # noqa: E402
from fortyk.engine.combat import build_melee_profiles, build_shooting_profiles, expected_shooting, shooting_targets  # noqa: E402
from fortyk.engine.game import Game  # noqa: E402
from fortyk.engine.movement import FormationMove, MoveKind, apply_translation, candidate_moves, charge_move, legal_translation  # noqa: E402
from fortyk.toy import new_toy_state  # noqa: E402

HAS_DATA = (default_raw_dir() / "Datasheets.csv").exists()


def place(unit, x, y):
    c = unit.centroid
    apply_translation(unit, x - c[0], y - c[1])
    unit.reset_turn_flags()


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class ArmyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def test_default_loadouts(self):
        sm = build_unit(self.cat, UnitSpec("A", "Intercessor Squad", 5, "SM"), "attacker")
        self.assertEqual([m.name for m in sm.models][:2], ["Intercessor Sergeant", "Intercessor"])
        for m in sm.models:
            self.assertEqual(sorted(w.name for w in m.weapons), ["Bolt pistol", "Bolt rifle", "Close combat weapon"])
        ec = build_unit(self.cat, UnitSpec("B", "Infractors", 5, "EC"), "defender")
        self.assertEqual(ec.models[0].name, "Obsessionist")
        self.assertEqual(sorted(w.name for w in ec.models[0].weapons), ["Bolt pistol", "Power sword"])
        self.assertEqual(sorted(w.name for w in ec.models[1].weapons), ["Bolt pistol", "Duelling sabre"])
        self.assertTrue(sm.coherency_ok() and ec.coherency_ok())
        self.assertEqual(sm.strength, 5)
        self.assertFalse(sm.below_half_strength)

    def test_overrides_and_formation(self):
        u = build_unit(self.cat, UnitSpec("C", "Intercessor Squad", 10, "SM", weapon_overrides={0: ("bolt pistol", "power fist")}), "attacker")
        self.assertIn("Power fist", [w.name for w in u.models[0].weapons])
        self.assertEqual(len(formation_offsets(10, 0.63)), 10)
        self.assertTrue(u.coherency_ok())

    def test_damage_allocation(self):
        u = build_unit(self.cat, UnitSpec("D", "Intercessor Squad", 5, "SM"), "attacker")
        killed = u.allocate_damage([1, 1, 2, 5])
        self.assertEqual(killed, 3)  # 1+1 tuent une figurine (blessée d'abord), 2 en tue une, 5 en tue une (excédent perdu)
        self.assertEqual(u.strength, 2)
        self.assertTrue(u.below_half_strength)
        self.assertEqual(u.allocate_mortal_wounds(3), 1)  # 3 BM : une figurine (2 PV) meurt, la suivante perd 1
        self.assertEqual(sum(m.wounds for m in u.alive_models), 1)


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class MovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        self.s = new_toy_state(self.cat, seed=3, rosters=TOY_ROSTERS_INFANTRY)
        self.sm1, self.sm2, self.ec1, self.ec2 = (self.s.units[k] for k in ("SM1", "SM2", "EC1", "EC2"))
        place(self.sm1, 10, 10)
        place(self.sm2, 35, 6)
        place(self.ec1, 10, 50)
        place(self.ec2, 35, 55)

    def test_translation_limits(self):
        s, u = self.s, self.sm1
        self.assertTrue(legal_translation(s, u, 6, 0, MoveKind.NORMAL, 6))
        self.assertFalse(legal_translation(s, u, 6.5, 0, MoveKind.NORMAL, 6))
        self.assertFalse(legal_translation(s, u, -20, 0, MoveKind.NORMAL, 25))  # hors table

    def test_engagement_blocks_normal_move(self):
        s = self.s
        place(self.ec1, 10, 16)  # ennemi à ~4" sous SM1
        self.assertFalse(legal_translation(s, self.sm1, 0, 3, MoveKind.NORMAL, 6))  # finirait à portée d'engagement
        self.assertTrue(legal_translation(s, self.sm1, 0, -3, MoveKind.NORMAL, 6))
        self.assertFalse(legal_translation(s, self.sm1, 0, 6, MoveKind.NORMAL, 6))  # traverserait l'ennemi

    def test_candidates_are_legal_and_include_advance(self):
        s = self.s
        moves = candidate_moves(s, self.sm1)
        kinds = {m.kind for m in moves}
        self.assertIn(MoveKind.STATIONARY, kinds)
        self.assertIn(MoveKind.NORMAL, kinds)
        self.assertIn(MoveKind.ADVANCE, kinds)
        self.assertNotIn(MoveKind.FALL_BACK, kinds)
        for m in moves:
            if m.kind == MoveKind.NORMAL:
                self.assertTrue(legal_translation(s, self.sm1, m.dx, m.dy, m.kind, self.sm1.move_in), m.label)

    def test_fall_back_when_engaged(self):
        s = self.s
        place(self.ec1, 10, 12.5)  # à ~1" de SM1 : engagé
        self.assertTrue(s.is_engaged(self.sm1))
        moves = candidate_moves(s, self.sm1)
        kinds = {m.kind for m in moves}
        self.assertTrue(kinds <= {MoveKind.STATIONARY, MoveKind.FALL_BACK, MoveKind.DESPERATE})  # desperate escape là où le repli ordonné bloque
        self.assertIn(MoveKind.FALL_BACK, kinds)
        fb = [m for m in moves if m.kind == MoveKind.FALL_BACK]
        self.assertTrue(fb)
        apply_translation(self.sm1, fb[0].dx, fb[0].dy)
        self.assertFalse(s.is_engaged(self.sm1))

    def test_charge(self):
        s = self.s
        place(self.ec1, 10, 18)  # ≈ 6.7" de socle à socle
        gap = self.ec1.min_gap_to(self.sm1)
        self.assertFalse(charge_move(s, self.ec1, self.sm1, roll=3))  # trop court : rien ne bouge
        self.assertAlmostEqual(self.ec1.min_gap_to(self.sm1), gap)
        self.assertTrue(charge_move(s, self.ec1, self.sm1, roll=9))
        self.assertTrue(self.ec1.in_engagement_range_of(self.sm1))
        self.assertTrue(self.ec1.coherency_ok())
        self.assertGreaterEqual(len(self.ec1.models_in_engagement_range(self.sm1)), 3)


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class CombatFromBoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        self.s = new_toy_state(self.cat, seed=5, rosters=TOY_ROSTERS_INFANTRY)
        self.sm1, self.sm2, self.ec1, self.ec2 = (self.s.units[k] for k in ("SM1", "SM2", "EC1", "EC2"))
        place(self.sm1, 2.2, 20)
        place(self.sm2, 40, 4)
        place(self.ec1, 2.2, 36)  # ≈ 14.5" plus bas, à découvert (hors de la ruine T3m)
        place(self.ec2, 40, 56)

    def test_targets_range_and_pistols(self):
        s = self.s
        self.assertEqual([t.id for t in shooting_targets(s, self.sm1)], ["EC1"])  # EC2 hors de portée (bolt rifle 24")
        self.assertEqual(shooting_targets(s, self.ec1), [])  # pistolets à 12" : trop loin
        place(self.ec1, 2.2, 30)  # ≈ 8.7"
        self.assertEqual([t.id for t in shooting_targets(s, self.ec1)], ["SM1"])

    def test_hail_of_bolts_heavy_and_cover(self):
        s = self.s
        profiles, haz = build_shooting_profiles(s, self.sm1, self.ec1)
        self.assertEqual(haz, 0)
        (p,) = profiles  # bolt rifles uniquement (meilleur lot que les pistolets)
        self.assertEqual(p.count, 5)
        self.assertEqual(p.attacks.flat, 4)  # A2 + 2 (Hail of Bolts)
        self.assertEqual(p.hit_modifier, 1)  # Heavy : unité immobile
        self.sm1.moved_in = 4.0
        (p,) = build_shooting_profiles(s, self.sm1, self.ec1)[0]
        self.assertEqual(p.hit_modifier, 0)
        # cible entièrement dans la ruine dense T3m → couvert (V11 13.08 : CT dégradée de 1, le +1 de
        # Heavy reste un modificateur de touche). Dans un décor dense = Hidden : il faut tirer de 15" ou moins.
        self.sm1.moved_in = 0.0
        place(self.ec1, 7.75, 41)
        self.assertTrue(all(s.layout.piece_at(m.position) for m in self.ec1.alive_models))
        from fortyk.engine.combat import is_hidden, shooting_ineligibility

        self.assertTrue(is_hidden(s, self.ec1))
        self.assertIn("Hidden", shooting_ineligibility(s, self.sm1) or "")  # à ~21" : inciblable
        place(self.sm1, 2.2, 28)  # ≈ 12" : ciblable
        self.sm1.moved_in = 0.0
        (p,) = build_shooting_profiles(s, self.sm1, self.ec1)[0]
        self.assertEqual(p.hit_modifier, 1)
        self.assertEqual(p.skill, 4)  # bolt rifle 3+ → 4+
        self.assertIn("couvert", p.label)
        self.assertGreater(expected_shooting(s, self.sm1, self.ec1), 0)

    def test_oath_and_excessive_assault(self):
        s = self.s
        s.oath_target = "EC1"
        (p,) = build_shooting_profiles(s, self.sm1, self.ec1)[0]
        self.assertEqual(p.reroll_hits, "fails")
        place(self.ec1, 2.2, 22.8)  # au contact
        self.assertTrue(self.ec1.in_engagement_range_of(self.sm1))
        melee = build_melee_profiles(s, self.ec1, self.sm1)
        self.assertTrue(melee)
        self.assertTrue(all(p.reroll_wounds == "ones" for p in melee))  # pas d'objectif à portée
        place(self.sm1, 22, 26.5)  # SM1 à portée de l'objectif central
        place(self.ec1, 22, 29.3)
        melee = build_melee_profiles(s, self.ec1, self.sm1)
        self.assertTrue(all(p.reroll_wounds == "fails" for p in melee))

    def test_assault_after_advance_and_engaged_pistols(self):
        s = self.s
        self.sm1.advanced = True
        (p,) = build_shooting_profiles(s, self.sm1, self.ec1)[0]
        self.assertEqual(p.label.split(" ×")[0], "Bolt rifle")  # Assault : autorisé après Advance
        place(self.ec1, 2.2, 22.8)  # engagé → pistolets seulement
        self.sm1.advanced = False
        profiles, _ = build_shooting_profiles(s, self.sm1, self.ec1)
        self.assertEqual([p.label.split(" ×")[0] for p in profiles], ["Bolt pistol"])


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class FullGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def _play(self, seed):
        state = new_toy_state(self.cat, seed=seed)
        game = Game(state, {"attacker": RandomAgent(seed=seed), "defender": RandomAgent(seed=seed + 1)})
        return state, game.play()

    def test_games_complete(self):
        for seed in (1, 2, 3):
            state, result = self._play(seed)
            self.assertEqual(result.rounds_played, 5)
            for side in ("attacker", "defender"):
                self.assertGreaterEqual(result.scores[side], 0)
                for r in range(1, 6):
                    self.assertLessEqual(state.scoreboard.round_total(side, r), 15)
            for u in state.units.values():
                for m in u.alive_models:
                    self.assertTrue(state.on_board(m.disk))
                if not u.is_destroyed:
                    self.assertTrue(u.coherency_ok(), u.name)

    def test_deterministic(self):
        _, a = self._play(7)
        _, b = self._play(7)
        self.assertEqual(a.log, b.log)
        self.assertEqual(a.scores, b.scores)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class RulesFeedbackTests(unittest.TestCase):
    """Retours de Valentin : pas de tir sur une unité engagée avec une unité amie ; raisons de charge."""

    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        self.s = new_toy_state(self.cat, seed=9, rosters=TOY_ROSTERS_INFANTRY)
        self.sm1, self.sm2, self.ec1, self.ec2 = (self.s.units[k] for k in ("SM1", "SM2", "EC1", "EC2"))
        place(self.sm1, 2.2, 20)
        place(self.sm2, 2.2, 8)
        place(self.ec1, 2.2, 22.8)  # au contact de SM1
        place(self.ec2, 40, 56)

    def test_no_shooting_at_unit_locked_with_friend(self):
        from fortyk.engine.combat import shooting_ineligibility, target_locked_in_combat

        s = self.s
        self.assertTrue(self.ec1.in_engagement_range_of(self.sm1))
        self.assertTrue(target_locked_in_combat(s, self.sm2, self.ec1))
        self.assertFalse(target_locked_in_combat(s, self.sm1, self.ec1))  # l'unité engagée elle-même : pistolets
        self.assertEqual(shooting_targets(s, self.sm2), [])  # EC1 verrouillée, EC2 hors de portée
        self.assertIn("aucune cible", shooting_ineligibility(s, self.sm2))
        self.assertEqual([t.id for t in shooting_targets(s, self.sm1)], ["EC1"])
        profiles, _ = build_shooting_profiles(s, self.sm1, self.ec1)
        self.assertEqual([p.label.split(" ×")[0] for p in profiles], ["Bolt pistol"])

    def test_charge_ineligibility_reasons(self):
        from fortyk.engine.game import Game

        s = self.s
        g = Game(s, {"attacker": RandomAgent(), "defender": RandomAgent()})
        self.assertIn("engagement", g.charge_ineligibility(self.sm1))  # déjà engagée
        self.assertIn("12\"", g.charge_ineligibility(self.sm2))  # EC1 à ~12.5" (verrouillée) — aucune cible à 12" ? on vérifie la distance
        place(self.ec2, 2.2, 14)  # à ~4.7" de SM2
        self.assertIsNone(g.charge_ineligibility(self.sm2))
        self.sm2.advanced = True
        self.assertIn("Advance", g.charge_ineligibility(self.sm2))
        self.sm2.advanced = False
        self.sm2.fell_back = True
        self.assertIn("repliée", g.charge_ineligibility(self.sm2))
        self.sm2.fell_back = False
        self.assertIsNone(g.charge_ineligibility(self.ec2))  # Thrill Seekers : peut charger même après Advance
        self.ec2.advanced = True
        self.assertIsNone(g.charge_ineligibility(self.ec2))


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class ChargeV11Tests(unittest.TestCase):
    """Charge V11 : jet d'abord, contact socle à socle requis, double 1 = échec, règle 1" / 2" / plus près."""

    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        self.s = new_toy_state(self.cat, seed=13, rosters=TOY_ROSTERS_INFANTRY)
        self.sm1, self.sm2, self.ec1, self.ec2 = (self.s.units[k] for k in ("SM1", "SM2", "EC1", "EC2"))
        place(self.sm1, 10, 10)
        place(self.sm2, 40, 4)
        place(self.ec1, 10, 17.5)  # socle à socle ≈ 5.0"
        place(self.ec2, 40, 56)

    def test_roll_semantics(self):
        from fortyk.engine.movement import charge_gap, charge_roll_succeeds

        self.assertFalse(charge_roll_succeeds(6, 6.3))  # 6 laisse 0.3" : pas de contact
        self.assertTrue(charge_roll_succeeds(7, 6.3))
        self.assertFalse(charge_roll_succeeds(2, 1.5))  # double 1 : toujours un échec
        self.assertTrue(charge_roll_succeeds(3, 2.5))
        gap = charge_gap(self.ec1, self.sm1)
        pairs = [((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5 - a.radius - b.radius for a in self.ec1.alive_models for b in self.sm1.alive_models]
        self.assertAlmostEqual(gap, min(pairs), places=6)
        self.assertGreater(gap, 4.5)

    def test_auto_charge_reaches_contact(self):
        from fortyk.engine.movement import auto_charge_move, charge_gap, check_charge_positions

        s = self.s
        gap = charge_gap(self.ec1, self.sm1)
        roll = int(gap) + 1
        before = {m.id: (m.x, m.y) for m in self.ec1.alive_models}
        self.assertTrue(auto_charge_move(s, self.ec1, self.sm1, roll)[0])
        self.assertLessEqual(charge_gap(self.ec1, self.sm1), 0.1)  # au contact
        self.assertTrue(self.ec1.coherency_ok())
        # le placement automatique respecte la validation « à la main »
        positions = {m.id: (m.x, m.y) for m in self.ec1.alive_models}
        for m in self.ec1.alive_models:
            m.move_to(*before[m.id])
        self.assertIsNone(check_charge_positions(s, self.ec1, self.sm1, positions, roll))

    def test_manual_charge_validation(self):
        from fortyk.engine.movement import check_charge_positions

        s = self.s
        roll = 8
        base = {m.id: (m.x, m.y) for m in self.ec1.alive_models}
        # personne ne bouge : pas de contact
        self.assertIn("socle à socle", check_charge_positions(s, self.ec1, self.sm1, dict(base), roll))
        # une figurine va au contact, les autres restent : elles pouvaient arriver à 1" → refus
        front = min(self.ec1.alive_models, key=lambda m: m.y)
        r = front.radius
        pos = dict(base)
        pos[front.id] = (front.x, 10 + 0.75 + 2 * r + 0.02)  # juste sous la deuxième rangée de SM1 (y = 10.75)
        err = check_charge_positions(s, self.ec1, self.sm1, pos, roll)
        self.assertIsNotNone(err)
        self.assertIn("pouvait arriver", err)
        # trop loin pour le jet
        pos2 = dict(base)
        pos2[front.id] = (front.x, front.y - roll - 1)
        self.assertIn("plus que le jet", check_charge_positions(s, self.ec1, self.sm1, pos2, roll))
        # un placement cohérent : tout le monde se translate le long du vecteur de la paire la plus proche
        a, b = min(((x, y) for x in self.ec1.alive_models for y in self.sm1.alive_models), key=lambda p: ((p[0].x - p[1].x) ** 2 + (p[0].y - p[1].y) ** 2) ** 0.5)
        d = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
        gap = d - a.radius - b.radius
        ux, uy = (b.x - a.x) / d, (b.y - a.y) / d
        pos3 = {m.id: (m.x + ux * (gap - 0.02), m.y + uy * (gap - 0.02)) for m in self.ec1.alive_models}
        # une seule figurine au contact, les autres en deuxième ligne alors qu'il reste des places à 1" : refusé
        err = check_charge_positions(s, self.ec1, self.sm1, pos3, roll)
        self.assertIsNotNone(err)
        self.assertIn("pouvait arriver à 1\"", err)
        # le placement automatique, lui, est accepté ; puis on ajoute une unité ennemie à portée d'engagement
        from fortyk.engine.movement import auto_charge_move

        saved = {m.id: (m.x, m.y) for m in self.ec1.alive_models}
        self.assertTrue(auto_charge_move(s, self.ec1, self.sm1, roll)[0])
        auto = {m.id: (m.x, m.y) for m in self.ec1.alive_models}
        for m in self.ec1.alive_models:
            m.move_to(*saved[m.id])
        self.assertIsNone(check_charge_positions(s, self.ec1, self.sm1, auto, roll))
        place(self.sm2, 15.5, 12.2)  # une autre unité ennemie au flanc de SM1 : interdit de finir à 2" d'elle
        err = check_charge_positions(s, self.ec1, self.sm1, auto, roll)
        self.assertIsNotNone(err)
        self.assertIn("SM2", err)


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class ChargePhaseFlowTests(unittest.TestCase):
    """Le déroulé complet de la phase de charge V11 avec un agent scripté (déclare, cible, placement)."""

    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def test_declare_roll_target_place(self):
        from fortyk.engine.actions import ChargeAction, DeclareChargeAction, ModelMoveAction, SelectUnitAction, EndPhaseAction, AutoChargeMoveAction
        from fortyk.engine.game import Game

        s = new_toy_state(self.cat, seed=2, rosters=TOY_ROSTERS_INFANTRY)
        sm1, sm2, ec1, ec2 = (s.units[k] for k in ("SM1", "SM2", "EC1", "EC2"))
        place(sm1, 10, 10)
        place(sm2, 40, 4)
        place(ec1, 10, 17.5)  # ≈ 4.8" socle à socle
        place(ec2, 40, 56)
        s.battle_round = 1
        s.side_to_move = "attacker"
        seen = []

        class Scripted:
            name = "script"
            interactive = True

            def out(self, msg):
                seen.append(("error", msg))

            def choose(self, state, decision):
                seen.append(decision.kind)
                if decision.kind == "select_unit":
                    return next(o for o in decision.options if isinstance(o, SelectUnitAction) and o.unit_id == "SM1")
                if decision.kind == "charge_declare":
                    return DeclareChargeAction("SM1")
                if decision.kind == "charge_target":
                    return next(o for o in decision.options if o.target_id == "EC1")
                if decision.kind == "charge_move":
                    # placement à la main : les positions du placement automatique, calculées sur une copie
                    from fortyk.engine.movement import auto_charge_move

                    saved = {m.id: (m.x, m.y) for m in sm1.alive_models}
                    self_ok = auto_charge_move(state, sm1, ec1, int(decision.max_distance))[0]
                    pos = tuple((m.id, m.x, m.y) for m in sm1.alive_models)
                    for m in sm1.alive_models:
                        m.move_to(*saved[m.id])
                    assert self_ok
                    return ModelMoveAction("SM1", "charge", pos)
                return decision.options[0]

        g = Game(s, {"attacker": Scripted(), "defender": RandomAgent()})
        s.rng.seed(5)  # jet de charge déterministe
        g.charge_phase("attacker")
        kinds = [k for k in seen if isinstance(k, str)]
        self.assertEqual(kinds[0], "select_unit")
        self.assertEqual(kinds[1], "charge_declare")
        roll_lines = [l for l in s.log if "lance 2D6" in l]
        self.assertTrue(roll_lines, s.log)
        if "charge_move" in kinds:  # jet suffisant
            self.assertTrue(sm1.charged and sm1.fights_first)
            self.assertLessEqual(sm1.min_gap_to(ec1), 0.1)
            self.assertNotIn("charge_target", kinds)  # une seule cible atteignable : pas de choix demandé
            self.assertFalse([e for e in seen if isinstance(e, tuple)], seen)
        else:
            self.assertIn("ratée", s.log[-1])
            self.assertFalse(sm1.charged)


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class HiddenTests(unittest.TestCase):
    """Retour de Valentin (capture EC2 à cheval sur T5m L et T4m D) : Hidden s'évalue figurine par
    figurine sur l'union des empreintes, et une figurine qui déborde de l'empreinte casse Hidden."""

    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        self.s = new_toy_state(self.cat, seed=1, rosters=TOY_ROSTERS_INFANTRY)
        self.sm1, self.sm2, self.ec1, self.ec2 = (self.s.units[k] for k in ("SM1", "SM2", "EC1", "EC2"))
        place(self.sm1, 18.5, 25.5)  # comme sur la capture : SM1.4 / SM1.5 dans la ruine centrale, qui voient dehors
        place(self.sm2, 35.0, 20.5)  # entièrement dans T3
        place(self.ec1, 19.0, 57.0)

    def test_light_terrain_does_not_hide(self):
        """V11 13.09 : seule une zone contenant un décor dense cache. EC2 à cheval sur la barricade
        légère T5m et la barricade dense T4m : seules les figurines qui touchent T4m sont cachées,
        les autres restent visibles → l'unité est ciblable à 21"."""
        from fortyk.engine.combat import hidden_model_ids, is_hidden, models_outside_terrain

        s = self.s
        place(self.ec2, 35.0, 44.4)
        pieces = {s.layout.piece_at(m.position).id for m in self.ec2.alive_models}
        self.assertEqual(pieces, {"T5m", "T4m"})
        self.assertEqual(sorted(hidden_model_ids(s, self.ec2)), ["EC2.3", "EC2.5"])  # au moins un orteil dans T4m (x ≥ 36)
        self.assertFalse(is_hidden(s, self.ec2))
        self.assertEqual(models_outside_terrain(s, self.ec2), ["EC2.1", "EC2.2", "EC2.4"])
        self.assertEqual([t.id for t in shooting_targets(s, self.sm1)], ["EC2"])

    def test_unit_across_two_touching_dense_pieces_is_hidden(self):
        from fortyk.engine.combat import is_hidden, models_outside_terrain, shooting_ineligibility

        s = self.s
        place(self.sm1, 22.0, 21.0)
        place(self.ec2, 23.0, 42.6)  # barricade dense T6m (17–23 × 40–42,75) et ruine dense T1m (22,5–30 × 42,5–54)
        touched = {s.layout.piece_touching(m.disk).id for m in self.ec2.alive_models}
        self.assertEqual(touched, {"T6m", "T1m"})
        self.assertIsNone(s.layout.piece_containing_disks(self.ec2.disks()))  # aucune pièce seule ne contient tout
        self.assertTrue(is_hidden(s, self.ec2))
        self.assertEqual(models_outside_terrain(s, self.ec2), [])
        # à ~19" et ~21" : inciblable par SM1 comme par SM2
        self.assertEqual(shooting_targets(s, self.sm1), [])
        self.assertEqual(shooting_targets(s, self.sm2), [])
        self.assertIn("Hidden", shooting_ineligibility(s, self.sm1))

    def test_toe_in_terrain_is_enough(self):
        """« Un orteil suffit » : la 2e rangée d'EC2 a son centre hors de T6m mais son socle la touche → Hidden."""
        from fortyk.engine.combat import is_hidden, models_outside_terrain

        s = self.s
        place(self.sm1, 22.0, 21.0)
        place(self.ec2, 20.0, 42.2)  # 2e rangée à y = 43,11 > 42,75, socle r 0,63
        self.assertFalse(s.layout.disks_wholly_in_area(self.ec2.disks()))
        self.assertTrue(all(m.y > 42.75 for m in self.ec2.alive_models[3:]))
        self.assertTrue(is_hidden(s, self.ec2))
        self.assertEqual(models_outside_terrain(s, self.ec2), [])
        self.assertEqual(shooting_targets(s, self.sm1), [])  # ~18" > 15"

    def test_model_clear_of_terrain_breaks_hidden(self):
        from fortyk.engine.combat import is_hidden, models_outside_terrain

        s = self.s
        place(self.ec2, 20.0, 43.5)  # 2e rangée à y = 44,41 : socle entièrement sous y = 42,75
        self.assertFalse(is_hidden(s, self.ec2))
        self.assertEqual(models_outside_terrain(s, self.ec2), ["EC2.4", "EC2.5"])

    def test_shooting_breaks_hidden_for_two_turns(self):
        """13.09 : une unité qui a fait des attaques à distance ce tour ou au tour précédent n'est pas cachée."""
        from fortyk.engine.combat import is_hidden

        s = self.s
        place(self.ec2, 20.0, 41.4)  # entièrement dans T6m
        s.turn_counter = 3
        self.assertTrue(is_hidden(s, self.ec2))
        self.ec2.last_shot_turn = 3  # a tiré pendant son tour
        self.assertFalse(is_hidden(s, self.ec2))
        s.turn_counter = 4  # tour adverse suivant : toujours repérée
        self.assertFalse(is_hidden(s, self.ec2))
        s.turn_counter = 5  # de nouveau son tour, sans avoir tiré au précédent
        self.assertTrue(is_hidden(s, self.ec2))

    def test_toe_in_ruin_sees_through_it(self):
        """Un socle qui touche l'empreinte d'une ruine voit à travers elle (et se fait voir)."""
        from fortyk.engine.geometry import Disk, is_visible

        s = self.s
        t9 = next(t for t in s.terrain if t.name == "T9")  # ruine dense (16,25–27,75 × 25–35)
        far = Disk(22.0, 40.0, 0.63)  # au sud de la ruine, dehors
        inside_toe = Disk(22.0, 24.5, 0.63)  # centre hors de T9 (y < 25) mais le socle touche l'empreinte
        clear = Disk(22.0, 24.2, 0.63)  # à 0,17" du bord : ne touche pas
        self.assertTrue(is_visible(inside_toe, far, s.terrain))
        self.assertTrue(is_visible(far, inside_toe, s.terrain))
        self.assertFalse(is_visible(clear, far, s.terrain))

    def test_monster_never_hidden(self):
        from fortyk.engine.combat import is_hidden

        s = new_toy_state(self.cat, seed=1)
        ec3 = s.units["EC3"]
        place(ec3, 36.2, 19.0)  # entièrement dans T3
        self.assertTrue(s.layout.disks_wholly_in_area(ec3.disks()))
        self.assertFalse(is_hidden(s, ec3))


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class MonsterVehicleTests(unittest.TestCase):
    """Redemptor Dreadnought (SM3) et Daemon Prince of Slaanesh (EC3) : règles propres aux MONSTER / VEHICLE."""

    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        self.s = new_toy_state(self.cat, seed=7)
        self.u = {k: self.s.units[k] for k in ("SM1", "SM2", "SM3", "EC1", "EC2", "EC3")}
        place(self.u["SM1"], 4, 4)
        place(self.u["SM2"], 40, 4)
        place(self.u["SM3"], 22, 6)
        place(self.u["EC1"], 4, 56)
        place(self.u["EC2"], 40, 56)
        place(self.u["EC3"], 22, 54)

    def test_build_and_status(self):
        sm3, ec3 = self.u["SM3"], self.u["EC3"]
        self.assertTrue(sm3.has_keyword("Vehicle") and ec3.has_keyword("Monster"))
        self.assertAlmostEqual(sm3.models[0].radius, 45 / 25.4, places=3)
        self.assertEqual(sorted(w.name for w in sm3.models[0].weapons), ["Heavy flamer", "Heavy onslaught gatling cannon", "Redemptor fist", "Twin fragstorm grenade launcher"])
        self.assertEqual(sm3.status, "Redemptor Dreadnought 12/12 PV")
        # sous la moitié : PV restants ≤ moitié pour une figurine seule
        self.assertFalse(sm3.below_half_strength)
        sm3.models[0].wounds = 7
        self.assertFalse(sm3.below_half_strength)
        sm3.models[0].wounds = 6
        self.assertTrue(sm3.below_half_strength)

    def test_duty_eternal_damage_reduction(self):
        import random

        from fortyk.engine.attack import AttackProfile, expected_attacks, resolve_attacks
        from fortyk.engine.combat import damage_reduction_of, defender_of
        from fortyk.data.parse import DiceExpr

        sm3, ec3 = self.u["SM3"], self.u["EC3"]
        self.assertEqual(damage_reduction_of(sm3), 1)
        self.assertEqual(damage_reduction_of(ec3), 0)
        dfd = defender_of(sm3)
        self.assertEqual(dfd.damage_reduction, 1)
        self.assertTrue(dfd.is_vehicle_or_monster)
        # infernal cannon D2 → 1 dégât par blessure passée sur le Redemptor
        p = AttackProfile(DiceExpr.fixed(3), 2, 5, -1, DiceExpr.fixed(2), count=1)
        out = resolve_attacks(p, dfd, random.Random(1))
        self.assertTrue(all(d == 1 for d in out.damage))
        exp = expected_attacks(p, dfd)
        self.assertAlmostEqual(exp.damage_per_wound, 1.0)
        # D3 → E[max(1, D3 − 1)] = (1 + 1 + 2) / 3
        p3 = AttackProfile(DiceExpr.fixed(1), 3, 8, -2, DiceExpr(1, 3, 0))
        self.assertAlmostEqual(expected_attacks(p3, dfd).damage_per_wound, 4 / 3)

    def test_big_guns_never_tire(self):
        from fortyk.engine.combat import shooting_ineligibility, target_locked_in_combat

        s = self.s
        sm3, ec1, ec2, sm1, sm2, ec3 = (self.u[k] for k in ("SM3", "EC1", "EC2", "SM1", "SM2", "EC3"))
        place(sm3, 24, 6)
        place(ec1, 24, 9.0)  # au contact du Redemptor (socle 1,77" + 0,63")
        self.assertTrue(sm3.in_engagement_range_of(ec1))
        # le Redemptor engagé tire quand même, toutes armes, sur l'unité engagée uniquement, à -1 hors Pistol
        self.assertIsNone(shooting_ineligibility(s, sm3))
        self.assertEqual([t.id for t in shooting_targets(s, sm3)], ["EC1"])
        profiles, _ = build_shooting_profiles(s, sm3, ec1)
        names = {p.label.split(" ×")[0]: p for p in profiles}
        self.assertIn("Heavy onslaught gatling cannon", names)
        self.assertEqual(names["Heavy onslaught gatling cannon"].hit_modifier, -1)
        self.assertNotIn("Twin fragstorm grenade launcher", names)  # Blast interdit sur une unité engagée avec une unité amie
        # une escouade d'infanterie engagée, elle, ne tire que ses pistolets
        place(sm1, 4, 4)
        place(ec2, 4, 6.8)
        self.assertTrue(sm1.in_engagement_range_of(ec2))
        p_inf, _ = build_shooting_profiles(s, sm1, ec2)
        self.assertEqual([p.label.split(" ×")[0] for p in p_inf], ["Bolt pistol"])
        # SM2 (libre) ne peut pas viser EC1, infanterie engagée avec le Redemptor
        self.assertTrue(target_locked_in_combat(s, sm2, ec1))
        # mais le Daemon Prince engagé avec SM1 reste ciblable par SM2, à -1 à la touche (couloir dégagé du bord droit)
        place(ec2, 40, 56)
        place(sm1, 41.7, 27)
        place(ec3, 42, 30.7)
        self.assertTrue(ec3.in_engagement_range_of(sm1))
        place(sm2, 41.7, 40)
        sm2.moved_in = 4.0  # pas de bonus Heavy, pour isoler le -1
        self.assertFalse(target_locked_in_combat(s, sm2, ec3))
        self.assertIn("EC3", [t.id for t in shooting_targets(s, sm2)])
        p_sm2, _ = build_shooting_profiles(s, sm2, ec3)
        self.assertTrue(p_sm2 and all(p.hit_modifier == -1 for p in p_sm2))
        # sans la règle, le prince engagé serait verrouillé comme n'importe quelle unité
        from dataclasses import replace

        s.rules = replace(s.rules, big_guns_never_tire=False)
        self.assertTrue(target_locked_in_combat(s, sm2, ec3))
        p_off, _ = build_shooting_profiles(s, sm3, ec1)
        self.assertEqual(p_off, [])  # le Redemptor engagé n'a pas de Pistol : plus rien à tirer

    def test_rapid_fire_and_hazardous_on_vehicle(self):
        from fortyk.engine.attack import hazardous_test
        import random

        s = new_toy_state(self.cat, seed=7, rosters={
            "attacker": (UnitSpec("SM3", "Redemptor Dreadnought", 1, "SM", weapon_overrides={0: ("twin storm bolter", "macro plasma incinerator", "Redemptor fist")}),),
            "defender": (UnitSpec("EC1", "Infractors", 5, "EC"),),
        })
        sm3, ec1 = s.units["SM3"], s.units["EC1"]
        place(sm3, 42, 6)  # couloir dégagé le long du bord droit
        place(ec1, 41.7, 16)  # ≈ 6,9" socle à socle : demi-portée du twin storm bolter (24")
        profiles, hazardous = build_shooting_profiles(s, sm3, ec1)
        by_name = {p.label.split(" ×")[0]: p for p in profiles}
        self.assertEqual(by_name["Twin storm bolter"].attacks.mean, 4)  # A2 + Rapid Fire 2
        self.assertEqual(by_name["Twin storm bolter"].reroll_wounds, "fails")  # Twin-linked
        self.assertIn("Macro plasma incinerator – supercharge", by_name)  # profil aux dégâts attendus maximaux
        self.assertEqual(by_name["Macro plasma incinerator – supercharge"].attacks.flat, 2)  # D6+1, Blast +1 (5 fig.)
        self.assertEqual(hazardous, 1)
        place(ec1, 41.7, 26)  # ≈ 17" : plus de Rapid Fire
        profiles, _ = build_shooting_profiles(s, sm3, ec1)
        by_name = {p.label.split(" ×")[0]: p for p in profiles}
        self.assertEqual(by_name["Twin storm bolter"].attacks.mean, 2)
        # Hazardous raté par un véhicule : 3 blessures mortelles
        rng = random.Random(0)
        results = {hazardous_test(random.Random(k), 1, True) for k in range(40)}
        self.assertEqual(results, {0, 3})
        self.assertEqual({hazardous_test(random.Random(k), 1, False) for k in range(40)}, {0, 1})

    def test_lone_operative_and_excessive_vigour(self):
        from fortyk.engine.combat import is_lone_operative, targeting_range_cap

        s = self.s
        ec3, ec1, sm1 = self.u["EC3"], self.u["EC1"], self.u["SM1"]
        place(sm1, 41.7, 30)  # couloir dégagé le long du bord droit
        place(ec3, 42, 50)  # ≈ 18" de SM1
        place(ec1, 4, 56)
        self.assertFalse(is_lone_operative(s, ec3))
        self.assertIn("EC3", [t.id for t in shooting_targets(s, sm1)])
        place(ec1, 41.3, 55)  # Infractors à ≤ 3" du prince : Lord of Excess → Lone Operative
        self.assertLessEqual(ec3.min_gap_to(ec1), 3.0)
        self.assertTrue(is_lone_operative(s, ec3))
        self.assertEqual(targeting_range_cap(s, ec3), 12.0)
        self.assertNotIn("EC3", [t.id for t in shooting_targets(s, sm1)])
        # Excessive Vigour : les Infractors qui ont chargé à 6" du prince gagnent +1 PA en mêlée
        place(ec1, 41.7, 33.0)  # au contact de SM1
        self.assertTrue(ec1.in_engagement_range_of(sm1))
        ec1.charged = True
        place(ec3, 42, 40)  # à ~4,5" des Infractors
        self.assertLessEqual(ec1.min_gap_to(ec3), 6.0)
        profiles = build_melee_profiles(s, ec1, sm1)
        self.assertEqual({p.label.split(" ×")[0]: p.ap for p in profiles}, {"Power sword": -3, "Duelling sabre": -2})
        place(ec3, 42, 54)  # trop loin
        profiles = build_melee_profiles(s, ec1, sm1)
        self.assertEqual({p.label.split(" ×")[0]: p.ap for p in profiles}, {"Power sword": -2, "Duelling sabre": -1})
        ec1.charged = False
        place(ec3, 42, 40)
        profiles = build_melee_profiles(s, ec1, sm1)
        self.assertEqual({p.label.split(" ×")[0]: p.ap for p in profiles}, {"Power sword": -2, "Duelling sabre": -1})  # pas de charge : pas de bonus

    def test_deadly_demise(self):
        from fortyk.engine.game import Game

        s = self.s
        sm3, ec1, ec3 = self.u["SM3"], self.u["EC1"], self.u["EC3"]
        place(ec1, 22, 10)  # à ~1,6" du Redemptor
        place(ec3, 26, 6)   # à ~1" du Redemptor
        g = Game(s, {"attacker": RandomAgent(), "defender": RandomAgent()})
        sm3.models[0].wounds = 0
        sm3.models[0].alive = False

        class Fixed:
            def __init__(self, values):
                self.values = list(values)

            def randint(self, a, b):
                return self.values.pop(0)

            def choice(self, seq):
                return seq[0]

        s.rng = Fixed([6, 3])  # D6 = 6 → explose ; D3 = 3 blessures mortelles
        g._on_unit_destroyed(sm3)
        self.assertEqual(ec1.strength, 4)  # 3 BM : une Infractor (2 PV) meurt, la suivante perd 1
        self.assertEqual(sum(m.wounds for m in ec1.alive_models), 7)
        self.assertEqual(ec3.models[0].wounds, 7)
        self.assertTrue(any("explose !" in line for line in s.log))


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class CoverTests(unittest.TestCase):
    """Couvert V11 : infanterie qui touche un décor ; monstre / véhicule entièrement dans une ruine ou partiellement occulté."""

    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def test_infantry_toe_and_vehicle_wholly_in_ruin(self):
        from fortyk.engine.combat import _unit_has_cover_from

        s = new_toy_state(self.cat, seed=11)
        sm1, ec1, ec3, sm3 = (s.units[k] for k in ("SM1", "EC1", "EC3", "SM3"))
        for u in s.units.values():
            place(u, 2.5, 58)  # tout le monde hors champ dans un coin, puis on place les acteurs
        place(sm1, 41.7, 30)
        shooter = sm1.alive_models[0]
        # infanterie : la 2e rangée touche la barricade T4m (y ≤ 46), pas entièrement dedans → couvert quand même
        place(ec1, 39.5, 42.0)  # rangées à y = 41,25 et 42,75 (r 0,63) : 42,75 + 0,63 > 43 → touche T4m ; 41,25 + 0,63 < 43 → ne touche pas
        self.assertFalse(_unit_has_cover_from(s, shooter, ec1))
        place(ec1, 39.5, 43.2)  # rangée du haut à 42,45 : touche (43,08 > 43) ; rangée du bas dedans
        self.assertTrue(all(s.layout.disk_touches_area(m.disk) for m in ec1.alive_models))
        self.assertTrue(_unit_has_cover_from(s, shooter, ec1))
        # monstre : un orteil dans une barricade légère ne suffit pas (et une barricade n'est pas une ruine)
        place(sm1, 41.7, 25)
        shooter = sm1.alive_models[0]
        place(ec3, 36.5, 33.0)  # touche T8m L (y ≤ 32) par le haut et T7m par la gauche ; pleinement visible
        self.assertTrue(s.layout.disk_touches_area(ec3.disks()[0]))
        self.assertIsNone(s.layout.ruin_containing_disk(ec3.disks()[0]))
        self.assertFalse(_unit_has_cover_from(s, shooter, ec3))
        # …mais partiellement occulté par une barricade qu'il ne touche pas, il a le couvert
        place(ec3, 39.0, 47.5)  # juste sous T4m (y 43–46) sans la toucher : la barricade masque une partie du socle
        self.assertFalse(s.layout.disk_touches_area(ec3.disks()[0]))
        place(sm1, 41.7, 30)
        shooter = sm1.alive_models[0]
        self.assertTrue(_unit_has_cover_from(s, shooter, ec3))
        # monstre entièrement dans la ruine T1m (22,5–30 × 42,5–54), visible depuis la ruine T9 ? on tire d'en face
        place(sm1, 26.0, 38.5)  # au sud de T9 (y ≤ 35), à découvert
        shooter = sm1.alive_models[0]
        place(ec3, 26.0, 48.0)
        self.assertIsNotNone(s.layout.ruin_containing_disk(ec3.disks()[0]))
        self.assertTrue(_unit_has_cover_from(s, shooter, ec3))
        place(ec3, 26.0, 41.5)  # à cheval sur le bord nord de T1m (42,5) : pas entièrement dedans, pleinement visible
        self.assertFalse(_unit_has_cover_from(s, shooter, ec3))


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class LightTerrainBlocksSightTests(unittest.TestCase):
    """Retour de Valentin : SM2 (dans T3) ne voit pas le Daemon Prince derrière la barricade légère T8m."""

    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def test_prince_behind_light_barricade_is_not_visible(self):
        from fortyk.engine.geometry import is_visible

        s = new_toy_state(self.cat, seed=1)
        for u in s.units.values():
            place(u, 2.5, 58)
        sm2, ec3 = s.units["SM2"], s.units["EC3"]
        place(sm2, 37.3, 21.3)  # dans T3, comme sur la capture
        place(ec3, 36.6, 36.3)  # à cheval sur le bord de T7m, derrière T8m L (33,5–40 × 30–32)
        prince = ec3.alive_models[0]
        self.assertFalse(any(is_visible(m.disk, prince.disk, s.terrain) for m in sm2.alive_models))
        self.assertEqual(shooting_targets(s, sm2), [])
        # en avançant le prince jusqu'à toucher T8m, il redevient visible (un orteil suffit)
        place(ec3, 36.6, 33.0)
        self.assertTrue(any(is_visible(m.disk, prince.disk, s.terrain) for m in sm2.alive_models))
        self.assertEqual([t.id for t in shooting_targets(s, sm2)], ["EC3"])
