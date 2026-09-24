"""Stratagèmes de base V11 (15) : restrictions de 15.01, fenêtres de réaction et effets."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fortyk.data import default_raw_dir, load_catalog  # noqa: E402
from fortyk.engine import Engine  # noqa: E402
from fortyk.engine.actions import (  # noqa: E402
    AutoChargeMoveAction, ChargeAction, DeclareAdvanceAction, DeclareChargeAction, EndPhaseAction, FightAction, SelectUnitAction, StratagemAction,
)
from fortyk.engine.combat import build_shooting_profiles  # noqa: E402
from fortyk.engine.stratagems import record_use, unavailable  # noqa: E402
from fortyk.toy import new_toy_state  # noqa: E402
from test_v11_rules import park_all, place  # noqa: E402

HAS_DATA = (default_raw_dir() / "Datasheets.csv").exists()
PASS = StratagemAction(None)


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class StratagemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        s = self.s = new_toy_state(self.cat, seed=1)
        park_all(s)
        s.stratagems = True
        s.cp = {"attacker": 3, "defender": 3}
        self.u = s.units
        self.eng = Engine()

    def dice(self, *values):
        """Les prochains D6 du moteur (jets d'Advance, de charge, de battle-shock, Explosives…)."""
        it = iter(values)
        self.s.d6 = lambda: next(it, 3)

    def strat_options(self, d, key):
        return [o for o in d.options if isinstance(o, StratagemAction) and o.stratagem == key]

    # ------------------------------------------------------------------ 15.01

    def test_restrictions_of_15_01(self):
        s, sm1, sm2 = self.s, self.u["SM1"], self.u["SM2"]
        s.phase, s.turn_counter = "shooting", 1
        self.assertIsNone(unavailable(s, "attacker", "explosives", sm1))
        record_use(s, "attacker", "explosives", sm1.id)
        self.assertEqual(s.cp["attacker"], 2)
        self.assertIn("déjà utilisé", unavailable(s, "attacker", "explosives", sm2))  # même stratagème, même phase
        self.assertIn("déjà ciblée", unavailable(s, "attacker", "command_reroll", sm1))  # même unité, même phase
        self.assertIsNone(unavailable(s, "attacker", "command_reroll", sm2))
        self.assertIsNone(unavailable(s, "defender", "explosives"))  # chaque joueur a ses limites
        s.phase = "charge"
        self.assertIsNone(unavailable(s, "attacker", "explosives", sm1))  # nouvelle phase
        sm2.battle_shocked = True
        self.assertIn("battle-shocked", unavailable(s, "attacker", "command_reroll", sm2))
        s.cp["attacker"] = 0
        self.assertIn("1 CP requis", unavailable(s, "attacker", "command_reroll", sm1))
        s.cp["attacker"] = 5
        record_use(s, "attacker", "insane_bravery", sm1.id)
        s.turn_counter += 2
        self.assertIn("une fois par bataille", unavailable(s, "attacker", "insane_bravery", sm1))
        s.cp["defender"] = 1
        self.assertIsNone(unavailable(s, "defender", "heroic_intervention", mode="leap"))
        self.assertIn("2 CP requis", unavailable(s, "defender", "heroic_intervention", mode="fray"))  # Into the Fray : +1 CP
        s.stratagems = False
        self.assertIsNotNone(unavailable(s, "defender", "explosives"))

    # ------------------------------------------------------------------ Command Re-roll

    def test_command_reroll_advance(self):
        s, eng = self.s, self.eng
        place(self.u["SM1"], 42, 20)
        eng.start_at(s, "movement", "attacker")
        eng.step(s, SelectUnitAction("SM1"))
        self.dice(2, 5)
        eng.step(s, DeclareAdvanceAction("SM1"))
        d = eng.decision(s)
        self.assertEqual((d.kind, d.window, d.side), ("stratagem", "reroll_advance", "attacker"))
        self.assertEqual(d.options, [StratagemAction("command_reroll", "SM1", mode="advance"), PASS])
        eng.step(s, d.options[0])
        d = eng.decision(s)
        self.assertEqual((d.kind, d.max_distance), ("advance_move", 6 + 5))
        self.assertEqual(s.cp["attacker"], 2)
        self.assertTrue(any("Command Re-roll" in l and "2 → 5" in l for l in s.log))

    def test_no_window_when_useless_or_unaffordable(self):
        s, eng = self.s, self.eng
        place(self.u["SM1"], 42, 20)
        place(self.u["SM2"], 30, 20)
        eng.start_at(s, "movement", "attacker")
        eng.step(s, SelectUnitAction("SM1"))
        self.dice(6)
        eng.step(s, DeclareAdvanceAction("SM1"))
        self.assertEqual(eng.decision(s).kind, "advance_move")  # un 6 : rien à relancer, rien n'est demandé
        s.cp["attacker"] = 0
        eng.step(s, eng.decision(s).options[0])
        eng.step(s, SelectUnitAction("SM2"))
        self.dice(1)
        eng.step(s, DeclareAdvanceAction("SM2"))
        self.assertEqual(eng.decision(s).kind, "advance_move")  # pas de CP : pas de fenêtre

    def test_command_reroll_charge_rerolls_both_dice(self):
        s, eng = self.s, self.eng
        place(self.u["SM1"], 42, 26)
        place(self.u["EC1"], 42, 35)
        eng.start_at(s, "charge", "attacker")
        eng.step(s, SelectUnitAction("SM1"))
        self.dice(1, 1, 6, 6)
        eng.step(s, DeclareChargeAction("SM1"))
        d = eng.decision(s)
        self.assertEqual((d.kind, d.window), ("stratagem", "reroll_charge"))  # double 1 : échec si on ne relance pas
        eng.step(s, d.options[0])
        d = eng.decision(s)
        self.assertEqual((d.kind, d.max_distance), ("charge_move", 12.0))
        self.assertTrue(any("relance son jet de charge : 2 → 12" in l for l in s.log))

    def test_charge_reroll_offered_even_when_a_target_is_reached(self):
        """Valentin : on peut vouloir une charge plus longue même si une unité est atteignable."""
        for rev, expected in ((3, "stratagem"), (2, "charge_move")):
            s = new_toy_state(self.cat, seed=1)
            park_all(s)
            s.stratagems, s.rev = True, rev
            s.cp = {"attacker": 3, "defender": 3}
            place(s.units["SM1"], 42, 26)
            place(s.units["EC1"], 42, 35)
            eng = Engine()
            eng.start_at(s, "charge", "attacker")
            eng.step(s, SelectUnitAction("SM1"))
            it = iter((4, 4))
            s.d6 = lambda: next(it, 3)
            eng.step(s, DeclareChargeAction("SM1"))
            d = eng.decision(s)
            self.assertEqual(d.kind, expected, f"révision {rev}")
            if rev == 3:
                self.assertIn("atteignable : EC1", d.note)

    # ------------------------------------------------------------------ Fire Overwatch, Smokescreen

    def test_fire_overwatch_at_end_of_opponent_movement(self):
        s, eng = self.s, self.eng
        place(self.u["SM1"], 42, 26)
        place(self.u["EC1"], 42, 35)
        eng.start_at(s, "movement", "attacker")
        eng.step(s, EndPhaseAction())
        d = eng.decision(s)
        self.assertEqual((d.kind, d.window, d.side), ("stratagem", "fire_overwatch", "defender"))
        opts = self.strat_options(d, "fire_overwatch")
        self.assertIn(StratagemAction("fire_overwatch", "EC1", target_id="SM1"), opts)
        profiles, _ = build_shooting_profiles(s, self.u["EC1"], self.u["SM1"], snap=True)
        self.assertTrue(profiles and all(p.min_unmodified_hit == 6 and p.reroll_hits == "none" for p in profiles))
        eng.step(s, StratagemAction("fire_overwatch", "EC1", target_id="SM1"))
        self.assertEqual(s.cp["defender"], 2)
        self.assertTrue(any(l.startswith("Stratagème Fire Overwatch") for l in s.log))
        self.assertTrue(any("tir d'opportunité : 6 non modifié" in l for l in s.log))
        self.assertTrue(self.u["EC1"].has_shot)
        d = eng.decision(s)
        self.assertEqual((d.side, s.phase), ("attacker", "shooting"))  # puis la phase de tir de l'attaquant

    def test_smokescreen_gives_cover(self):
        from fortyk.engine.combat import _unit_has_cover_from

        s = self.s
        s.phase, s.turn_counter = "shooting", 3
        ec1, sm1 = self.u["EC1"], self.u["SM1"]
        place(sm1, 42, 26)
        place(ec1, 42, 35)
        self.assertFalse(_unit_has_cover_from(s, sm1.models[0], ec1))
        record_use(s, "defender", "smokescreen", ec1.id, detail=True)
        self.assertTrue(_unit_has_cover_from(s, sm1.models[0], ec1))
        s.phase = "charge"
        self.assertFalse(_unit_has_cover_from(s, sm1.models[0], ec1))  # jusqu'à la fin de la phase seulement

    # ------------------------------------------------------------------ Explosives

    def test_explosives_in_own_shooting_phase(self):
        s, eng = self.s, self.eng
        place(self.u["SM1"], 42, 26)
        place(self.u["EC1"], 42, 35)
        eng.start_at(s, "shooting", "attacker")
        d = eng.decision(s)
        self.assertEqual(d.kind, "select_unit")
        grenade = StratagemAction("explosives", "SM1", target_id="EC1")
        self.assertIn(grenade, d.options)
        self.dice(4, 5, 6, 1, 2, 3)
        eng.step(s, grenade)
        ec1 = self.u["EC1"]
        self.assertEqual(sum(m.profile.wounds - m.wounds for m in ec1.models), 3)  # 3 BM
        self.assertEqual(s.cp["attacker"], 2)
        d = eng.decision(s)
        self.assertIn(SelectUnitAction("SM1"), d.options)  # l'unité peut encore tirer
        self.assertFalse(self.strat_options(d, "explosives"))  # une fois par phase

    # ------------------------------------------------------------------ Insane Bravery

    def test_insane_bravery_once_per_battle(self):
        s, eng = self.s, self.eng
        ec1 = self.u["EC1"]
        for m in ec1.models[:3]:
            m.alive, m.wounds = False, 0
        eng.start_at(s, "command", "defender")
        d = eng.decision(s)
        self.assertEqual((d.kind, d.window, d.side), ("stratagem", "insane_bravery", "defender"))
        eng.step(s, StratagemAction("insane_bravery", "EC1"))
        self.assertFalse(ec1.battle_shocked)
        self.assertTrue(any("Insane Bravery : test réussi d'office" in l for l in s.log))
        self.assertEqual(s.cp["defender"], 3)  # +1 CP de la phase, -1
        # plus tard dans la bataille : plus de fenêtre
        s.flow.step = "turn_start"
        s.flow.decision = None
        s.flow.turn_index = 0 if s.first_player == "defender" else 1
        self.dice(1, 1)
        eng._advance(s)
        self.assertNotEqual(eng.decision(s).kind, "stratagem")
        self.assertTrue(ec1.battle_shocked)  # 2 < Ld : raté

    def test_battle_shocked_unit_cannot_be_targeted(self):
        s, eng = self.s, self.eng
        ec1 = self.u["EC1"]
        for m in ec1.models[:3]:
            m.alive, m.wounds = False, 0
        ec1.battle_shocked = True
        eng.start_at(s, "command", "defender")
        self.assertNotEqual(eng.decision(s).kind, "stratagem")

    # ------------------------------------------------------------------ Heroic Intervention, Counteroffensive

    def charge_sm1_into_ec1(self, extra=()):
        s, eng = self.s, self.eng
        place(self.u["SM1"], 42, 26)
        place(self.u["EC1"], 42, 35)
        eng.start_at(s, "charge", "attacker")
        eng.step(s, SelectUnitAction("SM1"))
        self.dice(6, 6, *extra)
        eng.step(s, DeclareChargeAction("SM1"))
        d = eng.decision(s)
        if d.kind == "charge_target":
            eng.step(s, ChargeAction("SM1", "EC1"))
            d = eng.decision(s)
        self.assertEqual(d.kind, "charge_move")
        eng.step(s, AutoChargeMoveAction("SM1", "EC1"))
        self.assertTrue(self.u["SM1"].charged)

    def test_heroic_intervention_leap_to_defend(self):
        s, eng = self.s, self.eng
        place(self.u["EC2"], 42, 19)  # dans le dos de SM1 une fois la charge faite
        self.charge_sm1_into_ec1(extra=(6, 6))
        d = eng.decision(s)
        self.assertEqual((d.kind, d.window, d.side), ("stratagem", "heroic_intervention", "defender"))
        leap = StratagemAction("heroic_intervention", "EC2", mode="leap")
        self.assertIn(leap, d.options)
        self.assertFalse(any(o.unit_id == "EC1" for o in self.strat_options(d, "heroic_intervention")))  # engagée
        eng.step(s, leap)
        d = eng.decision(s)
        self.assertEqual((d.kind, d.side), ("charge_move", "defender"))
        eng.step(s, d.options[0])
        ec2 = self.u["EC2"]
        self.assertTrue(ec2.charged and ec2.fights_first)
        self.assertTrue(s.is_engaged(ec2))
        self.assertEqual(s.cp["defender"], 2)
        self.assertEqual(s.phase, "fight")

    def test_counteroffensive_fights_next(self):
        s, eng = self.s, self.eng
        self.charge_sm1_into_ec1()
        d = eng.decision(s)
        self.assertEqual((d.kind, d.side), ("fight", "attacker"))  # SM1 a chargé : Fights First
        eng.step(s, FightAction("SM1", "EC1"))
        d = eng.decision(s)
        self.assertEqual((d.kind, d.window, d.side), ("stratagem", "counteroffensive", "defender"))
        eng.step(s, StratagemAction("counteroffensive", "EC1"))
        self.assertEqual(s.cp["defender"], 1)
        d = eng.decision(s)
        self.assertEqual((d.kind, d.side), ("fight", "defender"))
        self.assertTrue(d.options and all(o.unit_id == "EC1" for o in d.options))

    def test_crushing_impact_after_vehicle_charge(self):
        s, eng = self.s, self.eng
        sm3, ec1 = self.u["SM3"], self.u["EC1"]
        place(sm3, 42, 25)
        place(ec1, 42, 35)
        eng.start_at(s, "charge", "attacker")
        eng.step(s, SelectUnitAction("SM3"))
        self.dice(6, 6, 1, 5, 6, 2, 3, 4, 5, 1, 1, 6)
        eng.step(s, DeclareChargeAction("SM3"))
        d = eng.decision(s)
        eng.step(s, AutoChargeMoveAction("SM3", "EC1"))
        d = eng.decision(s)
        self.assertEqual((d.kind, d.window, d.side), ("stratagem", "crushing_impact", "attacker"))
        opt = self.strat_options(d, "crushing_impact")[0]
        self.assertEqual((opt.target_id, opt.model_id), ("EC1", sm3.models[0].id))
        eng.step(s, opt)
        self.assertEqual(sm3.models[0].profile.wounds - sm3.models[0].wounds, 3)  # trois 1
        self.assertEqual(sum(m.profile.wounds - m.wounds for m in ec1.models), 4)  # quatre 5+
        self.assertTrue(any("Crushing Impact" in l and "10D6" in l for l in s.log))

    # ------------------------------------------------------------------ Epic Challenge

    def test_epic_challenge_gives_precision(self):
        s, eng = self.s, self.eng
        sm1, ec3 = self.u["SM1"], self.u["EC3"]
        place(sm1, 42, 26)
        sm1.models[0].is_leader = True  # un personnage mène l'unité : [PRECISION] peut le viser
        front = min(sm1.alive_models, key=lambda m: m.y)
        place(ec3, front.x, front.y - (front.radius + ec3.models[0].radius + 0.4))
        eng.start_at(s, "fight", "defender")
        d = eng.decision(s)
        self.assertEqual((d.kind, d.side), ("fight", "defender"))
        eng.step(s, FightAction("EC3", "SM1"))
        d = eng.decision(s)
        self.assertEqual((d.kind, d.window), ("stratagem", "epic_challenge"))
        eng.step(s, StratagemAction("epic_challenge", "EC3", model_id=ec3.models[0].id))
        self.assertTrue(any("(Epic Challenge)" in l for l in s.log))
        self.assertEqual(s.cp["defender"], 2)


if __name__ == "__main__":
    unittest.main()
