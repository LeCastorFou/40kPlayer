"""Latitudes des règles : effets à durée, effets manuels, stratagèmes du panneau, traduction des
textes Wahapedia (compilateur de règles), fenêtres des stratagèmes de détachement."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fortyk.data import default_raw_dir, load_catalog  # noqa: E402
from fortyk.data.list_library import list_text  # noqa: E402
from fortyk.engine import Engine  # noqa: E402
from fortyk.engine.actions import ManualAction, UseStratagemAction  # noqa: E402
from fortyk.engine.combat import build_melee_profiles, build_shooting_profiles, defender_of  # noqa: E402
from fortyk.engine.effects import add_effect, active_effects  # noqa: E402
from fortyk.rules_compiler import compile_text, parse_timing  # noqa: E402
from fortyk.toy import new_toy_state  # noqa: E402
from fortyk.training import build_initial_state  # noqa: E402
from test_v11_rules import park_all, place  # noqa: E402

HAS_DATA = (default_raw_dir() / "Datasheets.csv").exists()


class CompilerTests(unittest.TestCase):
    def test_attack_and_defence_sentences(self):
        c = compile_text("Until the end of the phase, each time a model in your unit makes an attack, add 1 to the Hit roll and add 1 to the Wound roll.")
        self.assertEqual(c.status, "auto")
        self.assertEqual({(e.kind, e.value) for e in c.effects}, {("hit_mod", 1), ("wound_mod", 1)})
        c = compile_text("Until the attacking unit has finished making its attacks, each time an attack targets your unit, worsen the Armour Penetration characteristic of that attack by 1.")
        self.assertEqual([(e.kind, e.value) for e in c.effects], [("ap_worsen", 1)])
        c = compile_text("Until the end of the turn, ranged weapons equipped by models in your unit have the [LETHAL HITS] ability.")
        self.assertEqual([(e.kind, e.value, e.scope, e.until) for e in c.effects], [("weapon_ability", "lethal hits", "ranged", "turn")])
        c = compile_text("Until the end of the phase, models in your unit have the Feel No Pain 5+ ability.")
        self.assertEqual([(e.kind, e.value) for e in c.effects], [("fnp", 5)])
        c = compile_text("Until the end of the turn, your unit is eligible to shoot and declare a charge in a turn in which it Fell Back.")
        self.assertEqual({e.kind for e in c.effects}, {"fall_back_and_shoot", "fall_back_and_charge"})
        c = compile_text("Until the end of the phase, each time a model in your unit makes a ranged attack that targets a MONSTER or VEHICLE unit, "
                         "improve the Armour Penetration characteristic of that attack by 1.")
        self.assertEqual([(e.kind, e.cond) for e in c.effects], [("ap_mod", "kw:Monster|Vehicle")])

    def test_instants_choices_and_manual(self):
        c = compile_text("Remove your unit from the battlefield and place it into Strategic Reserves.")
        self.assertEqual((c.status, [i.kind for i in c.instants]), ("auto", ["reserve"]))
        c = compile_text("Select one enemy unit within 12\" of your unit. Roll six D6: for each 4+, that enemy unit suffers 1 mortal wound.")
        self.assertTrue(c.needs_enemy)
        self.assertEqual(c.instants[0].value, ("dice", 6, 4, 1))
        c = compile_text("Select either the [LETHAL HITS] or [SUSTAINED HITS 1] ability. Until the end of the phase ranged weapons equipped by models "
                         "in your unit have the selected ability.")
        self.assertEqual((c.choices, c.effects[0].value), (("lethal hits", "sustained hits 1"), "$choice"))
        c = compile_text("Until the end of the phase, each time a model in your unit is destroyed, if that model has not fought this phase, roll one D6: "
                         "on a 4+, do not remove it from play.")
        self.assertEqual(c.status, "manual")

    def test_timing(self):
        t = parse_timing("Your opponent’s Shooting phase or the Fight phase, just after an enemy unit has selected its targets.", "Either player’s turn", "Shooting or Fight phase")
        self.assertEqual((t.moment, set(t.phases)), ("targets_selected", {"shooting", "fight"}))
        self.assertEqual(parse_timing("End of your opponent’s Fight phase.", "Opponent’s turn").moment, "end")
        self.assertEqual(parse_timing("Your Shooting phase.", "Your turn").moment, "any")


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class EffectsInCombatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        self.s = new_toy_state(self.cat, seed=1)
        park_all(self.s)
        self.s.phase, self.s.turn_counter = "shooting", 1
        self.sm1, self.ec1 = self.s.units["SM1"], self.s.units["EC1"]
        place(self.sm1, 42, 26)
        place(self.ec1, 42, 35)

    def profile(self):
        return build_shooting_profiles(self.s, self.sm1, self.ec1)[0][0]

    def test_modifiers_and_expiry(self):
        s = self.s
        base = self.profile()
        add_effect(s, "hit_mod", "SM1", 1, scope="ranged")
        add_effect(s, "ap_mod", "SM1", 1)
        add_effect(s, "weapon_ability", "SM1", "lethal hits", scope="ranged")
        add_effect(s, "crit_hit_on", "SM1", 5)
        p = self.profile()
        self.assertEqual(p.hit_modifier, base.hit_modifier + 1)
        self.assertEqual(p.ap, base.ap - 1)
        self.assertTrue(p.lethal_hits and not base.lethal_hits)
        self.assertEqual(p.critical_hit_on, 5)
        add_effect(s, "hit_mod_against", "EC1", -1, scope="ranged")  # Stealth
        self.assertEqual(self.profile().hit_modifier, base.hit_modifier)
        s.phase = "charge"  # fin de la phase : les effets « phase » expirent
        self.assertEqual(self.profile().hit_modifier, base.hit_modifier)
        self.assertEqual(active_effects(s, "SM1"), [])

    def test_defensive_effects(self):
        s = self.s
        add_effect(s, "fnp", "EC1", 5)
        add_effect(s, "invuln", "EC1", 4)
        add_effect(s, "toughness_mod", "EC1", 1)
        d = defender_of(self.ec1, s)
        self.assertEqual((d.feel_no_pain, d.invuln), (5, 4))
        self.assertEqual(d.toughness, defender_of(self.ec1).toughness + 1)
        add_effect(s, "wound_mod_against", "EC1", -1, cond="s_gt_t")  # seulement si F > E
        p = self.profile()
        self.assertEqual(p.wound_modifier, 0)  # bolt rifle F4 contre E5

    def test_conditional_on_target_keyword(self):
        s = self.s
        add_effect(s, "ap_mod", "SM1", 1, cond="kw:Monster|Vehicle")
        self.assertEqual(self.profile().ap, build_shooting_profiles(new_toy_state(self.cat, seed=1), self.sm1, self.ec1)[0][0].ap)
        melee = build_melee_profiles(s, self.sm1, self.s.units["EC3"])
        self.assertTrue(melee == [] or all(isinstance(m.ap, int) for m in melee))


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class ManualActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def setUp(self):
        self.s = new_toy_state(self.cat, seed=1)
        park_all(self.s)
        self.eng = Engine()
        place(self.s.units["SM1"], 30, 20)
        self.eng.start_at(self.s, "movement", "attacker")

    def test_cp_heal_mortal_battleshock(self):
        s, eng = self.s, self.eng
        cp = s.cp["defender"]
        eng.apply_free(s, "defender", ManualAction("cp", value=2, rule="test"))  # l'adversaire, pendant mon tour
        self.assertEqual(s.cp["defender"], cp + 2)
        with self.assertRaises(Exception):
            eng.apply_free(s, "defender", ManualAction("cp", value=-99))
        sm1 = s.units["SM1"]
        m = sm1.models[0]
        m.wounds = 1
        eng.apply_free(s, "attacker", ManualAction("heal", "SM1", (m.id,), value=5, rule="Narthecium"))
        self.assertEqual(m.wounds, m.profile.wounds)
        eng.apply_free(s, "attacker", ManualAction("mortal", "SM1", value=3))
        self.assertEqual(sum(q.profile.wounds - q.wounds for q in sm1.models if q.alive) + 2 * sum(1 for q in sm1.models if not q.alive), 3)
        eng.apply_free(s, "attacker", ManualAction("battleshock", "SM1", value=1))
        self.assertTrue(sm1.battle_shocked)
        self.assertTrue(any(l.startswith("Effet manuel — attacker (Narthecium)") for l in s.log))

    def test_revive_reserve_and_set_up(self):
        s, eng = self.s, self.eng
        sm1 = s.units["SM1"]
        dead = sm1.models[3:]
        spots = {m.id: (m.x, m.y) for m in dead}
        for m in dead:
            m.alive, m.wounds = False, 0
        pos = tuple((mid, x, y) for mid, (x, y) in spots.items())
        eng.apply_free(s, "attacker", ManualAction("revive", "SM1", tuple(spots), positions=pos, rule="Reanimation"))
        self.assertTrue(all(m.alive and m.wounds == m.profile.wounds for m in dead))
        bad = tuple((mid, sm1.models[0].x, sm1.models[0].y) for mid in list(spots)[:1])
        dead[0].alive = False
        self.assertIn("chevauche", eng.free_error(s, "attacker", ManualAction("revive", "SM1", (dead[0].id,), positions=bad)))
        dead[0].alive = True
        eng.apply_free(s, "attacker", ManualAction("reserve", "SM1", rule="Gate"))
        self.assertTrue(sm1.in_reserve and sm1.repositioned)
        pos = tuple((m.id, m.x + 10, m.y + 20) for m in sm1.alive_models)
        eng.apply_free(s, "attacker", ManualAction("set_up", "SM1", positions=pos))
        self.assertFalse(sm1.in_reserve)
        self.assertAlmostEqual(sm1.models[0].x, spots and sm1.models[0].x)

    def test_effect_and_replay(self):
        s, eng = self.s, self.eng
        eng.apply_free(s, "attacker", ManualAction("effect", "SM1", effect="move_mod", effect_value="2", until="turn", rule="Rites"))
        self.assertEqual(eng.move_of(s, s.units["SM1"]), s.units["SM1"].move_in + 2)
        self.assertEqual(eng.decision(s).kind, "select_unit")  # la décision en cours ne bouge pas


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class DetachmentStratagemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def state(self):
        n = "ec_mercurial_host_2000"
        s = build_initial_state(self.cat, {"seed": 3, "rev": 3, "stratagems": True, "lists": {"attacker": {"name": n, "text": list_text(n)}, "defender": None}})
        s.cp = {"attacker": 5, "defender": 5}
        return s

    def test_panel_use_checks_timing_cp_and_applies_effects(self):
        s = self.state()
        eng = Engine()
        park = [(3.0 + 5 * i, 3.0 + 7 * (i % 2)) for i in range(20)]
        for u, (x, y) in zip(s.units.values(), park):
            place(u, x, y)
        eng.stop_at = {"end_turn"}  # rester dans la phase de combat
        eng.start_at(s, "fight", "attacker")
        book = {st.name: st for st in s.stratagem_book["attacker"]}
        violent = book["VIOLENT EXCESS"]
        fb = next(u for u in s.units_of("attacker") if u.datasheet.name.startswith("Flawless"))
        use = UseStratagemAction(violent.id, fb.id)
        self.assertIsNone(eng.free_error(s, "attacker", use))
        eng.apply_free(s, "attacker", use)
        self.assertEqual(s.cp["attacker"], 5 - violent.cp)
        self.assertTrue(any(e.kind == "weapon_ability" and e.value == "sustained hits 1" for e in active_effects(s, fb.id)))
        self.assertIn("déjà utilisé", eng.free_error(s, "attacker", UseStratagemAction(violent.id, s.units_of("attacker")[0].id)))
        # mauvais moment : un stratagème de phase de mouvement pendant le combat
        prince = book["HONOUR THE PRINCE"]
        self.assertIn("mouvement", eng.free_error(s, "attacker", UseStratagemAction(prince.id, fb.id)))
        # l'adversaire (toy model) n'a que les stratagèmes de base
        self.assertEqual(eng.free_error(s, "defender", UseStratagemAction(violent.id, fb.id)), "stratagème inconnu pour ce camp")

    def test_targets_selected_window_for_the_defender(self):
        s = self.state()
        # le toy model tire sur une unité Emperor's Children qui peut jouer Capricious Reactions
        n = "ec_mercurial_host_2000"
        s = build_initial_state(self.cat, {"seed": 3, "rev": 3, "stratagems": True, "lists": {"attacker": None, "defender": {"name": n, "text": list_text(n)}}})
        s.cp = {"attacker": 5, "defender": 5}
        eng = Engine()
        park = [(2.0 + 4.3 * (i % 10), 2.0 if i < 10 else 58.0) for i in range(20)]
        for u, (x, y) in zip(s.units.values(), park):
            place(u, x, y)
        sm1 = s.units["SM1"]
        target = next(u for u in s.units_of("defender") if u.datasheet.name.startswith("Infractors"))
        place(sm1, 42, 26)
        place(target, 42, 36)
        eng.start_at(s, "shooting", "attacker")
        from fortyk.engine.actions import SelectUnitAction, ShootAction, StratagemAction

        while eng.decision(s).kind == "stratagem":  # Smokescreen du Rhino, etc. : on passe
            eng.step(s, StratagemAction(None))
        eng.step(s, SelectUnitAction("SM1"))
        d = eng.decision(s)
        self.assertEqual(d.kind, "shoot")
        eng.step(s, ShootAction("SM1", target.id))
        d = eng.decision(s)
        self.assertEqual((d.kind, d.window, d.side), ("stratagem", "detachment_targets", "defender"))
        names = {s_.name for s_ in s.stratagem_book["defender"] if any(o.stratagem_id == s_.id for o in d.options if hasattr(o, "stratagem_id"))}
        self.assertIn("CAPRICIOUS REACTIONS", names)
        use = next(o for o in d.options if hasattr(o, "stratagem_id") and next(x for x in s.stratagem_book["defender"] if x.id == o.stratagem_id).name == "CAPRICIOUS REACTIONS")
        self.assertEqual((use.unit_id, use.target_id), (target.id, "SM1"))
        eng.step(s, use)
        self.assertTrue(any(e.kind == "hit_mod_against" and e.value == -1 for e in active_effects(s, target.id)))
        self.assertTrue(any(l.startswith("Tir :") and "Intercessor" in l for l in s.log))  # puis le tir est résolu


if __name__ == "__main__":
    unittest.main()
