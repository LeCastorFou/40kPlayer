"""Séquence d'attaque : valeurs de référence calculées à la main, puis concordance
entre la version aux dés et la version en espérance sur des scénarios variés."""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data.parse import DiceExpr, parse_dice  # noqa: E402
from fortyk.engine.attack import (  # noqa: E402
    AttackProfile,
    Defender,
    estimate_models_slain,
    expected_attacks,
    expected_hazardous_mortal_wounds,
    hazardous_test,
    hit_roll_needed,
    resolve_attacks,
)

MARINE = Defender(toughness=4, save=3)  # Intercessor / Infractor : T4 Sv3+ W2
D = parse_dice


def bolt_rifles(n=5, **kw):
    return AttackProfile(attacks=D("2"), skill=3, strength=4, ap=-1, damage=D("1"), count=n, **kw)


class ReferenceValues(unittest.TestCase):
    """Chaque valeur est un calcul math-hammer classique fait à la main."""

    def test_bolt_rifles_vs_marines(self):
        # 10 tirs, touche 3+ (2/3), blesse 4+ (1/2), sauvegarde 4+ après AP-1 (échec 1/2)
        e = expected_attacks(bolt_rifles(5), MARINE)
        self.assertAlmostEqual(e.attacks, 10)
        self.assertAlmostEqual(e.hits, 10 * 2 / 3)
        self.assertAlmostEqual(e.wounds, 10 * 2 / 3 / 2)
        self.assertAlmostEqual(e.unsaved, 10 * 2 / 3 / 2 / 2)
        self.assertAlmostEqual(e.damage, 1.6667, places=3)

    def test_hail_of_bolts(self):
        # +2 A sur les bolt rifles : 20 tirs
        e = expected_attacks(AttackProfile(D("4"), 3, 4, -1, D("1"), count=5), MARINE)
        self.assertAlmostEqual(e.damage, 3.3333, places=3)

    def test_cover_is_minus_one_to_hit(self):
        e = expected_attacks(bolt_rifles(5, hit_modifier=-1), MARINE)
        self.assertEqual(hit_roll_needed(bolt_rifles(5, hit_modifier=-1)), 4)
        self.assertAlmostEqual(e.damage, 10 * 1 / 2 * 1 / 2 * 1 / 2, places=4)

    def test_heavy_cancels_cover_and_cap(self):
        self.assertEqual(hit_roll_needed(bolt_rifles(hit_modifier=+1 - 1)), 3)
        self.assertEqual(hit_roll_needed(bolt_rifles(hit_modifier=+3)), 2)  # plafonné à +1
        self.assertEqual(hit_roll_needed(bolt_rifles(hit_modifier=-4)), 4)  # plafonné à -1
        e_plus = expected_attacks(bolt_rifles(hit_modifier=+1), MARINE)
        self.assertAlmostEqual(e_plus.hits, 10 * 5 / 6)

    def test_oath_of_moment_rerolls(self):
        # relance des touches ratées : 2/3 + 1/3 · 2/3 = 8/9
        e = expected_attacks(bolt_rifles(5, reroll_hits="fails"), MARINE)
        self.assertAlmostEqual(e.hits, 10 * 8 / 9)
        self.assertAlmostEqual(e.damage, 10 * 8 / 9 / 4, places=4)
        # relance des 1 seulement : 2/3 + 1/6 · 2/3 = 7/9
        e1 = expected_attacks(bolt_rifles(5, reroll_hits="ones"), MARINE)
        self.assertAlmostEqual(e1.hits, 10 * 7 / 9)

    def test_torrent_hand_flamer(self):
        # A D6 (3.5), touche auto, S3 vs T4 → 5+ (1/3), AP0 vs 3+ → échec 1/3
        p = AttackProfile(D("D6"), None, 3, 0, D("1"))
        e = expected_attacks(p, MARINE)
        self.assertAlmostEqual(e.hits, 3.5)
        self.assertEqual(e.critical_hits, 0.0)  # pas de jet de touche → pas de critique
        self.assertAlmostEqual(e.damage, 3.5 / 3 / 3, places=4)

    def test_thunder_hammer_devastating(self):
        # A3 WS4+ S8 AP-2 D2 [DW] vs T4 Sv3+ : touche 1/2 ; S8 ≥ 2T → 2+ ; crit 1/6 sans sauvegarde ;
        # le reste (4/6) sauvegardé à 5+ (échec 2/3) ; D2
        p = AttackProfile(D("3"), 4, 8, -2, D("2"), devastating_wounds=True)
        e = expected_attacks(p, MARINE)
        self.assertAlmostEqual(e.hits, 1.5)
        self.assertAlmostEqual(e.wounds, 1.5 * 5 / 6)
        self.assertAlmostEqual(e.devastating, 1.5 / 6)
        self.assertAlmostEqual(e.saves_attempted, 1.5 * 4 / 6)
        self.assertAlmostEqual(e.unsaved, 1.5 * 4 / 6 * 2 / 3)
        self.assertAlmostEqual(e.damage, (1.5 * 4 / 6 * 2 / 3 + 1.5 / 6) * 2, places=4)

    def test_plasma_supercharge(self):
        # A1 BS3+ S8 AP-3 D2 : 2+ pour blesser, sauvegarde 6+ (échec 5/6)
        p = AttackProfile(D("1"), 3, 8, -3, D("2"))
        e = expected_attacks(p, MARINE)
        self.assertAlmostEqual(e.damage, 2 / 3 * 5 / 6 * 5 / 6 * 2, places=4)

    def test_no_save_possible(self):
        p = AttackProfile(D("1"), 3, 4, -4, D("1"))
        e = expected_attacks(p, MARINE)
        self.assertAlmostEqual(e.unsaved, e.saves_attempted)  # 7+ : tout passe
        e_inv = expected_attacks(p, Defender(4, 3, invuln=4))
        self.assertAlmostEqual(e_inv.unsaved, e_inv.saves_attempted / 2)

    def test_anti_and_sustained_and_lethal(self):
        # Anti-infantry 4+ : critique sur 4+ → avec DW, 1/2 des touches passent sans sauvegarde
        p = AttackProfile(D("6"), 3, 4, 0, D("1"), critical_wound_on=4, devastating_wounds=True)
        e = expected_attacks(p, MARINE)
        self.assertAlmostEqual(e.critical_wounds, 6 * 2 / 3 * 1 / 2)
        self.assertAlmostEqual(e.devastating, e.critical_wounds)
        # Sustained Hits 1 : 1/6 des attaques donnent une touche de plus
        s = expected_attacks(AttackProfile(D("6"), 3, 4, 0, D("1"), sustained_hits=DiceExpr.fixed(1)), MARINE)
        self.assertAlmostEqual(s.hits, 6 * 2 / 3 + 6 / 6)
        # Lethal Hits : les touches critiques blessent automatiquement
        l = expected_attacks(AttackProfile(D("6"), 3, 4, 0, D("1"), lethal_hits=True), MARINE)
        self.assertAlmostEqual(l.wounds, 6 / 6 + (6 * 2 / 3 - 6 / 6) / 2)

    def test_feel_no_pain(self):
        e = expected_attacks(bolt_rifles(5), Defender(4, 3, feel_no_pain=5))
        self.assertAlmostEqual(e.damage_per_wound, 1 - 2 / 6)
        self.assertAlmostEqual(e.damage, 10 / 6 * 4 / 6, places=4)

    def test_estimate_models_slain(self):
        self.assertAlmostEqual(estimate_models_slain(4, D("1"), 2), 2.0)  # 4 blessures D1 sur W2 → 2 morts
        self.assertAlmostEqual(estimate_models_slain(3, D("2"), 2), 3.0)  # D2 sur W2 : un mort par blessure
        self.assertAlmostEqual(estimate_models_slain(1, D("6"), 2), 1.0)  # dégâts plafonnés à W
        # D3 sur W2 : E[min(D3, 2)] = (1 + 2 + 2)/3 = 5/3
        self.assertAlmostEqual(estimate_models_slain(3, D("D3"), 2), 3 * (5 / 3) / 2)
        self.assertEqual(estimate_models_slain(3, D("2"), 0), 0.0)

    def test_hazardous(self):
        # 1 ou 2 sur D6 → 1 BM (infanterie), 3 (véhicule / monstre)
        self.assertAlmostEqual(expected_hazardous_mortal_wounds(3, False), 3 * 2 / 6)
        self.assertAlmostEqual(expected_hazardous_mortal_wounds(1, True), 1.0)
        rng = random.Random(1)
        total = sum(hazardous_test(rng, 1, False) for _ in range(6000))
        self.assertAlmostEqual(total / 6000, 1 / 3, delta=0.03)


class MonteCarloAgreement(unittest.TestCase):
    """La version aux dés doit converger vers la version en espérance."""

    SCENARIOS = [
        ("bolt rifles", bolt_rifles(10), MARINE),
        ("bolt rifles couvert + Oath", bolt_rifles(10, hit_modifier=-1, reroll_hits="fails"), MARINE),
        ("hand flamers torrent", AttackProfile(D("D6"), None, 3, 0, D("1"), count=6), MARINE),
        ("thunder hammers DW", AttackProfile(D("3"), 4, 8, -2, D("2"), count=6, devastating_wounds=True), MARINE),
        ("plasma vs invul + FNP", AttackProfile(D("2"), 3, 8, -3, D("D3"), count=6), Defender(4, 3, invuln=4, feel_no_pain=5)),
        ("anti 4+ DW twin-linked", AttackProfile(D("2"), 3, 5, -1, D("D6+1"), count=6, critical_wound_on=4, devastating_wounds=True, reroll_wounds="fails"), Defender(9, 2)),
        ("sustained D3 + lethal + reroll ones", AttackProfile(D("4"), 4, 4, 0, D("1"), count=6, sustained_hits=D("D3"), lethal_hits=True, reroll_hits="ones", reroll_wounds="ones"), Defender(5, 4)),
        ("S2 vs T9 malus", AttackProfile(D("3"), 2, 2, 0, D("1"), count=6, hit_modifier=-1, wound_modifier=-1), Defender(9, 3)),
    ]

    def test_agreement(self):
        rng = random.Random(2024)
        n = 3000
        for name, profile, defender in self.SCENARIOS:
            exp = expected_attacks(profile, defender)
            acc = {"attacks": 0.0, "hits": 0.0, "wounds": 0.0, "critical_wounds": 0.0, "devastating": 0.0, "unsaved": 0.0, "damage": 0.0}
            for _ in range(n):
                out = resolve_attacks(profile, defender, rng)
                acc["attacks"] += out.attacks
                acc["hits"] += out.hits
                acc["wounds"] += out.wounds
                acc["critical_wounds"] += out.critical_wounds
                acc["devastating"] += out.devastating
                acc["unsaved"] += out.unsaved
                acc["damage"] += out.total_damage
            for key, total in acc.items():
                mc = total / n
                ref = getattr(exp, key)
                tol = max(0.06, 0.04 * ref)  # bruit statistique pour n = 3000 répétitions
                self.assertAlmostEqual(mc, ref, delta=tol, msg=f"{name} / {key} : dés {mc:.3f} vs espérance {ref:.3f}")

    def test_outcome_consistency(self):
        rng = random.Random(7)
        p = AttackProfile(D("3"), 4, 8, -2, D("2"), count=5, devastating_wounds=True, sustained_hits=D("1"))
        for _ in range(300):
            out = resolve_attacks(p, MARINE, rng)
            self.assertLessEqual(out.critical_hits, out.hits)
            self.assertLessEqual(out.wounds, out.hits)
            self.assertLessEqual(out.critical_wounds, out.wounds)
            self.assertEqual(out.saves_attempted, out.wounds - out.devastating)
            self.assertLessEqual(out.unsaved, out.saves_attempted)
            self.assertEqual(len(out.damage), out.unsaved + out.devastating)
            self.assertTrue(all(d == 2 for d in out.damage))


if __name__ == "__main__":
    unittest.main()
