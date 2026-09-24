"""Machine à états (decision / step / clone / replay) et géométrie vectorisée (équivalence avec la référence)."""

import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fortyk.data import default_raw_dir, load_catalog  # noqa: E402
from fortyk.engine import Engine, IllegalAction  # noqa: E402
from fortyk.engine.actions import EndPhaseAction, SelectUnitAction  # noqa: E402
from fortyk.engine.army import TOY_ROSTERS_INFANTRY  # noqa: E402
from fortyk.engine.fastgeo import LineOfSight  # noqa: E402
from fortyk.engine.geometry import Disk, is_visible, visibility  # noqa: E402
from fortyk.engine.layout import load_layout  # noqa: E402
from fortyk.engine.movement import MoveKind, apply_translation, legal_translation  # noqa: E402
from fortyk.toy import new_toy_state  # noqa: E402

HAS_DATA = (default_raw_dir() / "Datasheets.csv").exists()


def play_random(engine, state, rng, max_steps=None):
    n = 0
    while not engine.is_over(state) and (max_steps is None or n < max_steps):
        d = engine.decision(state)
        opts = d.options
        if d.kind == "select_unit":
            opts = [o for o in opts if not isinstance(o, EndPhaseAction)] or opts
        engine.step(state, rng.choice(opts))
        n += 1
    return n


def fingerprint(state):
    return (
        tuple((m.id, round(m.x, 9), round(m.y, 9), m.wounds, m.alive) for u in state.units.values() for m in u.models),
        tuple(sorted(state.scoreboard.totals().items())),
        state.battle_round,
        state.phase,
        state.flow.step,
    )


class LineOfSightEquivalenceTests(unittest.TestCase):
    """L'index numpy donne exactement les réponses de la référence Python (geometry.py)."""

    def test_random_pairs_match_reference(self):
        terrain = load_layout("layout_a").terrain_objects()
        los = LineOfSight(terrain)
        rng = random.Random(7)
        pairs = []
        for _ in range(600):
            pairs.append((Disk(rng.uniform(1, 43), rng.uniform(1, 59), rng.choice([0.63, 1.18, 1.77])),
                          Disk(rng.uniform(1, 43), rng.uniform(1, 59), rng.choice([0.63, 1.18, 1.77]))))
        for _ in range(400):  # paires proches : orteils dans les décors, bords
            a = Disk(rng.uniform(1, 43), rng.uniform(1, 59), 0.63)
            pairs.append((a, Disk(a.x + rng.uniform(-5, 5), a.y + rng.uniform(-5, 5), 0.63)))
        for a, b in pairs:
            self.assertEqual(los.visible(a, b), is_visible(a, b, terrain), (a, b))
            self.assertAlmostEqual(los.fraction(a, b), visibility(a, b, terrain), places=9, msg=(a, b))

    def test_block_prefetch_equals_pairwise(self):
        terrain = load_layout("layout_a").terrain_objects()
        A = [Disk(20 + i * 1.3, 20, 0.63) for i in range(5)]
        B = [Disk(10 + i * 1.3, 45, 0.63) for i in range(5)] + [Disk(36.6, 36.3, 1.18)]
        bulk = LineOfSight(terrain)
        bulk.prefetch(A, B)
        single = LineOfSight(terrain)
        for a in A:
            for b in B:
                self.assertEqual(bulk.visible(a, b), single.visible(a, b))
                self.assertEqual(bulk.fraction(a, b), single.fraction(a, b))


@unittest.skipUnless(HAS_DATA, "CSV Wahapedia absents")
class EngineCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def test_step_protocol_and_full_game(self):
        eng = Engine()
        s = new_toy_state(self.cat, seed=4)
        eng.start(s)
        d = eng.decision(s)
        self.assertEqual((d.kind, d.side, d.phase), ("select_unit", "attacker", "deployment"))
        self.assertIs(eng.decision(s), d)  # mise en cache jusqu'au prochain step
        with self.assertRaises(IllegalAction):
            eng.step(s, SelectUnitAction("EC1"))  # pas une unité de l'attaquant à déployer
        n = play_random(eng, s, random.Random(4))
        self.assertTrue(eng.is_over(s))
        self.assertIsNone(eng.decision(s))
        self.assertEqual(len(s.history), n)
        res = eng.result(s)
        self.assertEqual(res.rounds_played, s.battle_round)
        self.assertEqual(res.scores, s.scoreboard.totals())
        # journal structuré parallèle au texte
        self.assertEqual(len(s.events), len(s.log))
        kinds = {e["kind"] for e in s.events}
        self.assertTrue({"deploy", "roll_off", "round", "turn", "move", "result"} <= kinds, kinds)
        self.assertEqual(s.events[-1]["scores"], s.scoreboard.totals())

    def test_clone_is_independent_and_replays_same_future(self):
        eng = Engine()
        s = new_toy_state(self.cat, seed=9)
        eng.start(s)
        play_random(eng, s, random.Random(1), max_steps=25)  # au milieu du round 1
        c = s.clone()
        self.assertEqual(fingerprint(c), fingerprint(s))
        # jouer la copie ne change pas l'original
        before = fingerprint(s)
        play_random(eng, c, random.Random(2))
        self.assertEqual(fingerprint(s), before)
        self.assertTrue(eng.is_over(c))
        # mêmes dés + mêmes choix ⇒ même futur ; autre graine ⇒ (en général) autre futur
        a, b = s.clone(), s.clone()
        play_random(eng, a, random.Random(3))
        play_random(eng, b, random.Random(3))
        self.assertEqual(fingerprint(a), fingerprint(b))
        # copie légère pour la recherche : plus de journal, mais le jeu est le même
        q = s.clone(record=False)
        play_random(eng, q, random.Random(3))
        self.assertEqual(fingerprint(q), fingerprint(a))
        self.assertEqual(q.log, [])

    def test_replay_from_history_and_json(self):
        from fortyk.engine.record import load_game, save_game

        eng = Engine()
        init = new_toy_state(self.cat, seed=12)
        s = init.clone()
        eng.start(s)
        play_random(eng, s, random.Random(5))
        r = eng.replay(init, s.history)
        self.assertEqual(fingerprint(r), fingerprint(s))
        self.assertEqual(r.log, s.log)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partie.json"
            save_game(path, s, {"seed": 12, "layout": "layout_a"})
            doc = load_game(path)
            r2 = eng.replay(new_toy_state(self.cat, seed=doc["meta"]["seed"]), doc["history"])
            self.assertEqual(fingerprint(r2), fingerprint(s))

    def test_play_single_phase(self):
        from fortyk.agents import RandomAgent
        from fortyk.engine.game import Game

        s = new_toy_state(self.cat, seed=2, rosters=TOY_ROSTERS_INFANTRY)
        for uid, (x, y) in {"SM1": (10, 22), "SM2": (40, 4), "EC1": (10, 29), "EC2": (40, 56)}.items():
            u = s.units[uid]
            c = u.centroid
            apply_translation(u, x - c[0], y - c[1])
        g = Game(s, {"attacker": RandomAgent(seed=1), "defender": RandomAgent(seed=2)})
        g.play_phase("attacker", "shooting")
        self.assertEqual(s.flow.step, "charge_start")  # arrêtée à l'entrée de la phase suivante
        self.assertTrue(any(line.startswith("Tir") for line in s.log))

    def test_fast_move_legality_is_never_laxer_than_sampled_reference(self):
        """Le test exact des trajectoires (segment contre socle) refuse tout ce que l'ancien test
        échantillonné refusait ; il peut refuser en plus un frôlement que l'échantillonnage ratait."""
        import math

        from fortyk.engine.geometry import disks_overlap, within

        s = new_toy_state(self.cat, seed=3, rosters=TOY_ROSTERS_INFANTRY)
        rng = random.Random(11)

        def sampled(unit, dx, dy, kind, max_d):
            length = math.hypot(dx, dy)
            if length > max_d + 1e-9:
                return False
            if length <= 1e-9:
                return True
            friends = [m.disk for u in s.units_of(unit.side) if u.id != unit.id for m in u.alive_models]
            enemies = [m.disk for m in s.enemy_models(unit.side)]
            steps = max(2, int(math.ceil(length / 0.5)) + 1)
            for m in unit.alive_models:
                end = Disk(m.x + dx, m.y + dy, m.radius)
                if not s.on_board(end) or any(disks_overlap(end, o) for o in friends + enemies):
                    return False
                # V11 : fin hors portée d'engagement ; trajet sans traverser de socle ennemi ; amis traversables
                if kind in ("normal", "advance", "fall_back") and any(within(end, e, 2.0) for e in enemies):
                    return False
                for k in range(1, steps):
                    t = k / (steps - 1)
                    pos = Disk(m.x + dx * t, m.y + dy * t, m.radius)
                    if kind in ("normal", "advance", "fall_back") and any(disks_overlap(pos, e) for e in enemies):
                        return False
            return True

        stricter = 0
        for trial in range(300):
            for uid, (x, y) in {"SM1": (rng.uniform(4, 40), rng.uniform(4, 30)), "SM2": (rng.uniform(4, 40), rng.uniform(4, 30)),
                                "EC1": (rng.uniform(4, 40), rng.uniform(20, 56)), "EC2": (rng.uniform(4, 40), rng.uniform(30, 56))}.items():
                u = s.units[uid]
                c = u.centroid
                apply_translation(u, x - c[0], y - c[1])
            u = s.units["SM1"]
            a = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(0, 7)
            dx, dy = dist * math.cos(a), dist * math.sin(a)
            for kind in (MoveKind.NORMAL, MoveKind.FALL_BACK):
                fast = legal_translation(s, u, dx, dy, kind, 6.0)
                ref = sampled(u, dx, dy, kind, 6.0)
                if fast:
                    self.assertTrue(ref, (trial, kind, dx, dy))
                elif ref:
                    stricter += 1
        self.assertLess(stricter, 15)  # quelques frôlements seulement


if __name__ == "__main__":
    unittest.main()
