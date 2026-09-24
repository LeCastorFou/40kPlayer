"""Bot aléatoire : choisit uniformément parmi les actions légales.

Il sert à valider le moteur (toute partie doit se terminer sans erreur) et de plancher
de mesure pour les bots suivants.
"""

from __future__ import annotations

import random

from ..engine.actions import Action, Decision, EndPhaseAction
from ..engine.state import GameState


class RandomAgent:
    def __init__(self, name: str = "Aléatoire", seed: int = 0):
        self.name = name
        self.rng = random.Random(seed)

    def choose(self, state: GameState, decision: Decision) -> Action:
        options = decision.options
        if decision.kind == "select_unit":
            # ne termine la phase que s'il n'y a plus d'unité à activer
            units = [o for o in options if not isinstance(o, EndPhaseAction)]
            if units:
                options = units
        return self.rng.choice(options)
