"""Interface minimale d'un agent."""

from __future__ import annotations

from typing import Protocol

from ..engine.actions import Action, Decision
from ..engine.state import GameState


class Agent(Protocol):
    name: str

    def choose(self, state: GameState, decision: Decision) -> Action:  # pragma: no cover - protocole
        ...
