"""Pilote de partie : relie la machine à états (:class:`~fortyk.engine.engine.Engine`) aux agents.

Le :class:`Game` demande à chaque décision l'action de l'agent du camp concerné, la valide et la
passe au moteur. Il relaie aussi les événements du moteur : lignes de journal (``on_log``), fin de
tour (``on_turn_end``) et « action adverse terminée » vers les agents qui ont une méthode
``observe`` (mode pas à pas de l'interface web ; plus tard, les stratagèmes réactifs).

Toute la logique de jeu est dans le moteur : le pilote ne garde aucun état de partie, il suffit de
l'état (:class:`GameState`) pour reprendre, cloner ou rejouer une partie.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .actions import Action, ContinueAction, Decision
from .engine import PHASE_STARTS, Engine, GameResult, IllegalAction
from .state import GameState, Unit

__all__ = ["Game", "GameResult", "IllegalAction"]

_NEXT_PHASE_START = {
    "movement": "shoot_start",
    "shooting": "charge_start",
    "charge": "fight_start",
    "fight": "end_turn",
}


class Game:
    def __init__(self, state: GameState, agents: Dict[str, object], verbose: bool = True):
        self.state = state
        self.agents = agents
        self.verbose = verbose
        self.on_turn_end: Optional[Callable[[GameState], None]] = None  #: callback(state) après chaque tour (rendu PNG…)
        self.on_log: Optional[Callable[[str], None]] = None  #: callback(str) à chaque ligne de journal (affichage en direct)
        self.engine = Engine(listener=self._on_event)

    # ------------------------------------------------------------ événements

    def _on_event(self, state: GameState, event: dict) -> None:
        kind = event["kind"]
        if kind == "log":
            if self.on_log:
                self.on_log(event["text"])
        elif kind == "notify":
            for side, agent in self.agents.items():
                if side != event["side"] and hasattr(agent, "observe"):
                    agent.observe(state, Decision("observe", side, [ContinueAction()], note=event["message"], phase=state.phase))
        elif kind == "turn_end":
            if self.on_turn_end:
                self.on_turn_end(state)

    # ------------------------------------------------------------ pilotage

    def _ask(self, decision: Decision) -> Action:
        agent = self.agents[decision.side]
        while True:
            action = agent.choose(self.state, decision)
            if action in decision.options:
                return action
            error = self.engine.free_action_error(self.state, decision, action)
            if error is None:
                return action
            if getattr(agent, "interactive", False):
                say = getattr(agent, "out", print)
                say(f"Action illégale : {error}")
                continue
            raise ValueError(f"{decision.side} a renvoyé une action illégale : {action} ({error})")

    def _drive(self) -> None:
        while True:
            decision = self.engine.decision(self.state)
            if decision is None:
                return
            self.engine.step(self.state, self._ask(decision), validate=False)

    def play(self, deploy: bool = True, first_player: Optional[str] = None) -> GameResult:
        """Joue la partie. ``deploy=False`` part des positions actuelles (scénarios de test) ;
        ``first_player`` force le jet d'initiative."""
        self.engine.start(self.state, deploy=deploy, first_player=first_player)
        self._drive()
        return self.engine.result(self.state)

    def play_phase(self, side: str, phase: str) -> None:
        """Joue une seule phase du tour de ``side`` depuis l'état courant (scénarios de test)."""
        self.engine.stop_at = {_NEXT_PHASE_START[phase]}
        try:
            self.engine.start_at(self.state, phase, side)
            self._drive()
        finally:
            self.engine.stop_at = set()

    # ------------------------------------------------------------ compatibilité (tests, scripts)

    def charge_phase(self, side: str) -> None:
        self.play_phase(side, "charge")

    def charge_targets(self, unit: Unit) -> List[Unit]:
        return self.engine.charge_targets(self.state, unit)

    def charge_ineligibility(self, unit: Unit) -> Optional[str]:
        return self.engine.charge_ineligibility(self.state, unit)

    def deployment_ok(self, unit: Unit, x: float, y: float) -> bool:
        return self.engine.deployment_ok(self.state, unit, x, y)

    def _on_unit_destroyed(self, unit: Unit) -> None:
        self.engine.on_unit_destroyed(self.state, unit)

    def free_action_error(self, decision: Decision, action: Action) -> Optional[str]:
        return self.engine.free_action_error(self.state, decision, action)
