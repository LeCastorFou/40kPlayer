"""Moteur de jeu en machine à états : ``decision(state)`` → ``step(state, action)`` → …

Toute la progression d'une partie vit dans l'état (:class:`~fortyk.engine.state.Flow`, le
« compteur de programme ») et non dans la pile d'appels : on peut donc cloner une partie à
n'importe quelle décision (:meth:`GameState.clone`), explorer plusieurs coups sur des copies et
rejouer une partie à l'identique depuis sa graine et la liste de ses actions. C'est ce qu'exigent
la recherche arborescente et l'entraînement par self-play.

Protocole ::

    engine = Engine()
    engine.start(state)                    # déploiement, jet d'initiative… jusqu'à la 1re décision
    while not engine.is_over(state):
        d = engine.decision(state)         # Decision : qui décide, quoi, options légales
        engine.step(state, choose(d))      # applique, puis avance jusqu'à la décision suivante
    engine.result(state)

Une étape est soit *automatique* (le moteur l'exécute : battle-shock, score, jets…), soit une
*décision*. ``step`` applique l'action puis enchaîne les étapes automatiques jusqu'à la décision
suivante. Les agents interactifs (humain) peuvent proposer des actions hors de la liste des options
(placement libre des figurines) : ``free_action_error`` les valide.

Le moteur lui-même est sans état (hors l'écouteur d'événements) : une instance sert pour toutes
les parties et toutes les copies. L'écouteur reçoit les événements au fil de l'eau (journal,
« action adverse terminée » pour le mode pas à pas, fin de tour) ; les simulations n'en ont pas.

Règles de séquence (validées avec Valentin) : déploiement alterné (le joueur choisit l'unité),
jet d'initiative, 5 rounds ; Commandement (battle-shock, Oath of Moment, score) → Mouvement →
Tir → Charge (V11 : déclaration, 2D6, cibles atteignables socle à socle) → Combat (Fights First
en alternance en commençant par le joueur actif, puis les autres en commençant par celui qui n'a
pas activé en dernier) → fin de tour (score).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..data.parse import parse_dice
from .actions import (
    Action,
    AutoChargeMoveAction,
    ChargeAction,
    DeclareAdvanceAction,
    DeclareChargeAction,
    Decision,
    DeployAction,
    DeployModelsAction,
    DisembarkAction,
    DisembarkModelsAction,
    EmbarkAction,
    EndPhaseAction,
    FightAction,
    ModelMoveAction,
    MoveAction,
    OathAction,
    SelectUnitAction,
    FREE_ACTIONS,
    ReserveAction,
    ShootAction,
    StratagemAction,
    UseStratagemAction,
)
import numpy as np

from .combat import (
    apply_mortal_wounds,
    explosives_targets,
    fight_targets,
    is_character_model,
    is_vehicle_or_monster,
    resolve_fight,
    resolve_shooting,
    shooting_ineligibility,
    shooting_targets,
    snap_targets,
)
from .effects import effect_total, has_effect, prune_effects
from .reserves import formation_at, has_deep_strike, has_infiltrators, infiltrate_error, ingress_candidates, ingress_error, reserve_error
from .stratagems import CORE_STRATAGEMS, WINDOWS, cost_of, record_use, unavailable

from .fastgeo import deployment_grid_legal, shape_arrays
from .geometry import disk_gap, extreme_points, point_in_polygon, within
from .movement import (
    FormationMove,
    MoveKind,
    apply_model_positions,
    apply_translation,
    auto_charge_move,
    candidate_moves,
    charge_gap,
    charge_reachable_targets,
    check_charge_positions,
    check_model_positions,
    consolidate,
    legal_translation,
    pile_in,
    shrink_to_legal,
)
from .state import SIDES, Flow, GameState, Unit, other_side
from .transport import (
    check_disembark_positions,
    check_scout_positions,
    disembark_candidates,
    disembark_error,
    embark_options,
    emergency_positions,
    disembark_mode,
    place_formation,
    ring_placements,
    scout_distance,
    scout_moves,
)

__all__ = ["Engine", "GameResult", "IllegalAction", "PHASE_STARTS"]


class IllegalAction(ValueError):
    """Action refusée par le moteur (hors options et invalide comme action libre)."""


@dataclass
class GameResult:
    winner: Optional[str]
    scores: Dict[str, int]
    rounds_played: int
    log: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        w = self.winner or "égalité"
        return f"{w} — attaquant {self.scores['attacker']} VP, défenseur {self.scores['defender']} VP ({self.rounds_played} rounds)"


#: étape d'entrée de chaque phase (pour jouer une phase isolée dans les tests et scénarios)
PHASE_STARTS = {
    "command": "turn_start",
    "movement": "move_start",
    "shooting": "shoot_start",
    "charge": "charge_start",
    "fight": "fight_start",
    "end_turn": "end_turn",
}

#: étapes de décision → genre de Decision
DECISION_KINDS = {
    "deploy_select": "select_unit",
    "deploy_place": "deploy",
    "scout_select": "select_unit",
    "scout": "scout",
    "oath": "oath",
    "move_select": "select_unit",
    "move": "move",
    "advance_move": "advance_move",
    "disembark": "disembark",
    "embark": "embark",
    "shoot_select": "select_unit",
    "shoot": "shoot",
    "charge_select": "select_unit",
    "charge_declare": "charge_declare",
    "charge_target": "charge_target",
    "charge_move": "charge_move",
    "fight": "fight",
    "stratagem": "stratagem",
    "ingress": "ingress",
}


Listener = Callable[[GameState, dict], None]


class Engine:
    """Machine à états du jeu. ``listener(state, event)`` reçoit chaque événement émis."""

    def __init__(self, listener: Optional[Listener] = None):
        self.listener = listener
        self.stop_at: set = set()  #: étapes automatiques où s'arrêter (jouer une seule phase)

    # ================================================================ API

    def start(self, state: GameState, deploy: bool = True, first_player: Optional[str] = None) -> None:
        """Démarre une partie : déploiement (ou positions actuelles si ``deploy=False``), jet
        d'initiative (forcé par ``first_player``), puis avance jusqu'à la première décision."""
        state.flow = Flow(step="setup", deploy=deploy, forced_first_player=first_player)
        self._advance(state)

    def start_at(self, state: GameState, phase: str, side: str) -> None:
        """Place la partie au début d'une phase du tour de ``side`` (scénarios de test). Les unités
        sont supposées déployées et les drapeaux de tour dans l'état voulu."""
        flow = state.flow or Flow()
        flow.step = PHASE_STARTS[phase]
        flow.active = side
        flow.turn_index = 0 if side == state.first_player else 1
        flow.deployed = list(state.units)
        flow.decision = None
        state.flow = flow
        state.side_to_move = side
        if state.battle_round == 0:
            state.battle_round = 1
        self._advance(state)

    def is_over(self, state: GameState) -> bool:
        return state.flow is not None and state.flow.step == "game_over"

    def result(self, state: GameState) -> GameResult:
        return GameResult(winner=state.scoreboard.leader(), scores=state.scoreboard.totals(), rounds_played=state.battle_round, log=list(state.log))

    def decision(self, state: GameState) -> Optional[Decision]:
        """Décision en attente (mise en cache jusqu'au prochain ``step``) ; None si la partie est finie
        ou arrêtée sur une étape de ``stop_at``."""
        flow = state.flow
        if flow is None or flow.step not in DECISION_KINDS:
            return None
        if flow.decision is None:
            flow.decision = getattr(self, "_d_" + flow.step)(state, flow)
        return flow.decision

    def legal_actions(self, state: GameState) -> List[Action]:
        d = self.decision(state)
        return list(d.options) if d is not None else []

    def to_move(self, state: GameState) -> Optional[str]:
        d = self.decision(state)
        return d.side if d is not None else None

    def step(self, state: GameState, action: Action, validate: bool = True) -> None:
        """Applique ``action`` à la décision en attente, puis avance jusqu'à la décision suivante."""
        d = self.decision(state)
        if d is None:
            raise IllegalAction("aucune décision en attente (partie terminée ?)")
        if validate and action not in d.options:
            err = self.free_action_error(state, d, action)
            if err is not None:
                raise IllegalAction(err)
        flow = state.flow
        flow.decision = None
        state.history.append((d.side, action))
        getattr(self, "_a_" + flow.step)(state, flow, action)
        self._advance(state)

    def free_error(self, state: GameState, side: str, action) -> Optional[str]:
        """Action libre (effet manuel, stratagème du panneau) de ``side`` : message d'erreur ou None."""
        from .free import free_error

        return free_error(self, state, side, action)

    def apply_free(self, state: GameState, side: str, action, validate: bool = True) -> str:
        """Joue une action libre à n'importe quel moment : la décision en cours n'avance pas (elle est
        recalculée, l'état ayant changé)."""
        from .free import apply_free

        if validate:
            err = self.free_error(state, side, action)
            if err is not None:
                raise IllegalAction(err)
        text = apply_free(self, state, side, action)
        self._say(state, text, "stratagem" if isinstance(action, UseStratagemAction) else "manual", side=side)
        self._notify(state, side, text)
        state.history.append((side, action))
        if state.flow is not None:
            state.flow.decision = None
        return text

    # ================================================================ événements

    def _emit(self, state: GameState, event: dict) -> None:
        if self.listener is not None:
            self.listener(state, event)

    def _say(self, state: GameState, msg: str, kind: str = "info", **data) -> None:
        """Ligne de journal + événement structuré (kind : round, turn, deploy, move, shoot, charge,
        fight, score, battle_shock, oath, deadly_demise, result…)."""
        if state.recording:
            state.log.append(msg)
            ev = {"i": len(state.events), "round": state.battle_round, "phase": state.phase, "kind": kind, "text": msg}
            if data:
                ev.update(data)
            state.events.append(ev)
        if self.listener is not None:
            self.listener(state, {"kind": "log", "text": msg, "type": kind})

    def _notify(self, state: GameState, acting_side: str, message: str) -> None:
        """Fin d'une action de ``acting_side`` : point d'observation pour l'adversaire (mode pas à pas,
        futurs stratagèmes réactifs)."""
        if self.listener is not None:
            self.listener(state, {"kind": "notify", "side": acting_side, "message": message})

    # ================================================================ progression

    def _advance(self, state: GameState) -> None:
        flow = state.flow
        while flow.step not in DECISION_KINDS and flow.step != "game_over" and flow.step not in self.stop_at:
            getattr(self, "_auto_" + flow.step)(state, flow)

    # ---------------------------------------------------------------- mise en place

    def _auto_setup(self, s: GameState, flow: Flow) -> None:
        if flow.deploy:
            s.phase = "deployment"
            flow.deployed = []
            flow.queues = {side: [u.id for u in s.units_of(side)] for side in SIDES}
            flow.active = "attacker"
            flow.step = "deploy_next"
        else:
            flow.deployed = list(s.units)
            for u in s.units.values():
                u.reset_turn_flags()
            flow.step = "roll_off"

    def _auto_deploy_next(self, s: GameState, flow: Flow) -> None:
        q = flow.queues
        if not any(q.values()):
            flow.step = "roll_off"
            return
        if not q[flow.active]:
            flow.active = other_side(flow.active)
        if len(q[flow.active]) == 1:
            flow.unit_id = q[flow.active][0]
            flow.step = "deploy_place"
        else:
            flow.step = "deploy_select"

    def _d_deploy_select(self, s: GameState, flow: Flow) -> Decision:
        return Decision("select_unit", flow.active, [SelectUnitAction(uid) for uid in flow.queues[flow.active]], phase="deployment")

    def _a_deploy_select(self, s: GameState, flow: Flow, action: SelectUnitAction) -> None:
        flow.unit_id = action.unit_id
        flow.step = "deploy_place"

    def _d_deploy_place(self, s: GameState, flow: Flow) -> Decision:
        unit = s.unit(flow.unit_id)
        if unit.has_keyword("Aircraft"):  # 23.01 : toujours en réserve stratégique
            return Decision("deploy", flow.active, [ReserveAction(unit.id)], unit_id=unit.id, note="AIRCRAFT : en réserve stratégique (23.01)")
        options: List[Action] = self.deployment_candidates(s, unit) or self.deployment_candidates(s, unit, step=1.0)
        if has_infiltrators(unit):  # 24.20 : n'importe où à plus de 8" de la zone adverse et de l'ennemi
            options += [DeployAction(unit.id, x, y) for x, y in ingress_candidates(s, unit, step=2.0, limit=16, mode="infiltrate")]
        # commencer la partie à bord d'un transport ami déjà déployé (ou en réserve)
        options += [EmbarkAction(unit.id, t.id) for t in embark_options(s, unit, deployed=flow.deployed, check_range=False)]
        notes = []
        if reserve_error(s, unit) is None:
            options.append(ReserveAction(unit.id))
            notes.append("réserve stratégique possible" + (" (Deep Strike : arrivée n'importe où à plus de 8\" de l'ennemi)" if has_deep_strike(unit) else ""))
        if has_infiltrators(unit):
            notes.append("Infiltrators : n'importe où à plus de 8\" de la zone adverse et de toute unité ennemie")
        if not options:
            raise RuntimeError(f"impossible de déployer {unit.name}")
        return Decision("deploy", flow.active, options, unit_id=unit.id, note=" ; ".join(notes))

    def _a_deploy_place(self, s: GameState, flow: Flow, action) -> None:
        unit = s.unit(flow.unit_id)
        if isinstance(action, ReserveAction):
            unit.in_reserve = True
            unit.reset_turn_flags()
            flow.deployed.append(unit.id)
            flow.queues[flow.active].remove(unit.id)
            self._say(s, f"Déploiement : {unit.name} est placée en réserve stratégique" + (" (Deep Strike)" if has_deep_strike(unit) else ""),
                      "reserve", unit=unit.id)
            self._notify(s, flow.active, f"{unit.name} en réserve stratégique")
            flow.active = other_side(flow.active)
            flow.step = "deploy_next"
            return
        if isinstance(action, EmbarkAction) and action.transport_id is not None:
            t = s.unit(action.transport_id)
            unit.embarked_in = t.id
            unit.reset_turn_flags()
            flow.deployed.append(unit.id)
            flow.queues[flow.active].remove(unit.id)
            self._say(s, f"Déploiement : {unit.name} commence la partie à bord de {t.name}", "deploy", unit=unit.id, embarked_in=t.id)
            self._notify(s, flow.active, f"{unit.name} embarque dans {t.name}")
            flow.active = other_side(flow.active)
            flow.step = "deploy_next"
            return
        if isinstance(action, DeployModelsAction):
            apply_model_positions(unit, _positions(action))
        else:
            c = unit.centroid
            apply_translation(unit, action.x - c[0], action.y - c[1])
        unit.reset_turn_flags()
        flow.deployed.append(unit.id)
        flow.queues[flow.active].remove(unit.id)
        c = unit.centroid
        self._say(s, f"Déploiement : {unit.name} en ({c[0]:.1f}, {c[1]:.1f})", "deploy", unit=unit.id, x=round(c[0], 2), y=round(c[1], 2))
        self._notify(s, flow.active, f"{unit.name} déployée en ({c[0]:.1f}, {c[1]:.1f})")
        flow.active = other_side(flow.active)
        flow.step = "deploy_next"

    def _auto_roll_off(self, s: GameState, flow: Flow) -> None:
        s.first_player = flow.forced_first_player or s.rng.choice(SIDES)
        self._say(s, f"Jet d'initiative : {s.first_player} commence.", "roll_off", first=s.first_player)
        flow.step = "scouts_start" if flow.deploy else "battle_start"

    # ---------------------------------------------------------------- Scouts (avant le round 1)

    def _auto_scouts_start(self, s: GameState, flow: Flow) -> None:
        flow.queues = {side: [u.id for u in s.units_of(side) if scout_distance(s, u) is not None] for side in SIDES}
        flow.active = s.first_player
        if any(flow.queues.values()):
            s.phase = "scouts"
        flow.step = "scouts_next"

    def _auto_scouts_next(self, s: GameState, flow: Flow) -> None:
        q = flow.queues
        q[flow.active] = [uid for uid in q[flow.active] if scout_distance(s, s.unit(uid)) is not None]
        if not q[flow.active]:
            if flow.active == s.first_player and q[other_side(flow.active)]:
                flow.active = other_side(flow.active)
                return
            flow.step = "battle_start"
            return
        if len(q[flow.active]) == 1:  # une seule unité : directement son mouvement (qui propose « pas de scout »)
            uid = q[flow.active].pop()
            flow.unit_id = uid
            flow.max_distance = scout_distance(s, s.unit(uid))
            flow.step = "scout"
            return
        flow.step = "scout_select"

    def _d_scout_select(self, s: GameState, flow: Flow) -> Decision:
        options: List[Action] = [SelectUnitAction(uid) for uid in flow.queues[flow.active]]
        options.append(EndPhaseAction())
        return Decision("select_unit", flow.active, options, phase="scouts", note="mouvements de scout avant le round 1")

    def _a_scout_select(self, s: GameState, flow: Flow, action) -> None:
        if isinstance(action, EndPhaseAction):
            flow.queues[flow.active] = []
            flow.step = "scouts_next"
            return
        flow.queues[flow.active].remove(action.unit_id)
        flow.unit_id = action.unit_id
        flow.max_distance = scout_distance(s, s.unit(action.unit_id))
        flow.step = "scout"

    def _d_scout(self, s: GameState, flow: Flow) -> Decision:
        u = s.unit(flow.unit_id)
        x = flow.max_distance
        options: List[Action] = [MoveAction(u.id, FormationMove(MoveKind.STATIONARY, label="pas de mouvement de scout"))]
        options += [MoveAction(u.id, mv) for mv in scout_moves(s, u, x)]
        return Decision("scout", flow.active, options, unit_id=u.id, max_distance=x,
                        note=f"Scouts {x:g}\" : mouvement normal, fin à plus de {s.rules.scout_enemy_distance_in:g}\" de toute figurine ennemie")

    def _a_scout(self, s: GameState, flow: Flow, action) -> None:
        u = s.unit(flow.unit_id)
        if isinstance(action, ModelMoveAction):
            longest = apply_model_positions(u, _positions(action))
            msg = f"Scouts : {u.name} fait un mouvement de scout figurine par figurine (jusqu'à {longest:.1f}\")"
        elif action.move.kind == MoveKind.STATIONARY:
            msg = f"Scouts : {u.name} ne fait pas de mouvement de scout"
        else:
            apply_translation(u, action.move.dx, action.move.dy)
            msg = f"Scouts : {u.name} — {action.move.label}"
        u.reset_turn_flags()
        self._say(s, msg, "scout", unit=u.id)
        self._notify(s, flow.active, msg)
        flow.step = "scouts_next"  # le même joueur termine ses scouts avant l'adversaire

    def _auto_battle_start(self, s: GameState, flow: Flow) -> None:
        s.battle_round = 1
        flow.step = "round_start"

    # ---------------------------------------------------------------- rounds et tours

    def _auto_round_start(self, s: GameState, flow: Flow) -> None:
        self._say(s, f"\n===== ROUND {s.battle_round} =====", "round")
        flow.turn_index = 0
        flow.step = "turn_start"

    def _auto_turn_start(self, s: GameState, flow: Flow) -> None:
        side = s.first_player if flow.turn_index == 0 else other_side(s.first_player)
        flow.active = side
        s.side_to_move = side
        self._say(s, f"\n--- Tour de {side} (round {s.battle_round}) ---", "turn", side=side)
        # les drapeaux valent « ce tour » : rev 2, remis à zéro pour les deux camps (une unité qui a
        # chargé à son tour n'est plus « chargeante » au tour adverse) ; rev 1 : camp actif seulement
        for u in (s.units_of(side, include_embarked=True) if s.rev < 2 else list(s.units.values())):
            u.reset_turn_flags()
            if u.side == side:
                u.engaged_at_turn_start = {e.id for e in s.enemies_in_engagement_range(u)} if u.embarked_in is None else set()
        s.update_sticky_control()
        s.controlled_at_turn_start = s.controlled_objectives(side)
        s.destroyed_this_turn = 0
        s.turn_counter += 1
        # phase de commandement (08.02) : les deux joueurs gagnent 1 CP
        s.phase = "command"
        prune_effects(s)
        for sd in SIDES:
            s.cp[sd] = s.cp.get(sd, 0) + 1
        self._say(s, f"Phase de commandement : +1 CP chacun (attaquant {s.cp['attacker']} CP, défenseur {s.cp['defender']} CP)",
                  "cp", cp=dict(s.cp))
        # V11 (08.03) : test pour chaque unité battle-shocked ou à moitié de son effectif (ou moins) ;
        # l'état persiste tant que l'unité n'a pas réussi un test
        flow.bs_queue = [u.id for u in s.units_of(side) if u.battle_shocked or u.below_half_strength]
        self._boundary(s, flow, "start", "battle_shock_next")

    def _auto_battle_shock_next(self, s: GameState, flow: Flow) -> None:
        side = flow.active
        if not flow.bs_queue:
            flow.step = "oath" if any(u.has_ability("Oath of Moment") for u in s.units_of(side)) and s.enemies_of(side) else "command_end"
            return
        flow.unit_id = flow.bs_queue.pop(0)
        self._open_window(s, flow, "insane_bravery", side, "battle_shock_roll", flow.unit_id)

    def _auto_battle_shock_roll(self, s: GameState, flow: Flow) -> None:
        u = s.unit(flow.unit_id)
        flow.step = "battle_shock_next"
        if u.is_destroyed:
            return
        if any(e[1] == "insane_bravery" and e[2] == u.id and e[3] == s.turn_counter for e in s.strat_used):
            u.battle_shocked = False
            self._say(s, f"Battle-shock : {u.name} — Insane Bravery : test réussi d'office", "battle_shock", unit=u.id, roll=None, passed=True)
            return
        if self._has(s, u, "battleshock_pass"):
            u.battle_shocked = False
            self._say(s, f"Battle-shock : {u.name} réussit son test d'office (effet)", "battle_shock", unit=u.id, roll=None, passed=True)
            return
        was = u.battle_shocked
        roll = s.roll(2)
        if roll < u.leadership:
            u.battle_shocked = True
            self._say(s, f"Battle-shock : {u.name} rate son test ({roll} < {u.leadership}+) — battle-shocked (OC 0) jusqu'à un test réussi", "battle_shock", unit=u.id, roll=roll, passed=False)
        else:
            u.battle_shocked = False
            self._say(s, f"Battle-shock : {u.name} réussit son test ({roll})" + (" — n'est plus battle-shocked" if was else ""), "battle_shock", unit=u.id, roll=roll, passed=True)

    def _d_oath(self, s: GameState, flow: Flow) -> Decision:
        return Decision("oath", flow.active, [OathAction(e.id) for e in s.enemies_of(flow.active)], note="cible d'Oath of Moment")

    def _a_oath(self, s: GameState, flow: Flow, action: OathAction) -> None:
        s.oath_target = action.target_id
        self._say(s, f"Oath of Moment : {s.unit(action.target_id).name}", "oath", target=action.target_id)
        flow.step = "command_end"

    def _auto_command_end(self, s: GameState, flow: Flow) -> None:
        side = flow.active
        s.apply_objective_secured(side)
        s.update_sticky_control()
        controlled = s.controlled_objectives(side)
        gained = s.scoreboard.add_all(s.mission.score_command_phase(side, s.battle_round, controlled, s.layout))
        if gained:
            self._say(s, f"Score (phase de commandement) : +{gained} VP pour {side} — objectifs contrôlés : {', '.join(sorted(controlled)) or 'aucun'}",
                      "score", side=side, vp=gained, when="command", objectives=sorted(controlled))
        self._notify(s, side, f"Phase de commandement de {side} terminée" + (f" : +{gained} VP" if gained else ""))
        self._boundary(s, flow, "end", "move_start")

    # ---------------------------------------------------------------- mouvement

    def _auto_move_start(self, s: GameState, flow: Flow) -> None:
        s.phase = "movement"
        prune_effects(s)
        # les unités à bord au début de la phase peuvent débarquer (elles sont activées à part)
        flow.pending = [u.id for u in s.units_of(flow.active)] + [u.id for u in s.units_of(flow.active, include_embarked=True) if u.embarked_in is not None]
        flow.ineligible = {}
        # 20.03 : les réserves arrivent par un mouvement d'ingress, à partir du round 2
        for u in s.reserves_of(flow.active):
            if s.battle_round >= 2:
                flow.pending.append(u.id)
            else:
                flow.ineligible[u.id] = "en réserve : arrivée à partir du round 2"
        self._boundary(s, flow, "start", "move_next")

    def _auto_move_next(self, s: GameState, flow: Flow) -> None:
        pending, ineligible = [], {uid: why for uid, why in flow.ineligible.items() if s.unit(uid).embarked_in is not None or s.unit(uid).in_reserve}
        for uid in flow.pending:
            u = s.unit(uid)
            if u.is_destroyed:
                continue
            if u.in_reserve:
                pending.append(uid)
                continue
            if u.embarked_in is not None:
                why = disembark_error(s, u)
                if why is not None:
                    ineligible[uid] = f"à bord de {u.embarked_in} : {why}"
                    continue
            pending.append(uid)
        flow.pending = pending
        flow.ineligible = ineligible
        flow.step = "move_select" if flow.pending else "move_end"

    def _d_move_select(self, s: GameState, flow: Flow) -> Decision:
        options: List[Action] = [SelectUnitAction(uid) for uid in flow.pending]
        options.append(EndPhaseAction())
        return Decision("select_unit", flow.active, options, phase="movement", ineligible=dict(flow.ineligible))

    def _a_move_select(self, s: GameState, flow: Flow, action) -> None:
        if isinstance(action, EndPhaseAction):
            for uid in flow.pending:
                u = s.unit(uid)
                if u.in_reserve:
                    self._say(s, f"Mouvement : {u.name} reste en réserve", "ingress", unit=uid, stayed=True)
                elif u.embarked_in is None:
                    self._say(s, f"Mouvement : {u.name} reste immobile", "move", unit=uid, move="stationary")
            flow.pending = []
            flow.step = "move_end"
            return
        flow.pending.remove(action.unit_id)
        flow.unit_id = action.unit_id
        u = s.unit(action.unit_id)
        if u.in_reserve:
            flow.ingress_side, flow.after_ingress = None, "move_next"
            flow.step = "ingress"
            return
        flow.step = "disembark" if u.embarked_in is not None else "move"

    def _auto_move_end(self, s: GameState, flow: Flow) -> None:
        """Fin de la phase de mouvement : fenêtres Fire Overwatch (15.08) puis Rapid Ingress (15.07) pour
        le joueur inactif."""
        self._open_window(s, flow, "fire_overwatch", other_side(flow.active), "move_end_ingress")

    def _auto_move_end_ingress(self, s: GameState, flow: Flow) -> None:
        self._open_window(s, flow, "rapid_ingress", other_side(flow.active), "move_end_bnd")

    def _auto_move_end_bnd(self, s: GameState, flow: Flow) -> None:
        self._boundary(s, flow, "end", "shoot_start")

    # ---------------------------------------------------------------- réserves (20)

    def _d_ingress(self, s: GameState, flow: Flow) -> Decision:
        u = s.unit(flow.unit_id)
        side = flow.ingress_side or flow.active
        options: List[Action] = [ReserveAction(u.id)] + [DeployAction(u.id, x, y) for x, y in ingress_candidates(s, u)]
        if has_deep_strike(u):
            note = "Deep Strike : n'importe où à plus de 8\" de toute unité ennemie (zone adverse comprise)"
        else:
            note = "entièrement à 6\" d'un bord de table, à plus de 8\" de toute unité ennemie" + (
                ", hors de la zone de déploiement adverse (avant le round 3)" if s.battle_round < 3 else "")
        return Decision("ingress", side, options, unit_id=u.id, note=note, phase="deep_strike" if has_deep_strike(u) else "edge")

    def _a_ingress(self, s: GameState, flow: Flow, action) -> None:
        u = s.unit(flow.unit_id)
        side = flow.ingress_side or flow.active
        nxt = flow.after_ingress or "move_next"
        flow.ingress_side, flow.after_ingress = None, ""
        flow.step = nxt
        if isinstance(action, ReserveAction):
            self._say(s, f"Mouvement : {u.name} reste en réserve", "ingress", unit=u.id, stayed=True)
            return
        pos = _positions(action) if isinstance(action, DeployModelsAction) else formation_at(u, action.x, action.y)
        self.set_up_unit(s, u, pos)
        u.arrived = True
        c = u.centroid
        how = "Deep Strike" if has_deep_strike(u) else "ingress"
        self._say(s, f"Mouvement : {u.name} arrive des réserves ({how}) en ({c[0]:.1f}, {c[1]:.1f})", "ingress", unit=u.id, x=round(c[0], 2), y=round(c[1], 2))
        self._notify(s, side, f"{u.name} arrive des réserves")

    def set_up_unit(self, s: GameState, unit: Unit, positions) -> None:
        """Pose l'unité sur la table (ingress, figurines revenues…) : elle a été « posée ce tour »."""
        for m in unit.alive_models:
            p = positions[m.id]
            m.move_to(p[0], p[1], p[2] if len(p) > 2 and not m.is_round else None)
        unit.in_reserve = False
        unit.disembarked = True  # « set up this turn » : pas de Heavy, pas d'embarquement
        unit.remained_stationary = False

    def to_reserves(self, s: GameState, unit: Unit, source: str = "", quiet: bool = False) -> None:
        """Unité remise en réserve stratégique pendant la bataille (20.02, repositioned unit)."""
        unit.in_reserve = True
        unit.repositioned = True
        if not quiet:
            self._say(s, f"Réserves : {unit.name} retourne en réserve stratégique" + (f" ({source})" if source else ""), "reserve", unit=unit.id, source=source)

    def _end_of_round_reserves(self, s: GameState) -> None:
        """20.04 : à la fin du round 3, les réserves jamais arrivées sont détruites (sauf 20.02)."""
        for u in list(s.units.values()):
            if not u.in_reserve or u.is_destroyed or u.arrived or u.repositioned:
                continue
            victims = [u] + [p for p in s.passengers(u.id)]
            for v in victims:
                for m in v.models:
                    m.alive, m.wounds = False, 0
            self._say(s, f"Réserves : {u.name} n'est jamais arrivée — détruite à la fin du round 3" + (
                f" (avec {', '.join(p.name for p in victims[1:])})" if len(victims) > 1 else ""), "reserve_destroyed", unit=u.id)

    # ---------------------------------------------------------------- transports

    _DISEMBARK_NOTES = {
        "rapid": "débarquement rapide (le transport a bougé) : entièrement à {d:g}\" du transport, hors portée d'engagement ; l'unité ne bougera plus et ne chargera pas ce tour (elle peut tirer)",
        "tactical": "débarquement tactique : entièrement à {d:g}\" du transport, hors portée d'engagement ; l'unité fait ensuite un mouvement normal ou une Advance",
        "combat": "débarquement de combat (pas de place à 3\") : entièrement à {d:g}\" du transport, un jet de danger par figurine, au contact possible des ennemis engagés avec le transport ; l'unité sera battle-shocked et ne chargera pas ce tour",
    }

    def _d_disembark(self, s: GameState, flow: Flow) -> Decision:
        u = s.unit(flow.unit_id)
        t = s.unit(u.embarked_in)
        mode, dist, allowed = disembark_mode(s, u)
        options: List[Action] = [DisembarkAction(u.id)]
        options += [DisembarkModelsAction(u.id, tuple((mid, x, y) for mid, (x, y) in pl.items())) for pl in ring_placements(s, u, t, dist, allowed_engaged=allowed)]
        options += [DisembarkAction(u.id, x, y) for x, y in disembark_candidates(s, u, t, dist=dist, step=0.25, limit=24, allowed_engaged=allowed)]
        return Decision("disembark", flow.active, options, unit_id=u.id, target_id=t.id, max_distance=dist,
                        note=f"{t.name} — " + self._DISEMBARK_NOTES[mode].format(d=dist), phase=mode)

    def _a_disembark(self, s: GameState, flow: Flow, action) -> None:
        u = s.unit(flow.unit_id)
        t = s.unit(u.embarked_in)
        if isinstance(action, DisembarkAction) and action.x is None:
            self._say(s, f"Mouvement : {u.name} reste à bord de {t.name}", "disembark", unit=u.id, transport=t.id, stayed=True)
            flow.step = "move_next"
            return
        mode, dist, _ = disembark_mode(s, u)
        detail = ""
        if mode == "combat":  # jets de danger avant de poser l'unité
            rolls, mw, lost = self.hazard_rolls(s, u, len(u.alive_models))
            detail = f" — jets de danger {rolls} : {mw} BM ({lost} fig.)"
        if isinstance(action, DisembarkModelsAction):
            pos = _positions(action)
            for m in u.alive_models:
                p = pos[m.id]
                m.move_to(p[0], p[1], p[2] if len(p) > 2 and not m.is_round else None)
        else:
            place_formation(u, action.x, action.y)
        u.embarked_in = None
        u.disembarked = True
        c = u.centroid if u.alive_models else t.models[0].position
        label = {"rapid": "débarquement rapide", "tactical": "débarquement tactique", "combat": "débarquement de combat"}[mode]
        if u.is_destroyed:
            self._say(s, f"Mouvement : {u.name} — {label} de {t.name}{detail} — UNITÉ DÉTRUITE", "disembark", unit=u.id, transport=t.id, mode=mode)
            self.on_unit_destroyed(s, u)
            flow.step = "move_next"
            return
        if mode == "tactical":
            self._say(s, f"Mouvement : {u.name} — {label} de {t.name} en ({c[0]:.1f}, {c[1]:.1f}), puis mouvement normal ou Advance", "disembark", unit=u.id, transport=t.id, mode=mode)
            self._notify(s, flow.active, f"{u.name} débarque de {t.name}")
            flow.step = "move"
            return
        u.remained_stationary = False
        u.moved_in = max(u.moved_in, t.moved_in)
        u.no_charge = True
        if mode == "combat":
            u.battle_shocked = True
        after = "ne bouge plus et ne chargera pas ce tour" + (" ; battle-shocked" if mode == "combat" else "")
        self._say(s, f"Mouvement : {u.name} — {label} de {t.name} en ({c[0]:.1f}, {c[1]:.1f}){detail} — {after}", "disembark", unit=u.id, transport=t.id, mode=mode)
        self._notify(s, flow.active, f"{u.name} débarque de {t.name}")
        flow.step = "move_next"

    def _auto_embark_check(self, s: GameState, flow: Flow) -> None:
        u = s.unit(flow.unit_id)
        if u.is_destroyed or u.disembarked or not embark_options(s, u):
            flow.step = "move_next"
            return
        flow.step = "embark"

    def _d_embark(self, s: GameState, flow: Flow) -> Decision:
        u = s.unit(flow.unit_id)
        options: List[Action] = [EmbarkAction(u.id, None)] + [EmbarkAction(u.id, t.id) for t in embark_options(s, u)]
        return Decision("embark", flow.active, options, unit_id=u.id, note=f"toutes les figurines sont à {s.rules.embark_range_in:g}\" d'un transport ami")

    def _a_embark(self, s: GameState, flow: Flow, action: EmbarkAction) -> None:
        u = s.unit(flow.unit_id)
        if action.transport_id is not None:
            t = s.unit(action.transport_id)
            u.embarked_in = t.id
            self._say(s, f"Mouvement : {u.name} embarque dans {t.name}", "embark", unit=u.id, transport=t.id)
            self._notify(s, flow.active, f"{u.name} embarque dans {t.name}")
        flow.step = "move_next"

    def hazard_rolls(self, s: GameState, unit: Unit, n: int) -> Tuple[List[int], int, int]:
        """``n`` jets de danger (06.03) pour l'unité : 1-2 = raté, 1 BM par échec (3 si l'unité est
        MONSTER / VEHICLE). Retourne (jets, blessures mortelles, figurines perdues)."""
        rules = s.rules
        rolls = [s.roll(1) for _ in range(n)]
        fails = sum(1 for r in rolls if r in rules.hazardous_fail_on)
        per = rules.hazardous_mortal_wounds_vehicle_monster if is_vehicle_or_monster(unit) else rules.hazardous_mortal_wounds_infantry
        mw = fails * per
        lost = self.mortal_wounds(s, unit, mw) if mw else 0
        return rolls, mw, lost

    def mortal_wounds(self, s: GameState, unit: Unit, n: int) -> int:
        """Blessures mortelles : Feel No Pain d'abord (révision 3), puis allocation ; figurines perdues."""
        if n <= 0:
            return 0
        if s.rev >= 3:
            return apply_mortal_wounds(s, unit, n)[0]
        return unit.allocate_mortal_wounds(n)

    def _emergency_disembark(self, s: GameState, transport: Unit) -> None:
        """Transport détruit (V11, 18.05) : pour chaque unité à bord, un jet de danger par figurine,
        puis chaque figurine est posée entièrement à 6" de l'épave et aussi près que possible d'elle
        (sinon elle est détruite) ; l'unité est battle-shocked et ne peut pas charger ce tour."""
        for p in s.passengers(transport.id):
            rolls, mw, lost = self.hazard_rolls(s, p, len(p.alive_models))
            detail = f"jets de danger {rolls} : {mw} BM ({lost} fig.)"
            placed = emergency_positions(s, p, transport) if not p.is_destroyed else {}
            unplaced = [m for m in p.alive_models if m.id not in placed]
            for m in p.alive_models:
                if m.id in placed:
                    m.move_to(*placed[m.id])
            for m in unplaced:
                m.alive, m.wounds = False, 0
            p.embarked_in = None
            p.disembarked = True
            p.battle_shocked = True
            p.no_charge = True
            where = f"en ({p.centroid[0]:.1f}, {p.centroid[1]:.1f})" if p.alive_models else ""
            self._say(s, f"Transport détruit : {p.name} débarque d'urgence {where} — {detail}"
                      + (f" ; {len(unplaced)} fig. sans place, détruite(s)" if unplaced else "")
                      + (" — UNITÉ DÉTRUITE" if p.is_destroyed else " — battle-shocked, pas de charge ce tour"),
                      "emergency_disembark", unit=p.id, transport=transport.id, rolls=rolls, mortal_wounds=mw, lost=lost + len(unplaced))
            if p.is_destroyed:
                if p.side != s.side_to_move:
                    s.destroyed_this_turn += 1
                self.on_unit_destroyed(s, p)

    def _d_move(self, s: GameState, flow: Flow) -> Decision:
        u = s.unit(flow.unit_id)
        return Decision("move", flow.active, [MoveAction(u.id, mv) for mv in candidate_moves(s, u)], unit_id=u.id, max_distance=self.move_of(s, u))

    def _a_move(self, s: GameState, flow: Flow, action) -> None:
        u = s.unit(flow.unit_id)
        if isinstance(action, DeclareAdvanceAction):
            roll = self.advance_roll(s, u)
            flow.roll = roll
            flow.max_distance = self.move_of(s, u) + roll + self._mod(s, u, "advance_mod")
            u.advanced = True
            self._say(s, f"Mouvement : {u.name} déclare une Advance — D6 = {roll}, jusqu'à {flow.max_distance:g}\" par figurine", "advance", unit=u.id, roll=roll)
            self._open_window(s, flow, "reroll_advance", flow.active, "advance_move", u.id)
            return
        msg = self.apply_move_action(s, u, action, self.move_of(s, u))
        self._notify(s, flow.active, msg)
        flow.step = "embark_check" if not (isinstance(action, MoveAction) and action.move.kind == MoveKind.STATIONARY) else "move_next"

    def _d_advance_move(self, s: GameState, flow: Flow) -> Decision:
        u = s.unit(flow.unit_id)
        max_d = flow.max_distance
        options: List[Action] = [MoveAction(u.id, FormationMove(MoveKind.STATIONARY, label="reste immobile"))]
        for mv in candidate_moves(s, u, distances=(1.0,)):
            if mv.kind == MoveKind.NORMAL:
                stretched = shrink_to_legal(s, u, FormationMove(MoveKind.NORMAL, mv.dx, mv.dy, mv.label).scaled(max_d / max(mv.distance, 1e-9)), max_d)
                if stretched is not None and stretched.distance > 1e-6:
                    options.append(MoveAction(u.id, stretched))
        return Decision("advance_move", flow.active, options, unit_id=u.id, max_distance=max_d)

    def _a_advance_move(self, s: GameState, flow: Flow, action) -> None:
        u = s.unit(flow.unit_id)
        msg = self.apply_move_action(s, u, action, flow.max_distance)
        self._notify(s, flow.active, msg)
        flow.step = "embark_check"

    # ---------------------------------------------------------------- tir

    def _auto_shoot_start(self, s: GameState, flow: Flow) -> None:
        s.update_sticky_control()  # fin de la phase de mouvement (14.02 : contrôle à la fin de chaque phase)
        s.phase = "shooting"
        prune_effects(s)
        s.charge_targets_this_phase = set()
        flow.done = []
        self._open_window(s, flow, "smokescreen", other_side(flow.active), "shoot_start_bnd")

    def _auto_shoot_start_bnd(self, s: GameState, flow: Flow) -> None:
        self._boundary(s, flow, "start", "shoot_next")

    def _auto_shoot_end(self, s: GameState, flow: Flow) -> None:
        self._boundary(s, flow, "end", "charge_start")

    def _auto_shoot_next(self, s: GameState, flow: Flow) -> None:
        eligible, ineligible = [], {}
        for u in s.units_of(flow.active):
            if u.id in flow.done:
                continue
            reason = shooting_ineligibility(s, u)
            if reason is None:
                eligible.append(u.id)
            else:
                ineligible[u.id] = reason
        if not eligible:
            if not flow.done and ineligible:
                self._say(s, "Tir : aucune unité ne peut tirer — " + " ; ".join(f"{uid} : {why}" for uid, why in ineligible.items()), "shoot_skip")
            flow.step = "shoot_end"
            return
        flow.eligible, flow.ineligible = eligible, ineligible
        flow.step = "shoot_select"

    def _d_shoot_select(self, s: GameState, flow: Flow) -> Decision:
        options: List[Action] = [SelectUnitAction(uid) for uid in flow.eligible]
        options += self._explosives_options(s, flow)
        options.append(EndPhaseAction())
        return Decision("select_unit", flow.active, options, phase="shooting", ineligible=dict(flow.ineligible))

    def _explosives_options(self, s: GameState, flow: Flow) -> List[Action]:
        """Explosives (15.05), proposé avant que l'unité ne tire : unité GRENADES / EXPLOSIVES désengagée,
        éligible au tir, sans Advance ce tour."""
        if not s.stratagems or unavailable(s, flow.active, "explosives") is not None:
            return []
        out: List[Action] = []
        for uid in flow.eligible:
            u = s.unit(uid)
            if not (u.has_keyword("Grenades") or u.has_keyword("Explosives")) or u.advanced or s.is_engaged(u):
                continue
            if unavailable(s, flow.active, "explosives", u) is not None:
                continue
            out += [StratagemAction("explosives", u.id, target_id=t.id) for t in explosives_targets(s, u)]
        return out

    def _a_shoot_select(self, s: GameState, flow: Flow, action) -> None:
        if isinstance(action, EndPhaseAction):
            flow.step = "shoot_end"
            return
        if isinstance(action, StratagemAction):
            self._use_explosives(s, flow, action)
            flow.step = "shoot_next"
            return
        flow.done.append(action.unit_id)
        flow.unit_id = action.unit_id
        self._open_window(s, flow, "detachment_selected", flow.active, "shoot", action.unit_id)

    def _d_shoot(self, s: GameState, flow: Flow) -> Decision:
        u = s.unit(flow.unit_id)
        on_table = not (u.is_destroyed or u.in_reserve or u.embarked_in is not None)
        options = [ShootAction(u.id, None)] + ([ShootAction(u.id, t.id) for t in shooting_targets(s, u)] if on_table else [])
        return Decision("shoot", flow.active, options, unit_id=u.id)

    def _a_shoot(self, s: GameState, flow: Flow, action: ShootAction) -> None:
        u = s.unit(flow.unit_id)
        if action.target_id is None:
            self._say(s, f"Tir : {u.name} ne tire pas", "shoot", unit=u.id, target=None)
            flow.step = "shoot_next"
            return
        # l'unité visée peut réagir (stratagèmes « just after an enemy unit has selected its targets »)
        flow.target_id = action.target_id
        self._open_window(s, flow, "detachment_targets", s.unit(action.target_id).side, "shoot_resolve", u.id)

    def _auto_shoot_resolve(self, s: GameState, flow: Flow) -> None:
        u = s.unit(flow.unit_id)
        target = s.unit(flow.target_id)
        flow.step = "shoot_next"
        if u.is_destroyed or target.is_destroyed or u.in_reserve or target.in_reserve:
            self._say(s, f"Tir : {u.name} ne tire pas (cible ou tireur retiré)", "shoot", unit=u.id, target=None)
            return
        report = resolve_shooting(s, u, target)
        s.charge_targets_this_phase.add(target.id)
        self._say(s, f"Tir : {report}", "shoot", unit=u.id, target=target.id, damage=report.damage, slain=report.models_slain,
                  destroyed=report.target_destroyed, hazardous=report.hazardous_mortal_wounds)
        for d in report.details:
            self._say(s, f"      {d}", "detail")
        if report.target_destroyed:
            s.destroyed_this_turn += 1
            self.on_unit_destroyed(s, target)
        if u.is_destroyed:  # Hazardous
            self.on_unit_destroyed(s, u)
        self._notify(s, flow.active, f"Tir : {report}")
        flow.step = "shoot_next"

    # ---------------------------------------------------------------- charge

    def charge_targets(self, s: GameState, unit: Unit) -> List[Unit]:
        if unit.is_destroyed or unit.no_charge or unit.embarked_in is not None or s.is_engaged(unit):
            return []
        thrill = unit.has_ability("Thrill Seekers")
        if (unit.advanced and not (thrill or self._has(s, unit, "advance_and_charge"))) or (unit.fell_back and not (thrill or self._has(s, unit, "fall_back_and_charge"))):
            return []
        out = []
        for e in s.enemies_of(unit.side):
            if unit.min_gap_to(e) > s.rules.charge_declare_range_in:
                continue
            if (unit.advanced or unit.fell_back) and thrill and (e.id in unit.engaged_at_turn_start or e.id in s.charge_targets_this_phase):
                continue
            out.append(e)
        return out

    def charge_ineligibility(self, s: GameState, unit: Unit) -> Optional[str]:
        """Pourquoi l'unité ne peut pas déclarer de charge (None si elle peut)."""
        if unit.is_destroyed:
            return "détruite"
        if unit.charged:
            return "a déjà chargé"
        if unit.no_charge:
            return "a débarqué ce tour d'un transport qui avait bougé ou qui a été détruit : pas de charge"
        if s.is_engaged(unit):
            return "déjà à portée d'engagement d'un ennemi"
        thrill = unit.has_ability("Thrill Seekers")
        if unit.advanced and not (thrill or self._has(s, unit, "advance_and_charge")):
            return "a fait une Advance ce tour"
        if unit.fell_back and not (thrill or self._has(s, unit, "fall_back_and_charge")):
            return "s'est repliée ce tour"
        enemies = s.enemies_of(unit.side)
        if not enemies:
            return "aucun ennemi"
        nearest = min(unit.min_gap_to(e) for e in enemies)
        if nearest > s.rules.charge_declare_range_in:
            return f"aucun ennemi à 12\" (le plus proche est à {nearest:.1f}\")"
        if not self.charge_targets(s, unit):
            return "Thrill Seekers : les cibles à portée sont interdites (engagée au début du tour ou déjà visée)"
        return None

    def _auto_charge_start(self, s: GameState, flow: Flow) -> None:
        s.update_sticky_control()  # fin de la phase de tir
        s.phase = "charge"
        prune_effects(s)
        s.charge_targets_this_phase = set()
        flow.done = []
        self._boundary(s, flow, "start", "charge_next")

    def _auto_charge_next(self, s: GameState, flow: Flow) -> None:
        eligible, ineligible = [], {}
        for u in s.units_of(flow.active):
            if u.id in flow.done:
                continue
            reason = self.charge_ineligibility(s, u)
            if reason is None:
                eligible.append(u.id)
            else:
                ineligible[u.id] = reason
        if not eligible:
            if not flow.done and ineligible and not any(u.charged for u in s.units_of(flow.active)):
                self._say(s, "Charge : aucune unité ne peut charger — " + " ; ".join(f"{uid} : {why}" for uid, why in ineligible.items()), "charge_skip")
            flow.step = "charge_end"
            return
        flow.eligible, flow.ineligible = eligible, ineligible
        flow.step = "charge_select"

    def _d_charge_select(self, s: GameState, flow: Flow) -> Decision:
        options: List[Action] = [SelectUnitAction(uid) for uid in flow.eligible]
        options.append(EndPhaseAction())
        return Decision("select_unit", flow.active, options, phase="charge", ineligible=dict(flow.ineligible))

    def _a_charge_select(self, s: GameState, flow: Flow, action) -> None:
        if isinstance(action, EndPhaseAction):
            flow.step = "charge_end"
            return
        flow.unit_id = action.unit_id
        flow.candidates = [t.id for t in self.charge_targets(s, s.unit(action.unit_id))]
        flow.step = "charge_declare"

    def _d_charge_declare(self, s: GameState, flow: Flow) -> Decision:
        u = s.unit(flow.unit_id)
        gaps = ", ".join(f"{tid} à {charge_gap(u, s.unit(tid)):.1f}\"" for tid in flow.candidates)
        return Decision("charge_declare", flow.active, [DeclareChargeAction(u.id), ChargeAction(u.id, None)], unit_id=u.id, note=f"cibles à 12\" : {gaps}")

    def _a_charge_declare(self, s: GameState, flow: Flow, action) -> None:
        u = s.unit(flow.unit_id)
        if isinstance(action, ChargeAction):
            flow.done.append(u.id)
            flow.step = "charge_next"
            return
        roll = s.roll(2) + self._mod(s, u, "charge_mod")
        flow.roll = roll
        flow.charger = None
        if s.stratagems:  # le joueur voit ce que son jet atteint avant de décider de relancer
            flow.reachable = [t.id for t in charge_reachable_targets(s, u, [s.unit(t) for t in flow.candidates], roll)]
        self._open_window(s, flow, "reroll_charge", flow.active, "charge_roll_result", u.id)

    def _auto_charge_roll_result(self, s: GameState, flow: Flow) -> None:
        """Jet de charge fait (éventuellement relancé) : cibles atteignables, ou charge ratée."""
        u = s.unit(flow.unit_id)
        side = flow.charger if flow.hi else flow.active
        roll = flow.roll
        candidates = [s.unit(tid) for tid in flow.candidates if not s.unit(tid).is_destroyed]
        reachable = charge_reachable_targets(s, u, candidates, roll) if candidates else []
        if roll == 2 or not reachable:
            if roll == 2:
                self._say(s, f"Charge : {u.name} lance 2D6 = 2 (double 1) : échec automatique", "charge", unit=u.id, roll=roll, success=False)
            else:
                nearest = min((charge_gap(u, t) for t in candidates), default=0.0)
                self._say(s, f"Charge : {u.name} lance 2D6 = {roll} — aucune cible atteignable socle à socle (la plus proche est à {nearest:.1f}\") : charge ratée",
                          "charge", unit=u.id, roll=roll, success=False)
            flow.done.append(u.id)
            self._notify(s, side, s.log[-1] if s.log else f"{u.name} rate sa charge")
            flow.step = self._after_charge(flow)
            return
        self._say(s, f"Charge : {u.name} lance 2D6 = {roll} — atteignable(s) : " + ", ".join(f"{t.id} ({charge_gap(u, t):.1f}\")" for t in reachable),
                  "charge_roll", unit=u.id, roll=roll, reachable=[t.id for t in reachable])
        flow.reachable = [t.id for t in reachable]
        if len(reachable) == 1:
            flow.target_id = reachable[0].id
            flow.targets = [reachable[0].id]
            flow.step = "charge_move"
        else:
            flow.step = "charge_target"

    def _auto_charge_end(self, s: GameState, flow: Flow) -> None:
        """Fin de la phase de charge : fenêtre Heroic Intervention pour le joueur inactif (15.11)."""
        self._open_window(s, flow, "heroic_intervention", other_side(flow.active), "charge_end_bnd")

    def _auto_charge_end_bnd(self, s: GameState, flow: Flow) -> None:
        self._boundary(s, flow, "end", "fight_start")

    def charge_target_options(self, s: GameState, unit: Unit, reachable: List[str]) -> List[ChargeAction]:
        """Choix des cibles (V11 11.04 : une ou plusieurs unités à portée du jet) : chaque cible seule,
        puis les paires (et le trio) de cibles assez proches l'une de l'autre pour être engagées
        ensemble (6" au plus entre elles)."""
        from itertools import combinations

        out = [ChargeAction(unit.id, tid) for tid in reachable]
        for k in (2, 3):
            if len(reachable) < k:
                break
            for combo in combinations(reachable, k):
                us = [s.unit(t) for t in combo]
                if all(a.min_gap_to(b) <= 6.0 for a, b in combinations(us, 2)):
                    out.append(ChargeAction(unit.id, combo[0], tuple(combo[1:])))
        return out

    def _after_charge(self, flow: Flow) -> str:
        """Étape suivant une charge résolue : la suivante, ou la fin de la phase après une Heroic Intervention."""
        if flow.hi:
            flow.hi, flow.charger = False, None
            return "charge_end_bnd"
        return "charge_next"

    def _d_charge_target(self, s: GameState, flow: Flow) -> Decision:
        u = s.unit(flow.unit_id)
        return Decision("charge_target", flow.charger if flow.hi else flow.active, self.charge_target_options(s, u, flow.reachable), unit_id=flow.unit_id, max_distance=float(flow.roll),
                        note="une ou plusieurs cibles : l'unité devra finir engagée (2\") avec chacune, une figurine au moins socle à socle")

    def _a_charge_target(self, s: GameState, flow: Flow, action: ChargeAction) -> None:
        flow.target_id = action.target_id
        flow.targets = list(action.targets)
        flow.step = "charge_move"

    def _d_charge_move(self, s: GameState, flow: Flow) -> Decision:
        names = " + ".join(flow.targets)
        return Decision("charge_move", flow.charger if flow.hi else flow.active, [AutoChargeMoveAction(flow.unit_id, flow.target_id)], unit_id=flow.unit_id, max_distance=float(flow.roll),
                        target_id=flow.target_id, targets=tuple(flow.targets),
                        note=f"jet {flow.roll} : au moins une figurine socle à socle avec {names}, l'unité engagée (2\") avec chaque cible ; "
                             f"celles qui peuvent arriver à 1\" doivent le faire, sinon à 2\", sinon plus près")

    def _a_charge_move(self, s: GameState, flow: Flow, action) -> None:
        u, roll = s.unit(flow.unit_id), flow.roll
        targets = [s.unit(t) for t in (flow.targets or [flow.target_id])]
        names = " + ".join(t.name for t in targets)
        reason = ""
        if isinstance(action, ModelMoveAction):
            apply_model_positions(u, _positions(action))
            ok = True
        else:
            ok, reason = auto_charge_move(s, u, targets, roll)
        if ok:
            u.charged = True
            u.fights_first = s.rules.charger_fights_first
            for t in targets:
                s.charge_targets_this_phase.add(t.id)
            self._say(s, f"Charge : {u.name} charge {names} avec un jet de {roll} : RÉUSSIE, au contact", "charge", unit=u.id, target=targets[0].id,
                      targets=[t.id for t in targets], roll=roll, success=True)
        else:
            flow.done.append(u.id)
            self._say(s, f"Charge : {u.name} charge {names} avec {roll} : placement impossible ({reason}) — charge ratée", "charge", unit=u.id, target=targets[0].id,
                      targets=[t.id for t in targets], roll=roll, success=False)
        side = flow.charger if flow.hi else flow.active
        self._notify(s, side, s.log[-1] if s.log else f"{u.name} : charge")
        was_hi = flow.hi
        flow.step = self._after_charge(flow)
        if ok and not was_hi and is_vehicle_or_monster(u):  # Crushing Impact (15.06) : « ta » phase de charge
            self._open_window(s, flow, "crushing_impact", side, flow.step, u.id)

    # ---------------------------------------------------------------- combat

    # Phase de combat V11 (12) : étape pile-in (joueur actif puis adversaire), étape combat (Fights First
    # en alternance en commençant par le joueur actif, puis les autres ; overrun fight pour une unité
    # éligible mais désengagée), puis étape consolidation (joueur actif puis adversaire).

    def fight_eligible(self, s: GameState, unit: Unit) -> bool:
        """12.04 : pas encore choisie pour combattre, et engagée (ou engagée au début de l'étape de
        combat) ou a chargé ce tour."""
        if unit.is_destroyed or unit.embarked_in is not None or unit.has_fought:
            return False
        return unit.charged or unit.id in s.flow.fight_start_engaged or s.is_engaged(unit)

    def fight_options(self, s: GameState, unit: Unit) -> List[Unit]:
        """Cibles possibles : les unités engagées (normal fight) ; sinon, pour une overrun fight, les
        unités ennemies à 5" que le pile-in supplémentaire peut engager. Vide = l'unité ne peut pas
        combattre maintenant (son joueur « passe », annexe des règles)."""
        engaged = fight_targets(s, unit)
        if engaged:
            return engaged
        return [e for e in s.enemies_of(unit.side) if unit.min_gap_to(e) <= s.rules.fight_pass_range_in + 1e-9]

    def fight_pool(self, s: GameState, fights_first: bool) -> Dict[str, List[Unit]]:
        pool: Dict[str, List[Unit]] = {side: [] for side in SIDES}
        for u in s.on_table_units():
            if self.has_fights_first(s, u) != fights_first or not self.fight_eligible(s, u):
                continue
            if u.id not in s.flow.fight_seen:
                s.flow.fight_seen.append(u.id)
            if self.fight_options(s, u):
                pool[u.side].append(u)
        return pool

    def _auto_fight_start(self, s: GameState, flow: Flow) -> None:
        s.update_sticky_control()  # fin de la phase de charge
        s.phase = "fight"
        prune_effects(s)
        for u in s.units.values():
            u.has_fought = False
        flow.fights_first = True
        flow.fight_side = flow.active
        flow.last_activator = None
        flow.fight_start_engaged = []
        flow.fight_seen = []
        flow.forced_fighter = None
        self._boundary(s, flow, "start", "pile_in_step")

    def _auto_pile_in_step(self, s: GameState, flow: Flow) -> None:
        """12.02 : pile-in de toutes les unités éligibles (engagées ou qui ont chargé), joueur actif d'abord."""
        for side in (flow.active, other_side(flow.active)):
            for u in sorted(s.units_of(side), key=lambda x: x.id):
                if not (u.charged or s.is_engaged(u)):
                    continue
                before = [(m.x, m.y) for m in u.alive_models]
                if pile_in(s, u) and any(abs(m.x - x) + abs(m.y - y) > 1e-6 for m, (x, y) in zip(u.alive_models, before)):
                    self._say(s, f"Combat : pile-in de {u.name}", "pile_in", unit=u.id)
        flow.fight_start_engaged = [u.id for u in s.on_table_units() if s.is_engaged(u)]
        flow.step = "fight_next"

    def _auto_fight_next(self, s: GameState, flow: Flow) -> None:
        while True:
            if not flow.fights_first and any(self.fight_pool(s, True).values()):
                flow.fights_first = True  # un Fights First est redevenu éligible : on y retourne
                flow.fight_side = flow.active
            pool = self.fight_pool(s, flow.fights_first)
            nxt = flow.fight_side
            if pool[nxt]:
                flow.step = "fight"
                return
            if pool[other_side(nxt)]:
                flow.fight_side = other_side(nxt)
                continue
            if flow.fights_first:
                flow.fights_first = False
                flow.fight_side = other_side(flow.last_activator) if flow.last_activator else flow.active
                continue
            flow.step = "consolidate_step"
            return

    def _d_fight(self, s: GameState, flow: Flow) -> Decision:
        pool = self.fight_pool(s, flow.fights_first)[flow.fight_side]
        if flow.forced_fighter is not None:  # Counteroffensive : cette unité combat la première
            forced = [u for u in pool if u.id == flow.forced_fighter]
            if forced:
                pool = forced
        options = [FightAction(u.id, t.id) for u in pool for t in self.fight_options(s, u)]
        return Decision("fight", flow.fight_side, options)

    def _a_fight(self, s: GameState, flow: Flow, action: FightAction) -> None:
        flow.unit_id, flow.target_id = action.unit_id, action.target_id
        if flow.forced_fighter == action.unit_id:
            flow.forced_fighter = None
        self._open_window(s, flow, "epic_challenge", flow.fight_side, "fight_selected", action.unit_id)

    def _auto_fight_selected(self, s: GameState, flow: Flow) -> None:
        self._open_window(s, flow, "detachment_selected", flow.fight_side, "fight_targets", flow.unit_id)

    def _auto_fight_targets(self, s: GameState, flow: Flow) -> None:
        self._open_window(s, flow, "detachment_targets", s.unit(flow.target_id).side, "fight_resolve", flow.unit_id)

    def _auto_fight_resolve(self, s: GameState, flow: Flow) -> None:
        unit = s.unit(flow.unit_id)
        if unit.is_destroyed or unit.in_reserve:
            unit.has_fought = True
            flow.last_activator = flow.fight_side
            flow.fight_side = other_side(flow.fight_side)
            flow.step = "fight_next"
            return
        self.activate_fight(s, unit, s.unit(flow.target_id))
        flow.last_activator = flow.fight_side
        flow.fight_side = other_side(flow.fight_side)
        flow.step = "fight_next"
        if unit.side == flow.active:  # Counteroffensive (15.12) : « phase de combat de ton adversaire »
            self._open_window(s, flow, "counteroffensive", other_side(flow.active), "fight_next")

    def activate_fight(self, s: GameState, unit: Unit, target: Unit) -> None:
        """L'unité combat : overrun fight (pile-in supplémentaire vers ``target``) si elle est
        désengagée ; puis chaque figurine frappe une unité qu'elle engage — ``target`` d'abord,
        les autres figurines frappent l'unité ennemie la plus proche qu'elles engagent (04.02)."""
        unit.has_fought = True
        if not s.is_engaged(unit):
            pile_in(s, unit, targets=[target])
            self._say(s, f"Combat : overrun fight de {unit.name} (pile-in supplémentaire vers {target.name})", "overrun", unit=unit.id, target=target.id)
        engaged = fight_targets(s, unit)
        if not engaged:
            self._say(s, f"Combat : {unit.name} n'a plus d'ennemi à portée", "fight", unit=unit.id, target=None)
            return
        if target not in engaged:
            target = engaged[0]
        er = s.rules.engagement_range_in
        # répartition : chaque figurine frappe la cible choisie si elle l'engage, sinon l'ennemi engagé le plus proche
        groups: Dict[str, List] = {}
        for m in unit.alive_models:
            mine = [e for e in engaged if any(within(m.disk, d, er) for d in e.disks())]
            if not mine:
                continue
            tgt = target if target in mine else min(mine, key=lambda e: min(disk_gap(m.disk, d) for d in e.disks()))
            groups.setdefault(tgt.id, []).append(m)
        order = [target.id] + [tid for tid in groups if tid != target.id]
        for tid in order:
            if tid not in groups:
                continue
            tgt = s.unit(tid)
            if tgt.is_destroyed:
                continue
            report = resolve_fight(s, unit, tgt, fighters=groups[tid])
            self._say(s, f"Combat : {report}", "fight", unit=unit.id, target=tgt.id, damage=report.damage, slain=report.models_slain, destroyed=report.target_destroyed)
            for d in report.details:
                self._say(s, f"      {d}", "detail")
            if report.target_destroyed:
                if tgt.side != s.side_to_move:
                    s.destroyed_this_turn += 1
                self.on_unit_destroyed(s, tgt)
            self._notify(s, unit.side, f"Combat : {report}")
            if unit.is_destroyed:
                break

    def _auto_consolidate_step(self, s: GameState, flow: Flow) -> None:
        """12.07 : consolidation de toutes les unités éligibles à combattre cette phase, joueur actif
        d'abord. Une consolidation « engaging » qui engage des unités ennemies qui n'ont pas encore
        combattu les fait combattre aussitôt."""
        for side in (flow.active, other_side(flow.active)):
            for uid in list(flow.fight_seen):
                u = s.unit(uid)
                if u.side != side or u.is_destroyed or u.embarked_in is not None:
                    continue
                mode, new = consolidate(s, u)
                if not mode:
                    continue
                label = {"ongoing": "reste au contact", "engaging": "engage un nouvel ennemi", "objective": "se rapproche d'un objectif"}[mode]
                self._say(s, f"Combat : consolidation de {u.name} — {label}", "consolidate", unit=u.id, mode=mode)
                if mode == "engaging":
                    for eid in new:
                        e = s.unit(eid)
                        if e.is_destroyed or e.has_fought:
                            continue
                        if eid not in flow.fight_seen:
                            flow.fight_seen.append(eid)
                        self.activate_fight(s, e, u)
        s.update_sticky_control()  # fin de la phase de combat
        self._boundary(s, flow, "end", "end_turn")

    # ---------------------------------------------------------------- fin de tour et de partie

    def _auto_end_turn(self, s: GameState, flow: Flow) -> None:
        side = flow.active
        s.phase = "end_turn"
        for u in s.units.values():
            u.fights_first = False  # le Fights First de la charge dure jusqu'à la fin du tour (11.04)
        self._restore_coherency(s)
        s.update_sticky_control()
        controlled = s.controlled_objectives(side)
        events = s.mission.score_end_of_turn(side, s.battle_round, controlled, s.controlled_at_turn_start, s.destroyed_this_turn, s.layout)
        gained = s.scoreboard.add_all(events)
        for e in events:
            self._say(s, f"Score (fin de tour) : {e.reason} → +{e.vp} VP", "score", side=side, vp=e.vp, when="end_turn")
        # 23.02 : à la fin du tour adverse, les AIRCRAFT sur la table retournent en réserve
        for u in s.units_of(other_side(side)):
            if u.has_keyword("Aircraft"):
                self.to_reserves(s, u, "AIRCRAFT")
        if flow.turn_index == 1 and s.battle_round == 3:
            self._end_of_round_reserves(s)
        self._say(s, f"Fin du tour de {side} : {s.summary()}", "turn_end", side=side)
        self._notify(s, side, f"Fin du tour de {side}" + (f" : +{gained} VP" if gained else ""))
        self._emit(s, {"kind": "turn_end", "side": side})
        flow.turn_index += 1
        if self.battle_over(s):
            self._say(s, "Un camp n'a plus d'unité : la bataille s'arrête.", "battle_over")
            flow.over_reason = "annihilation"
            flow.step = "battle_end"
        elif flow.turn_index < 2:
            flow.step = "turn_start"
        elif s.battle_round >= s.rules.battle_rounds:
            flow.step = "battle_end"
        else:
            s.battle_round += 1
            flow.step = "round_start"

    def _restore_coherency(self, s: GameState) -> None:
        """03.03 : en fin de tour, une unité qui n'est plus en cohérence perd des figurines (au choix de
        son joueur : on garde le plus grand groupe cohérent) ; ces pertes ne déclenchent rien."""
        rules = s.rules
        for u in s.on_table_units():
            if u.coherency_ok(rules):
                continue
            keep = _largest_coherent_group(u, rules)
            lost = [m for m in u.alive_models if m.id not in keep]
            for m in lost:
                m.alive, m.wounds = False, 0
            self._say(s, f"Cohérence : {u.name} n'est plus en cohérence en fin de tour — {len(lost)} figurine(s) retirée(s)", "coherency", unit=u.id, lost=len(lost))

    def battle_over(self, s: GameState) -> bool:
        return any(not s.units_of(side, include_embarked=True, include_reserves=True) for side in SIDES)

    def _auto_battle_end(self, s: GameState, flow: Flow) -> None:
        s.phase = "end_battle"
        s.update_sticky_control()
        for side in SIDES:
            events = s.mission.score_end_of_battle(side, s.controlled_objectives(side), s.layout)
            s.scoreboard.add_all(events)
            for e in events:
                self._say(s, f"Fin de bataille : {side} — {e.reason} → +{e.vp} VP", "score", side=side, vp=e.vp, when="end_battle")
        totals = s.scoreboard.totals()
        self._say(s, f"\nRÉSULTAT : {s.summary()}", "result", scores=totals, winner=s.scoreboard.leader())
        flow.step = "game_over"

    # ================================================================ stratagèmes (15)

    def _open_window(self, s: GameState, flow: Flow, window: str, side: str, resume: str, unit_id: Optional[str] = None) -> None:
        """Ouvre une fenêtre de stratagème si au moins une option est utilisable ; sinon on passe
        directement à ``resume`` (passe automatique : rien n'est demandé au joueur)."""
        flow.step = resume
        if not s.stratagems:
            return
        flow.window, flow.window_side, flow.window_unit, flow.resume = window, side, unit_id, resume
        if self._window_options(s, flow):
            flow.step = "stratagem"
        else:
            flow.window = ""

    def _window_options(self, s: GameState, flow: Flow) -> List[StratagemAction]:
        return getattr(self, "_opts_" + flow.window)(s, flow, flow.window_side)

    def _d_stratagem(self, s: GameState, flow: Flow) -> Decision:
        if flow.window.startswith("detachment"):
            opts = self._window_options(s, flow)
            from .free import stratagem_name

            names = ", ".join(sorted({stratagem_name(s, flow.window_side, o.stratagem_id) for o in opts}))
            note = {"detachment_targets": "l'ennemi vient de choisir ses cibles", "detachment_selected": "ton unité vient d'être choisie",
                    "detachment_start": "début de la phase", "detachment_end": "fin de la phase"}.get(flow.window, "") + \
                f" — stratagèmes possibles : {names}. Tu as {s.cp.get(flow.window_side, 0)} CP."
            return Decision("stratagem", flow.window_side, list(opts) + [StratagemAction(None)], unit_id=None, phase=s.phase, note=note, window=flow.window)
        key, when = WINDOWS[flow.window]
        st = CORE_STRATAGEMS[key]
        options: List[Action] = list(self._window_options(s, flow)) + [StratagemAction(None)]
        note = f"{st.name} ({st.ref}, {cost_of(key)} CP) — {when} : {st.summary}. Tu as {s.cp.get(flow.window_side, 0)} CP."
        if flow.window == "reroll_charge":
            u = s.unit(flow.window_unit)
            reach = ", ".join(f"{t} ({charge_gap(u, s.unit(t)):.1f}\")" for t in flow.reachable) or "aucune cible"
            far = ", ".join(f"{t} ({charge_gap(u, s.unit(t)):.1f}\")" for t in flow.candidates if t not in flow.reachable)
            note = f"Jet de charge {flow.roll}" + (" (double 1 : échec)" if flow.roll == 2 else "") + f" — atteignable : {reach}" \
                   + (f" ; hors d'atteinte : {far}" if far else "") + ". " + note
        elif flow.window == "reroll_advance":
            note = f"Jet d'Advance {flow.roll} (jusqu'à {flow.max_distance:g}\"). " + note
        return Decision("stratagem", flow.window_side, options, unit_id=None, phase=s.phase, note=note, window=flow.window)

    def _a_stratagem(self, s: GameState, flow: Flow, action) -> None:
        flow.window = ""
        flow.step = flow.resume  # l'effet peut rediriger (Heroic Intervention → charge)
        if isinstance(action, UseStratagemAction):  # stratagème de détachement proposé dans une fenêtre
            from .free import apply_free

            text = apply_free(self, s, flow.window_side, action)
            self._say(s, text, "stratagem", side=flow.window_side)
            self._notify(s, flow.window_side, text)
            return
        if action.stratagem is None:
            return
        getattr(self, "_use_" + action.stratagem)(s, flow, action)

    def _strat_say(self, s: GameState, side: str, key: str, text: str, cost: int, **data) -> None:
        st = CORE_STRATAGEMS[key]
        self._say(s, f"Stratagème {st.name} ({cost} CP, reste {s.cp[side]}) — {side} : {text}", "stratagem", side=side, stratagem=key, cost=cost, **data)

    # --- stratagèmes de détachement (révision 3) : fenêtres selon le moment traduit du WHEN

    def _detachment_options(self, s: GameState, side: str, moment: str, units: List[Unit], target_id: Optional[str] = None,
                            limit: int = 24) -> List[UseStratagemAction]:
        from ..rules_compiler import compile_stratagem
        from .free import stratagem_timing_error, target_ok
        from .stratagems import key_of

        if s.rev < 3 or not s.stratagems:
            return []
        out: List[UseStratagemAction] = []
        for st in s.stratagem_book.get(side, []):
            if not st.detachment_id:  # les stratagèmes de base ont leurs propres fenêtres
                continue
            comp = compile_stratagem(st)
            if comp.timing.moment != moment or s.phase not in comp.timing.phases or stratagem_timing_error(s, side, st) is not None:
                continue
            if comp.choices or (comp.needs_enemy and target_id is None):
                continue  # un choix à faire : par le panneau des stratagèmes
            for u in units:
                if u.is_destroyed or not target_ok(u, st):
                    continue
                if unavailable(s, side, key_of(st.name), u, cost=st.cp or 0) is not None:
                    continue
                out.append(UseStratagemAction(st.id, u.id, target_id=target_id))
                if len(out) >= limit:
                    return out
        return out

    def _opts_detachment_targets(self, s: GameState, flow: Flow, side: str) -> List[UseStratagemAction]:
        """L'ennemi vient de choisir ses cibles (tir ou combat) : l'unité visée peut réagir."""
        target = s.unit(flow.target_id)
        return self._detachment_options(s, side, "targets_selected", [target], target_id=flow.window_unit)

    def _opts_detachment_selected(self, s: GameState, flow: Flow, side: str) -> List[UseStratagemAction]:
        """Une unité du joueur vient d'être choisie pour tirer ou combattre."""
        return self._detachment_options(s, side, "selected", [s.unit(flow.window_unit)])

    def _opts_detachment_start(self, s: GameState, flow: Flow, side: str) -> List[UseStratagemAction]:
        return self._detachment_options(s, side, "start", s.units_of(side))

    def _opts_detachment_end(self, s: GameState, flow: Flow, side: str) -> List[UseStratagemAction]:
        return self._detachment_options(s, side, "end", s.units_of(side))

    def _boundary(self, s: GameState, flow: Flow, moment: str, resume: str) -> None:
        """Début / fin de phase : fenêtre des stratagèmes de détachement, joueur actif puis adversaire."""
        flow.bnd_moment, flow.bnd_resume = moment, resume
        flow.step = "boundary_a"

    def _auto_boundary_a(self, s: GameState, flow: Flow) -> None:
        self._open_window(s, flow, "detachment_" + flow.bnd_moment, flow.active, "boundary_b")

    def _auto_boundary_b(self, s: GameState, flow: Flow) -> None:
        self._open_window(s, flow, "detachment_" + flow.bnd_moment, other_side(flow.active), flow.bnd_resume)

    # --- Command Re-roll (15.02) : jets d'Advance et de charge

    def _opts_reroll_advance(self, s: GameState, flow: Flow, side: str) -> List[StratagemAction]:
        u = s.unit(flow.window_unit)
        if flow.roll >= 6 or unavailable(s, side, "command_reroll", u) is not None:
            return []
        return [StratagemAction("command_reroll", u.id, mode="advance")]

    def _opts_reroll_charge(self, s: GameState, flow: Flow, side: str) -> List[StratagemAction]:
        u = s.unit(flow.window_unit)
        if unavailable(s, side, "command_reroll", u) is not None:
            return []
        if flow.roll >= 12:
            return []  # rien de mieux possible (on peut vouloir une charge plus longue même si une cible est atteinte)
        if s.rev < 3 and flow.roll != 2 and len(flow.reachable) >= len(flow.candidates):
            return []  # parties enregistrées en révision 2 : fenêtre seulement si une cible manquait
        return [StratagemAction("command_reroll", u.id, mode="charge")]

    def _use_command_reroll(self, s: GameState, flow: Flow, action: StratagemAction) -> None:
        u = s.unit(action.unit_id)
        side = u.side
        cost = record_use(s, side, "command_reroll", u.id, detail=action.mode)
        old = flow.roll
        if action.mode == "advance":
            flow.roll = s.d6()
            flow.max_distance = self.move_of(s, u) + flow.roll + self._mod(s, u, "advance_mod")
            self._strat_say(s, side, "command_reroll", f"{u.name} relance son jet d'Advance : {old} → {flow.roll}, jusqu'à {flow.max_distance:g}\"", cost, unit=u.id, roll=flow.roll)
        else:
            flow.roll = s.roll(2)  # une charge se relance en entier (les deux dés)
            self._strat_say(s, side, "command_reroll", f"{u.name} relance son jet de charge : {old} → {flow.roll}", cost, unit=u.id, roll=flow.roll)

    # --- Insane Bravery (15.04)

    def _opts_insane_bravery(self, s: GameState, flow: Flow, side: str) -> List[StratagemAction]:
        u = s.unit(flow.window_unit)
        if u.is_destroyed or unavailable(s, side, "insane_bravery", u) is not None:
            return []  # une unité battle-shocked ne peut pas être ciblée (01.07)
        return [StratagemAction("insane_bravery", u.id)]

    def _use_insane_bravery(self, s: GameState, flow: Flow, action: StratagemAction) -> None:
        u = s.unit(action.unit_id)
        cost = record_use(s, u.side, "insane_bravery", u.id)
        self._strat_say(s, u.side, "insane_bravery", f"{u.name} réussira son test de battle-shock", cost, unit=u.id)

    # --- Epic Challenge (15.03)

    def _opts_epic_challenge(self, s: GameState, flow: Flow, side: str) -> List[StratagemAction]:
        u = s.unit(flow.window_unit)
        if unavailable(s, side, "epic_challenge", u) is not None:
            return []
        chars = [m for m in u.alive_models if is_character_model(u, m)]
        if not chars:
            return []
        # [PRECISION] ne sert que contre une unité qui a une figurine PERSONNAGE
        foes = self.fight_options(s, u)
        if not any(is_character_model(e, m) for e in foes for m in e.alive_models):
            return []
        return [StratagemAction("epic_challenge", u.id, model_id=m.id) for m in chars]

    def _use_epic_challenge(self, s: GameState, flow: Flow, action: StratagemAction) -> None:
        u = s.unit(action.unit_id)
        cost = record_use(s, u.side, "epic_challenge", u.id, detail=action.model_id)
        self._strat_say(s, u.side, "epic_challenge", f"les armes de mêlée de {action.model_id} ({u.name}) gagnent [PRECISION]", cost, unit=u.id, model=action.model_id)

    # --- Explosives (15.05), dans la phase de tir du joueur actif

    def _use_explosives(self, s: GameState, flow: Flow, action: StratagemAction) -> None:
        u, t = s.unit(action.unit_id), s.unit(action.target_id)
        cost = record_use(s, u.side, "explosives", u.id, detail=t.id)
        rolls = [s.d6() for _ in range(6)]
        mw = sum(1 for r in rolls if r >= 4)
        lost = self.mortal_wounds(s, t, mw)
        self._strat_say(s, u.side, "explosives", f"{u.name} lance ses grenades sur {t.name} : {rolls} → {mw} BM ({lost} fig.)"
                        + (" — UNITÉ DÉTRUITE" if t.is_destroyed else ""), cost, unit=u.id, target=t.id, rolls=rolls, mortal_wounds=mw)
        self._notify(s, u.side, s.log[-1] if s.log else "Explosives")
        if t.is_destroyed:
            s.destroyed_this_turn += 1
            self.on_unit_destroyed(s, t)

    # --- Crushing Impact (15.06)

    def _opts_crushing_impact(self, s: GameState, flow: Flow, side: str) -> List[StratagemAction]:
        u = s.unit(flow.window_unit)
        if u.is_destroyed or unavailable(s, side, "crushing_impact", u) is not None:
            return []
        er = s.rules.engagement_range_in
        out = []
        for e in s.enemies_in_engagement_range(u):
            engaged = [m for m in u.alive_models if any(within(m.disk, d, er) for d in e.disks())]
            if engaged:  # la figurine la plus robuste lance le plus de dés
                best = max(engaged, key=lambda m: m.profile.toughness)
                out.append(StratagemAction("crushing_impact", u.id, target_id=e.id, model_id=best.id))
        return out

    def _use_crushing_impact(self, s: GameState, flow: Flow, action: StratagemAction) -> None:
        u, t = s.unit(action.unit_id), s.unit(action.target_id)
        m = next(q for q in u.models if q.id == action.model_id)
        cost = record_use(s, u.side, "crushing_impact", u.id, detail=t.id)
        rolls = [s.d6() for _ in range(m.profile.toughness)]
        own = min(6, sum(1 for r in rolls if r == 1))
        dealt = min(6, sum(1 for r in rolls if r >= 5))
        lost_t = self.mortal_wounds(s, t, dealt)
        lost_u = self.mortal_wounds(s, u, own)
        self._strat_say(s, u.side, "crushing_impact", f"{u.name} écrase {t.name} : {len(rolls)}D6 {rolls} → {dealt} BM à {t.name} ({lost_t} fig.), "
                        f"{own} BM à {u.name} ({lost_u} fig.)", cost, unit=u.id, target=t.id, rolls=rolls)
        self._notify(s, u.side, s.log[-1] if s.log else "Crushing Impact")
        if t.is_destroyed:
            s.destroyed_this_turn += 1
            self.on_unit_destroyed(s, t)
        if u.is_destroyed:
            self.on_unit_destroyed(s, u)

    # --- Fire Overwatch (15.08) : fin de la phase de mouvement adverse

    def _opts_fire_overwatch(self, s: GameState, flow: Flow, side: str) -> List[StratagemAction]:
        if unavailable(s, side, "fire_overwatch") is not None:
            return []
        out = []
        for u in s.units_of(side):
            if u.has_keyword("Titanic") or s.is_engaged(u) or unavailable(s, side, "fire_overwatch", u) is not None:
                continue
            if not any(m.ranged_weapons for m in u.alive_models):
                continue
            out += [StratagemAction("fire_overwatch", u.id, target_id=t.id) for t in snap_targets(s, u)]
        return out

    def _use_fire_overwatch(self, s: GameState, flow: Flow, action: StratagemAction) -> None:
        u, t = s.unit(action.unit_id), s.unit(action.target_id)
        cost = record_use(s, u.side, "fire_overwatch", u.id, detail=t.id)
        report = resolve_shooting(s, u, t, snap=True)
        self._strat_say(s, u.side, "fire_overwatch", f"tir d'opportunité — {report}", cost, unit=u.id, target=t.id, damage=report.damage,
                        slain=report.models_slain, destroyed=report.target_destroyed)
        for d in report.details:
            self._say(s, f"      {d}", "detail")
        self._notify(s, u.side, f"Fire Overwatch : {report}")
        if report.target_destroyed:
            self.on_unit_destroyed(s, t)
        if u.is_destroyed:
            s.destroyed_this_turn += 1
            self.on_unit_destroyed(s, u)

    # --- Rapid Ingress (15.07) : fin de la phase de mouvement adverse

    def _opts_rapid_ingress(self, s: GameState, flow: Flow, side: str) -> List[StratagemAction]:
        if s.battle_round < 2 or unavailable(s, side, "rapid_ingress") is not None:
            return []
        return [StratagemAction("rapid_ingress", u.id) for u in s.reserves_of(side)
                if not u.has_keyword("Aircraft") and unavailable(s, side, "rapid_ingress", u) is None]

    def _use_rapid_ingress(self, s: GameState, flow: Flow, action: StratagemAction) -> None:
        u = s.unit(action.unit_id)
        cost = record_use(s, u.side, "rapid_ingress", u.id)
        self._strat_say(s, u.side, "rapid_ingress", f"{u.name} arrive des réserves", cost, unit=u.id)
        flow.unit_id = u.id
        flow.ingress_side, flow.after_ingress = u.side, flow.resume
        flow.step = "ingress"

    # --- Smokescreen (15.10) : début de la phase de tir adverse

    def _opts_smokescreen(self, s: GameState, flow: Flow, side: str) -> List[StratagemAction]:
        if unavailable(s, side, "smokescreen") is not None:
            return []
        smoke = [u for u in s.units_of(side) if u.has_keyword("Smoke") and unavailable(s, side, "smokescreen", u) is None]
        if not smoke:
            return []
        threatened = {t.id for e in s.units_of(other_side(side)) if shooting_ineligibility(s, e) is None for t in shooting_targets(s, e)}
        return [StratagemAction("smokescreen", u.id) for u in smoke if u.id in threatened]

    def _use_smokescreen(self, s: GameState, flow: Flow, action: StratagemAction) -> None:
        u = s.unit(action.unit_id)
        cost = record_use(s, u.side, "smokescreen", u.id, detail=True)
        self._strat_say(s, u.side, "smokescreen", f"{u.name} a le couvert contre les tirs jusqu'à la fin de la phase", cost, unit=u.id)

    # --- Heroic Intervention (15.11) : fin de la phase de charge adverse

    def heroic_candidates(self, s: GameState, unit: Unit, mode: str) -> List[Unit]:
        """Cibles possibles (avant le jet) : Leap to Defend — unités ennemies qui ont fait un mouvement
        de charge cette phase, à 12" ; Into the Fray — toute unité ennemie à 6"."""
        enemies = s.enemies_of(unit.side)
        if mode == "leap":
            return [e for e in enemies if e.charged and unit.min_gap_to(e) <= s.rules.charge_declare_range_in + 1e-9]
        return [e for e in enemies if unit.min_gap_to(e) <= 6.0 + 1e-9]

    def _opts_heroic_intervention(self, s: GameState, flow: Flow, side: str) -> List[StratagemAction]:
        if unavailable(s, side, "heroic_intervention") is not None:
            return []
        out = []
        for u in s.units_of(side):
            if s.is_engaged(u) or unavailable(s, side, "heroic_intervention", u) is not None:
                continue
            if u.has_keyword("Vehicle") and not (u.has_keyword("Character") or u.has_keyword("Walker")):
                continue
            if not any(u.min_gap_to(e) <= s.rules.charge_declare_range_in + 1e-9 for e in s.enemies_of(side)):
                continue
            for mode in ("leap", "fray"):
                if self.heroic_candidates(s, u, mode) and unavailable(s, side, "heroic_intervention", u, mode=mode) is None:
                    out.append(StratagemAction("heroic_intervention", u.id, mode=mode))
        return out

    def _use_heroic_intervention(self, s: GameState, flow: Flow, action: StratagemAction) -> None:
        u = s.unit(action.unit_id)
        side = u.side
        cost = record_use(s, side, "heroic_intervention", u.id, detail=action.mode, mode=action.mode)
        label = "Leap to Defend" if action.mode == "leap" else "Into the Fray"
        flow.hi, flow.charger = True, side
        flow.unit_id = u.id
        flow.candidates = [e.id for e in self.heroic_candidates(s, u, action.mode)]
        roll = s.roll(2)
        capped = ""
        if action.mode == "fray" and roll > 6:
            capped = f" (plafonné à 6, jet {roll})"
            roll = 6
        flow.roll = roll
        self._strat_say(s, side, "heroic_intervention", f"{u.name} intervient ({label}) — jet de charge {roll}{capped}", cost, unit=u.id, mode=action.mode, roll=roll)
        flow.step = "charge_roll_result"

    # --- Counteroffensive (15.12) : juste après qu'une unité ennemie a combattu

    def _opts_counteroffensive(self, s: GameState, flow: Flow, side: str) -> List[StratagemAction]:
        if not flow.fights_first or unavailable(s, side, "counteroffensive") is not None:
            return []  # hors de l'étape Fights First, l'alternance donne déjà la main au joueur inactif
        return [StratagemAction("counteroffensive", u.id) for u in s.units_of(side)
                if not self.has_fights_first(s, u) and self.fight_eligible(s, u) and self.fight_options(s, u) and unavailable(s, side, "counteroffensive", u) is None]

    def _use_counteroffensive(self, s: GameState, flow: Flow, action: StratagemAction) -> None:
        u = s.unit(action.unit_id)
        cost = record_use(s, u.side, "counteroffensive", u.id)
        u.fights_first = True
        flow.forced_fighter = u.id
        flow.fight_side = u.side
        flow.fights_first = True
        self._strat_say(s, u.side, "counteroffensive", f"{u.name} gagne Fights First et combat tout de suite", cost, unit=u.id)

    # ================================================================ règles partagées

    def advance_roll(self, s: GameState, unit: Unit) -> int:
        """Jet d'Advance (D6), ou distance fixe si un effet dit « do not make an Advance roll »."""
        if s.effects:
            from .effects import active_effects

            fixed = [int(e.value) for e in active_effects(s, unit.id, ("advance_fixed",)) if e.value is not None]
            if fixed:
                return max(fixed)
        return s.d6()

    def move_of(self, s: GameState, unit: Unit) -> float:
        """M de l'unité avec les effets actifs (« add 2" to the Move characteristic »)."""
        return unit.move_in + (effect_total(s, unit.id, "move_mod") if s.effects else 0)

    def _mod(self, s: GameState, unit: Unit, kind: str) -> int:
        return effect_total(s, unit.id, kind) if s.effects else 0

    def _has(self, s: GameState, unit: Unit, kind: str) -> bool:
        return bool(s.effects) and has_effect(s, unit.id, kind)

    def has_fights_first(self, s: GameState, unit: Unit) -> bool:
        return unit.fights_first or self._has(s, unit, "fights_first")

    def deployment_ok(self, s: GameState, unit: Unit, x: float, y: float) -> bool:
        """L'unité, centrée en (x, y) dans sa formation actuelle, est-elle entièrement dans sa zone,
        sur la table et sans chevaucher les figurines déjà posées ?"""
        zone = s.layout.deployment_zones[unit.side]
        c = unit.centroid
        deployed = set(s.flow.deployed) if s.flow is not None else set()
        others = [m.disk for u in s.on_table_units() if u.id != unit.id and u.id in deployed for m in u.alive_models]
        for m in unit.alive_models:
            d = m.disk.moved_to(x + m.x - c[0], y + m.y - c[1])
            if not s.on_board(d) or not all(point_in_polygon(p, zone) for p in extreme_points(d)):
                return False
            if any(disk_gap(d, o) <= 0 for o in others):
                return False
        return True

    def deployment_candidates(self, s: GameState, unit: Unit, step: float = 2.0, limit: int = 80) -> List[DeployAction]:
        """Grille de positions légales (calcul vectorisé) ; au-delà de ``limit``, sous-échantillonnage
        régulier (sans tirer de dé : construire une décision ne doit jamais consommer d'aléa)."""
        zone = s.layout.deployment_zones[unit.side]
        x0, y0, x1, y1 = zone.bbox
        xs = np.arange(x0 + step / 2, x1, step)
        ys = np.arange(y0 + step / 2, y1, step)
        if not len(xs) or not len(ys):
            return []
        gy, gx = np.meshgrid(ys, xs, indexing="ij")  # parcours ligne par ligne, comme la grille d'origine
        centers = np.stack([gx.ravel(), gy.ravel()], axis=1)
        c = unit.centroid
        ms = unit.alive_models
        offsets = np.array([(m.x - c[0], m.y - c[1]) for m in ms], dtype=float)
        radii = np.array([m.radius for m in ms], dtype=float)
        deployed = set(s.flow.deployed) if s.flow is not None else set()
        others = [m for u in s.on_table_units() if u.id != unit.id and u.id in deployed for m in u.alive_models]
        oxy = np.array([(m.x, m.y) for m in others], dtype=float).reshape(-1, 2)
        orr = np.array([m.radius for m in others], dtype=float)
        if all(m.is_round for m in ms) and all(m.is_round for m in others):
            ok = deployment_grid_legal(centers, offsets, radii, zone.vertices, s.layout.board, oxy, orr)
        else:
            ok = deployment_grid_legal(centers, offsets, radii, zone.vertices, s.layout.board, oxy, orr,
                                       shapes=shape_arrays([m.disk for m in ms]), others_shapes=shape_arrays([m.disk for m in others]))
        out = [DeployAction(unit.id, round(float(x), 2), round(float(y), 2)) for x, y in centers[ok]]
        if len(out) > limit:
            stride = len(out) / limit
            out = [out[int(k * stride)] for k in range(limit)]
        return out

    def _desperate_escape(self, s: GameState, unit: Unit) -> str:
        """Desperate Escape (09.07), avant de bouger : un jet de danger par figurine."""
        rolls, mw, lost = self.hazard_rolls(s, unit, len(unit.alive_models))
        return f" — Desperate Escape, jets de danger {rolls} : {mw} BM ({lost} fig.)"

    def _after_desperate(self, s: GameState, unit: Unit) -> None:
        """Après un Desperate Escape d'une unité qui n'était pas battle-shocked : test de battle-shock."""
        if unit.is_destroyed:
            self.on_unit_destroyed(s, unit)
            return
        if unit.battle_shocked:
            return
        roll = s.roll(2)
        if roll < unit.leadership:
            unit.battle_shocked = True
            self._say(s, f"Battle-shock : {unit.name} rate son test après le Desperate Escape ({roll} < {unit.leadership}+)", "battle_shock", unit=unit.id, roll=roll, passed=False)
        else:
            self._say(s, f"Battle-shock : {unit.name} réussit son test après le Desperate Escape ({roll})", "battle_shock", unit=unit.id, roll=roll, passed=True)

    def apply_move_action(self, s: GameState, unit: Unit, action: Action, max_distance: float) -> str:
        """Applique un mouvement (option du moteur ou placement libre) ; retourne la ligne de journal."""
        if isinstance(action, ModelMoveAction):
            extra = self._desperate_escape(s, unit) if action.kind == MoveKind.DESPERATE else ""
            pos = _positions(action)
            longest = apply_model_positions(unit, {mid: p for mid, p in pos.items() if any(m.id == mid for m in unit.alive_models)})
            if action.kind == MoveKind.DESPERATE:
                unit.fell_back = True
                msg = f"Mouvement : {unit.name} se replie figurine par figurine (jusqu'à {longest:.1f}\"){extra}"
                self._say(s, msg, "move", unit=unit.id, move=action.kind, distance=round(longest, 2))
                self._after_desperate(s, unit)
                return msg
            if action.kind == MoveKind.FALL_BACK:
                unit.fell_back = True
                msg = f"Mouvement : {unit.name} se replie figurine par figurine (jusqu'à {longest:.1f}\")"
            elif action.kind == "advance_move":
                msg = f"Mouvement : {unit.name} avance figurine par figurine (jusqu'à {longest:.1f}\")"
            else:
                msg = f"Mouvement : {unit.name} bouge figurine par figurine (jusqu'à {longest:.1f}\")"
            self._say(s, msg, "move", unit=unit.id, move=action.kind, distance=round(longest, 2))
            return msg
        return self.apply_move(s, unit, action.move)

    def apply_move(self, s: GameState, unit: Unit, move: FormationMove) -> str:
        if move.kind == MoveKind.STATIONARY:
            msg = f"Mouvement : {unit.name} reste immobile"
            self._say(s, msg, "move", unit=unit.id, move="stationary")
            return msg
        if move.kind == MoveKind.ADVANCE:
            roll = self.advance_roll(s, unit)
            max_d = self.move_of(s, unit) + roll + self._mod(s, unit, "advance_mod")
            mv = shrink_to_legal(s, unit, move, max_d)
            unit.advanced = True
            if mv is None or mv.distance <= 1e-6:
                msg = f"Mouvement : {unit.name} déclare une Advance ({roll}) mais ne peut pas bouger"
                self._say(s, msg, "move", unit=unit.id, move="advance", roll=roll, distance=0.0)
                return msg
            apply_translation(unit, mv.dx, mv.dy)
            msg = f"Mouvement : {unit.name} Advance de {mv.distance:.1f}\" (M{unit.move_in:g} + {roll}) — {move.label}"
            self._say(s, msg, "move", unit=unit.id, move="advance", roll=roll, distance=round(mv.distance, 2))
            return msg
        if move.kind == MoveKind.DESPERATE:
            extra = self._desperate_escape(s, unit)
            apply_translation(unit, move.dx, move.dy)
            unit.fell_back = True
            msg = f"Mouvement : {unit.name} se replie de {move.distance:.1f}\" — {move.label}{extra}"
            self._say(s, msg, "move", unit=unit.id, move=move.kind, distance=round(move.distance, 2))
            self._after_desperate(s, unit)
            return msg
        apply_translation(unit, move.dx, move.dy)
        if move.kind == MoveKind.FALL_BACK:
            unit.fell_back = True
            msg = f"Mouvement : {unit.name} se replie de {move.distance:.1f}\" — {move.label}"
        else:
            msg = f"Mouvement : {unit.name} bouge de {move.distance:.1f}\" — {move.label}"
        self._say(s, msg, "move", unit=unit.id, move=move.kind, distance=round(move.distance, 2))
        return msg

    def on_unit_destroyed(self, s: GameState, unit: Unit) -> None:
        """Effets déclenchés par la destruction d'une unité : les unités à bord d'un transport
        débarquent d'urgence, puis Deadly Demise (V11 : D6, sur 6 chaque unité à 6" de la figurine
        détruite subit X blessures mortelles)."""
        if s.passengers(unit.id):
            self._emergency_disembark(s, unit)
        for a in unit.datasheet.core_abilities:
            if a.name != "Deadly Demise" or not a.parameter:
                continue
            roll = s.roll(1)
            if roll != s.rules.deadly_demise_trigger:
                self._say(s, f"Deadly Demise : {unit.name} explose ? D6 = {roll} — rien", "deadly_demise", unit=unit.id, roll=roll, mortal_wounds=0)
                continue
            expr = parse_dice(a.parameter)
            if expr is None:  # paramètre illisible : on n'inflige rien
                continue
            mw = expr.roll(s.rng)
            self._say(s, f"Deadly Demise : {unit.name} explose ! D6 = {roll} → {mw} blessure(s) mortelle(s) à chaque unité à {s.rules.deadly_demise_range_in:g}\"",
                      "deadly_demise", unit=unit.id, roll=roll, mortal_wounds=mw)
            wrecks = [m.disk for m in unit.models]  # les figurines détruites gardent leur position
            victims = [o for o in s.on_table_units() if o.id != unit.id
                       and any(disk_gap(w, d) <= s.rules.deadly_demise_range_in for w in wrecks for d in o.disks())]
            for other in victims:
                lost = self.mortal_wounds(s, other, mw)
                self._say(s, f"      {other.name} subit {mw} BM ({lost} fig. perdue(s))" + (" — UNITÉ DÉTRUITE" if other.is_destroyed else ""), "detail")
                if other.is_destroyed:
                    if other.side != s.side_to_move:
                        s.destroyed_this_turn += 1
                    self.on_unit_destroyed(s, other)

    # ================================================================ actions libres

    def free_action_error(self, s: GameState, decision: Decision, action: Action) -> Optional[str]:
        """Un agent (humain) peut proposer une action hors de la liste des options : déploiement
        libre, placement figurine par figurine, déclaration d'Advance, translation libre.
        Retourne un message d'erreur, ou None si l'action est légale."""
        uid = getattr(action, "unit_id", None)
        if decision.unit_id is not None and uid != decision.unit_id:
            return f"l'action concerne {uid}, la décision porte sur {decision.unit_id}"
        if decision.kind == "deploy":
            unit = s.unit(uid)
            if isinstance(action, DeployAction):
                if self.deployment_ok(s, unit, action.x, action.y):
                    return None
                if has_infiltrators(unit) and infiltrate_error(s, unit, formation_at(unit, action.x, action.y)) is None:
                    return None
                return "formation hors zone, hors table ou sur d'autres figurines"
            if isinstance(action, DeployModelsAction):
                err = check_model_positions(s, unit, _positions(action), "deploy", 0.0)
                if err and has_infiltrators(unit):
                    alt = infiltrate_error(s, unit, _positions(action))
                    return None if alt is None else f"{err} (Infiltrators : {alt})"
                return err
            return "action de déploiement attendue"
        if decision.kind == "ingress":
            unit = s.unit(uid)
            if isinstance(action, DeployModelsAction):
                return ingress_error(s, unit, _positions(action))
            if isinstance(action, DeployAction):
                return ingress_error(s, unit, formation_at(unit, action.x, action.y))
            return "placement d'arrivée attendu"
        if decision.kind in ("move", "advance_move"):
            unit = s.unit(uid)
            engaged = s.is_engaged(unit)
            max_d = decision.max_distance if decision.max_distance is not None else self.move_of(s, unit)
            if isinstance(action, DeclareAdvanceAction):
                if decision.kind != "move":
                    return "l'Advance est déjà déclarée"
                return "une unité engagée ne peut pas faire d'Advance" if engaged else None
            if isinstance(action, ModelMoveAction):
                if action.kind in (MoveKind.FALL_BACK, MoveKind.DESPERATE):
                    if not engaged:
                        return "Fall Back réservé aux unités engagées"
                    if decision.kind != "move":
                        return "pas de Fall Back après une Advance"
                    if action.kind == MoveKind.FALL_BACK and unit.battle_shocked:
                        return "une unité battle-shocked se replie en Desperate Escape"
                elif action.kind in (MoveKind.NORMAL, "advance_move"):
                    if engaged:
                        return "une unité engagée ne peut que rester immobile ou se replier"
                    if decision.kind == "advance_move" and action.kind != "advance_move":
                        return "placement d'Advance attendu"
                else:
                    return f"type de déplacement inconnu : {action.kind}"
                return check_model_positions(s, unit, _positions(action), action.kind, max_d)
            if isinstance(action, MoveAction):
                mv = action.move
                if mv.kind == MoveKind.STATIONARY:
                    return None
                if mv.kind in (MoveKind.FALL_BACK, MoveKind.DESPERATE):
                    if not engaged:
                        return "Fall Back réservé aux unités engagées"
                    if mv.kind == MoveKind.FALL_BACK and unit.battle_shocked:
                        return "une unité battle-shocked se replie en Desperate Escape"
                    return None if legal_translation(s, unit, mv.dx, mv.dy, mv.kind, max_d) else "repli illégal (distance, table, chevauchement, socle ennemi traversé ou portée d'engagement)"
                if engaged:
                    return "une unité engagée ne peut que rester immobile ou se replier"
                if mv.kind == MoveKind.NORMAL:
                    return None if legal_translation(s, unit, mv.dx, mv.dy, mv.kind, max_d) else "déplacement illégal (distance, table, chevauchement ou portée d'engagement)"
                if mv.kind == MoveKind.ADVANCE:
                    if decision.kind != "move":
                        return "l'Advance est déjà déclarée"
                    return None if mv.distance > 1e-6 else "direction d'Advance vide"
            return "action de mouvement attendue"
        if decision.kind == "scout" and isinstance(action, ModelMoveAction):
            unit = s.unit(uid)
            x = float(decision.max_distance or 0.0)
            return check_model_positions(s, unit, _positions(action), MoveKind.NORMAL, x) or check_scout_positions(s, unit, _positions(action), x)
        if decision.kind == "scout" and isinstance(action, MoveAction):
            mv = action.move
            if mv.kind == MoveKind.STATIONARY:
                return None
            x = float(decision.max_distance or 0.0)
            unit = s.unit(uid)
            ok = legal_translation(s, unit, mv.dx, mv.dy, MoveKind.NORMAL, x)
            return None if ok and check_scout_positions(s, unit, {m.id: (m.x + mv.dx, m.y + mv.dy) for m in unit.alive_models}, x) is None \
                else "mouvement de scout illégal (distance, table, chevauchement, ou fin à 9\" d'un ennemi)"
        if decision.kind == "disembark" and isinstance(action, DisembarkModelsAction):
            unit = s.unit(uid)
            _, dist, allowed = disembark_mode(s, unit)
            return check_disembark_positions(s, unit, s.unit(unit.embarked_in), _positions(action), dist, allowed)
        if decision.kind == "charge_move" and isinstance(action, ModelMoveAction):
            if action.kind != "charge":
                return "placement de charge attendu"
            targets = [s.unit(t) for t in (decision.targets or (decision.target_id,))]
            return check_charge_positions(s, s.unit(uid), targets, _positions(action), int(decision.max_distance or 0))
        return "cette décision n'accepte que les options proposées"

    # ================================================================ rejouer

    def replay(self, initial: GameState, history, deploy: bool = True, first_player: Optional[str] = None) -> GameState:
        """Rejoue ``history`` ((camp, action), …) sur une copie de ``initial`` (même graine) :
        l'état obtenu est identique à celui de la partie d'origine."""
        s = initial.clone()
        self.start(s, deploy=deploy, first_player=first_player)
        for side, action in history:
            if isinstance(action, FREE_ACTIONS):
                self.apply_free(s, side, action, validate=False)
                continue
            d = self.decision(s)
            if d is None:
                raise IllegalAction("historique plus long que la partie")
            if d.side != side:
                raise IllegalAction(f"historique désynchronisé : {side} joue, la décision est pour {d.side}")
            self.step(s, action, validate=False)
        return s


def _positions(action) -> Dict[str, tuple]:
    """model_id → (x, y) ou (x, y, angle) : l'angle (radians) oriente les empreintes non rondes."""
    return {p[0]: tuple(float(v) for v in p[1:4]) for p in action.positions}


def _largest_coherent_group(unit: Unit, rules) -> set:
    """Plus grand groupe de figurines en cohérence (2" de proche en proche, 9" au plus) ; à égalité,
    celui qui garde le personnage et les premières figurines de la liste."""
    ms = unit.alive_models
    best: List = []
    for seed in ms:
        group, todo = [seed], [seed]
        while todo:
            a = todo.pop()
            for b in ms:
                if b not in group and within(a.disk, b.disk, rules.coherency_range_in):
                    group.append(b)
                    todo.append(b)
        while len(group) > 1:
            worst = max(group, key=lambda g: max(disk_gap(g.disk, o.disk) for o in group if o is not g))
            if max(disk_gap(worst.disk, o.disk) for o in group if o is not worst) <= rules.coherency_max_spread_in:
                break
            group.remove(worst)
        key = (len(group), any(m.is_leader for m in group), -min(ms.index(m) for m in group))
        if not best or key > best[0]:
            best = [key, group]
    return {m.id for m in best[1]} if best else set()
