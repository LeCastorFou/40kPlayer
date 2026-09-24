"""Mission principale et décompte des points de victoire.

Mission du toy model : **Unstoppable Force** (Take and Hold contre Take and Hold,
mission deck 2026-27). Texte de la carte :

* N'importe quel round, fin de ton tour : une ou plusieurs unités ennemies ont été
  détruites ce tour → 3 VP.
* À partir du deuxième round, fin de ta phase de commandement (ou fin de ton tour au
  cinquième round) : pour chaque objectif que tu contrôles (hors home objective) → 4 VP.
* À partir du deuxième round, fin de ton tour : tu contrôles un ou plusieurs objectifs
  que tu ne contrôlais pas au début du tour (hors home objective) → 3 VP.
* Fin de la bataille : tu contrôles un ou plusieurs objectifs centraux → 5 VP.

Plafond : 15 VP marqués par round de bataille et par joueur, hors fin de bataille.

Le contrôle d'un objectif suit la règle de base : chaque camp additionne les OC de ses
figurines à portée de l'objectif (« Level of Control ») ; le camp dont le total est
strictement supérieur contrôle l'objectif. Les objectifs « collants » (Objective
Secured des Intercessors, Objective Defiled des Tormentors) sont gérés par l'état de
partie, qui fournit à cette couche l'ensemble des objectifs contrôlés.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import math

from .geometry import Disk, point_core_distance, within_of_point
from .layout import Layout, ObjectivePoint
from .rules import DEFAULT_RULES, RulesConfig

__all__ = ["ScoringEvent", "Scoreboard", "UnstoppableForce", "level_of_control", "controller"]

SIDES = ("attacker", "defender")


@dataclass(frozen=True)
class ScoringEvent:
    side: str
    battle_round: int
    when: str  #: "command" | "end_turn" | "end_battle"
    reason: str
    vp: int


@dataclass
class Scoreboard:
    """Accumule les événements de score et applique le plafond par round."""

    round_cap: int = 15
    events: List[ScoringEvent] = field(default_factory=list)

    def add(self, event: ScoringEvent) -> int:
        """Enregistre l'événement, plafonné ; retourne les VP effectivement marqués."""
        if event.vp <= 0:
            return 0
        if event.when == "end_battle":
            self.events.append(event)
            return event.vp
        already = self.round_total(event.side, event.battle_round)
        allowed = max(0, min(event.vp, self.round_cap - already))
        if allowed:
            self.events.append(ScoringEvent(event.side, event.battle_round, event.when, event.reason, allowed))
        return allowed

    def add_all(self, events: Iterable[ScoringEvent]) -> int:
        return sum(self.add(e) for e in events)

    def round_total(self, side: str, battle_round: int) -> int:
        return sum(e.vp for e in self.events if e.side == side and e.battle_round == battle_round and e.when != "end_battle")

    def total(self, side: str) -> int:
        return sum(e.vp for e in self.events if e.side == side)

    def totals(self) -> Dict[str, int]:
        return {side: self.total(side) for side in SIDES}

    def leader(self) -> Optional[str]:
        t = self.totals()
        if t["attacker"] == t["defender"]:
            return None
        return max(t, key=t.get)


# ---------------------------------------------------------------- contrôle


def disk_in_objective_range(disk: Disk, objective: ObjectivePoint, rules: RulesConfig = DEFAULT_RULES) -> bool:
    """V11 (14.02) : à portée d'un objectif de terrain = dans son empreinte (un orteil suffit) ; à portée
    d'un pion de 40 mm (hors décor) = à 3" de son bord."""
    if objective.rect is not None:
        return Layout._rect_touches_disk(objective.rect, disk)
    return within_of_point(disk, objective.center, rules.objective_control_radius_in)


def disk_objective_distance(disk: Disk, objective: ObjectivePoint, rules: RulesConfig = DEFAULT_RULES) -> float:
    """Distance du socle à l'objectif (à son empreinte, ou au bord du pion) ; 0 s'il le touche."""
    if objective.rect is not None:
        x0, y0, x1, y1 = objective.rect
        if disk.hx == 0.0 and disk.hy == 0.0:
            dx = max(x0 - disk.x, 0.0, disk.x - x1)
            dy = max(y0 - disk.y, 0.0, disk.y - y1)
            return max(0.0, math.hypot(dx, dy) - disk.r)
        from .geometry import Polygon, disk_polygon_distance

        return disk_polygon_distance(disk, Polygon.rect(x0, y0, x1, y1))
    return max(0.0, point_core_distance(objective.center, disk) - disk.r - rules.objective_marker_radius_in)


def level_of_control(models: Iterable[Tuple[Disk, int]], objective: ObjectivePoint, rules: RulesConfig = DEFAULT_RULES) -> int:
    """Somme des OC des figurines ``(socle, OC)`` à portée de l'objectif.
    Les figurines battle-shockées doivent être passées avec OC 0."""
    return sum(oc for disk, oc in models if disk_in_objective_range(disk, objective, rules))


def controller(levels: Dict[str, int]) -> Optional[str]:
    """Camp qui contrôle l'objectif d'après les Level of Control ; None si égalité (0-0 inclus)."""
    a = levels.get("attacker", 0)
    d = levels.get("defender", 0)
    if a > d:
        return "attacker"
    if d > a:
        return "defender"
    return None


# ---------------------------------------------------------------- mission


@dataclass(frozen=True)
class UnstoppableForce:
    name: str = "Unstoppable Force"
    vp_destroyed_unit: int = 3
    vp_per_objective: int = 4
    vp_new_objective: int = 3
    vp_central_end: int = 5
    objectives_from_round: int = 2
    final_round: int = 5

    @staticmethod
    def _scoring_objectives(layout: Layout, controlled: Set[str], side: str) -> List[ObjectivePoint]:
        """Objectifs contrôlés qui rapportent : tous sauf le home objective du camp."""
        home = layout.home_objective(side)
        return [layout.objective(i) for i in controlled if not (home and i == home.id)]

    def score_command_phase(self, side: str, battle_round: int, controlled: Set[str], layout: Layout) -> List[ScoringEvent]:
        """Fin de la phase de commandement de ``side``. Au round final ce décompte passe en fin de tour."""
        if battle_round < self.objectives_from_round or battle_round >= self.final_round:
            return []
        n = len(self._scoring_objectives(layout, controlled, side))
        if not n:
            return []
        return [ScoringEvent(side, battle_round, "command", f"{n} objectif(s) contrôlé(s)", n * self.vp_per_objective)]

    def score_end_of_turn(
        self,
        side: str,
        battle_round: int,
        controlled_now: Set[str],
        controlled_at_start: Set[str],
        enemy_units_destroyed: int,
        layout: Layout,
    ) -> List[ScoringEvent]:
        events = []
        if enemy_units_destroyed > 0:
            events.append(ScoringEvent(side, battle_round, "end_turn", "unité(s) ennemie(s) détruite(s)", self.vp_destroyed_unit))
        if battle_round >= self.objectives_from_round:
            if battle_round >= self.final_round:
                n = len(self._scoring_objectives(layout, controlled_now, side))
                if n:
                    events.append(ScoringEvent(side, battle_round, "end_turn", f"{n} objectif(s) contrôlé(s) (round final)", n * self.vp_per_objective))
            new = {o.id for o in self._scoring_objectives(layout, controlled_now, side)} - controlled_at_start
            if new:
                events.append(ScoringEvent(side, battle_round, "end_turn", f"objectif(s) nouvellement pris : {', '.join(sorted(new))}", self.vp_new_objective))
        return events

    def score_end_of_battle(self, side: str, controlled: Set[str], layout: Layout) -> List[ScoringEvent]:
        central = [o.id for o in layout.objectives if o.kind == "central" and o.id in controlled]
        if not central:
            return []
        return [ScoringEvent(side, self.final_round, "end_battle", "objectif central contrôlé", self.vp_central_end)]

    def max_per_round(self, layout: Layout, side: str) -> int:
        """Borne théorique hors plafond : utile à la fonction d'évaluation de l'IA."""
        n = len([o for o in layout.objectives if not (o.is_home and o.owner == side)])
        return n * self.vp_per_objective + self.vp_destroyed_unit + self.vp_new_objective
