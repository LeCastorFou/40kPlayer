"""Actions proposées aux agents et décisions qui les encadrent.

Le moteur ne demande jamais « que veux-tu faire ? » en général : il pose une
:class:`Decision` précise (déployer telle unité, choisir la cible de tel tir…) avec la
liste finie des :data:`Action` légales, et l'agent en renvoie une. C'est ce qui rend
l'espace d'actions énumérable pour la recherche arborescente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

from .movement import FormationMove

__all__ = [
    "DeployAction",
    "DeployModelsAction",
    "SelectUnitAction",
    "EndPhaseAction",
    "ContinueAction",
    "OathAction",
    "MoveAction",
    "ModelMoveAction",
    "DeclareAdvanceAction",
    "ShootAction",
    "ChargeAction",
    "DeclareChargeAction",
    "AutoChargeMoveAction",
    "FightAction",
    "EmbarkAction",
    "DisembarkAction",
    "DisembarkModelsAction",
    "StratagemAction",
    "ReserveAction",
    "ManualAction",
    "UseStratagemAction",
    "FREE_ACTIONS",
    "Action",
    "Decision",
]


@dataclass(frozen=True)
class DeployAction:
    unit_id: str
    x: float
    y: float

    def __str__(self) -> str:
        return f"déploie {self.unit_id} en ({self.x:.1f}, {self.y:.1f})"


@dataclass(frozen=True)
class DeployModelsAction:
    """Déploiement figurine par figurine : ``positions`` = ((model_id, x, y), …) pour toutes les figurines."""

    unit_id: str
    positions: tuple

    def __str__(self) -> str:
        return f"déploie {self.unit_id} figurine par figurine"


@dataclass(frozen=True)
class ModelMoveAction:
    """Déplacement figurine par figurine. ``kind`` : normal | fall_back | advance_move (après le jet).
    ``positions`` = ((model_id, x, y), …) ; une figurine absente reste en place."""

    unit_id: str
    kind: str
    positions: tuple

    def __str__(self) -> str:
        return f"{self.unit_id} : déplacement {self.kind} de {len(self.positions)} figurine(s)"


@dataclass(frozen=True)
class DeclareAdvanceAction:
    """Déclare une Advance : le moteur lance le D6 puis redemande le placement (decision « advance_move »)."""

    unit_id: str

    def __str__(self) -> str:
        return f"{self.unit_id} déclare une Advance"


@dataclass(frozen=True)
class DeclareChargeAction:
    """Déclare une charge : le moteur lance 2D6 puis propose les cibles atteignables."""

    unit_id: str

    def __str__(self) -> str:
        return f"{self.unit_id} déclare une charge"


@dataclass(frozen=True)
class AutoChargeMoveAction:
    """Laisse le moteur placer les figurines de la charge (au contact, cohérence, hors ER des autres)."""

    unit_id: str
    target_id: str

    def __str__(self) -> str:
        return f"{self.unit_id} : placement automatique de la charge sur {self.target_id}"


@dataclass(frozen=True)
class SelectUnitAction:
    """Choix de la prochaine unité à activer dans la phase (déploiement, mouvement, tir, charge)."""

    unit_id: str

    def __str__(self) -> str:
        return f"active {self.unit_id}"


@dataclass(frozen=True)
class EndPhaseAction:
    """Termine la phase sans activer les unités restantes."""

    def __str__(self) -> str:
        return "termine la phase"


@dataclass(frozen=True)
class ContinueAction:
    """Accusé de réception d'une action adverse (mode pas à pas)."""

    def __str__(self) -> str:
        return "continuer"


@dataclass(frozen=True)
class OathAction:
    target_id: Optional[str]

    def __str__(self) -> str:
        return f"Oath of Moment → {self.target_id}" if self.target_id else "pas de serment"


@dataclass(frozen=True)
class MoveAction:
    unit_id: str
    move: FormationMove

    def __str__(self) -> str:
        return f"{self.unit_id} : {self.move.label or self.move.kind}"


@dataclass(frozen=True)
class ShootAction:
    unit_id: str
    target_id: Optional[str]  #: None = ne tire pas

    def __str__(self) -> str:
        return f"{self.unit_id} tire sur {self.target_id}" if self.target_id else f"{self.unit_id} ne tire pas"


@dataclass(frozen=True)
class ChargeAction:
    """Choix des cibles de la charge après le jet (V11 11.04 : une ou plusieurs unités à portée du jet)."""

    unit_id: str
    target_id: Optional[str]  #: None = pas de charge ; sinon première cible
    extra_targets: tuple = ()  #: autres cibles de la même charge

    @property
    def targets(self) -> tuple:
        return () if self.target_id is None else (self.target_id,) + tuple(self.extra_targets)

    def __str__(self) -> str:
        return f"{self.unit_id} charge {' + '.join(self.targets)}" if self.target_id else f"{self.unit_id} ne charge pas"


@dataclass(frozen=True)
class FightAction:
    unit_id: str
    target_id: str

    def __str__(self) -> str:
        return f"{self.unit_id} frappe {self.target_id}"


@dataclass(frozen=True)
class EmbarkAction:
    """Embarquer dans un transport ami (au déploiement, ou après un mouvement à 3" de lui) ;
    ``transport_id`` None = ne pas embarquer."""

    unit_id: str
    transport_id: Optional[str]

    def __str__(self) -> str:
        return f"{self.unit_id} embarque dans {self.transport_id}" if self.transport_id else f"{self.unit_id} n'embarque pas"


@dataclass(frozen=True)
class DisembarkAction:
    """Débarquer en formation centrée en (x, y) ; ``x`` None = rester à bord."""

    unit_id: str
    x: Optional[float] = None
    y: Optional[float] = None

    def __str__(self) -> str:
        return f"{self.unit_id} débarque en ({self.x:.1f}, {self.y:.1f})" if self.x is not None else f"{self.unit_id} reste à bord"


@dataclass(frozen=True)
class DisembarkModelsAction:
    """Débarquement figurine par figurine : ``positions`` = ((model_id, x, y[, angle]), …)."""

    unit_id: str
    positions: tuple

    def __str__(self) -> str:
        return f"{self.unit_id} débarque figurine par figurine"


@dataclass(frozen=True)
class ReserveAction:
    """Réserves stratégiques (20) : au déploiement, placer l'unité en réserve ; pendant un mouvement
    d'ingress, rester en réserve (l'unité est sélectionnée et reste immobile)."""

    unit_id: str

    def __str__(self) -> str:
        return f"{self.unit_id} en réserve"


@dataclass(frozen=True)
class StratagemAction:
    """Utiliser un stratagème (15) — ``stratagem`` = clé (voir :mod:`fortyk.engine.stratagems`) ;
    None = ne pas en utiliser (on laisse passer la fenêtre). ``unit_id`` : unité amie ciblée ;
    ``target_id`` : unité ennemie visée ; ``model_id`` : figurine choisie ; ``mode`` : variante
    (relance d'Advance ou de charge, Leap to Defend / Into the Fray)."""

    stratagem: Optional[str]
    unit_id: Optional[str] = None
    target_id: Optional[str] = None
    model_id: Optional[str] = None
    mode: Optional[str] = None

    def __str__(self) -> str:
        if self.stratagem is None:
            return "pas de stratagème"
        bits = [self.stratagem, self.unit_id or "", f"→ {self.target_id}" if self.target_id else "", self.mode or ""]
        return " ".join(b for b in bits if b)


@dataclass(frozen=True)
class ManualAction:
    """Effet manuel : ce qu'une règle pas encore traduite dans le moteur permet de faire, appliqué par
    un joueur à n'importe quel moment (journalisé, visible par l'adversaire, annulable).

    ``kind`` : note (règle invoquée, sans effet mécanique), cp (+/- ``value`` CP), heal (``value`` PV à
    ``model_ids``), revive (figurines détruites ``model_ids`` reposées en ``positions``, ``value`` PV
    chacune, défaut : tous), mortal (``value`` blessures mortelles), destroy (retirer ``model_ids``),
    reserve (retour en réserve), set_up (poser l'unité en ``positions``), move (déplacer des figurines
    en ``positions``), battleshock (``value`` 1 / 0), effect (effet à durée ``effect`` = genre,
    ``effect_value``, ``until``, ``scope``, ``vs``). ``rule`` : nom de la règle ; ``note`` : précision."""

    kind: str
    unit_id: Optional[str] = None
    model_ids: tuple = ()
    value: Optional[int] = None
    positions: tuple = ()
    effect: Optional[str] = None
    effect_value: Optional[str] = None
    until: str = "phase"
    scope: str = "all"
    vs: Optional[str] = None
    rule: str = ""
    note: str = ""

    def __str__(self) -> str:
        return f"effet manuel {self.kind} {self.unit_id or ''} ({self.rule})".strip()


@dataclass(frozen=True)
class UseStratagemAction:
    """Utiliser un stratagème du joueur (base ou détachement) depuis le panneau des stratagèmes : les
    CP et les limites de 15.01 sont vérifiés ; l'effet est appliqué si le moteur sait le traduire,
    sinon il est à appliquer à la main (effet manuel)."""

    stratagem_id: str
    unit_id: Optional[str] = None
    target_id: Optional[str] = None
    model_id: Optional[str] = None
    choice: Optional[str] = None  #: option choisie (« Select either the [LETHAL HITS] or [SUSTAINED HITS 1] ability »)
    note: str = ""

    def __str__(self) -> str:
        return f"stratagème {self.stratagem_id} {self.unit_id or ''}".strip()


#: actions libres : jouables par l'un ou l'autre joueur à tout moment, hors des décisions du moteur
FREE_ACTIONS = (ManualAction, UseStratagemAction)


Action = Union[
    DeployAction,
    DeployModelsAction,
    SelectUnitAction,
    EndPhaseAction,
    ContinueAction,
    OathAction,
    MoveAction,
    ModelMoveAction,
    DeclareAdvanceAction,
    ShootAction,
    ChargeAction,
    DeclareChargeAction,
    AutoChargeMoveAction,
    FightAction,
    EmbarkAction,
    DisembarkAction,
    DisembarkModelsAction,
    StratagemAction,
    ReserveAction,
]


@dataclass
class Decision:
    """Une question posée à l'agent : quoi (``kind``), pour qui (``side``), avec quelles options.

    ``kind`` : select_unit | deploy | oath | move | advance_move | shoot | charge_declare |
    charge_target | charge_move | fight | observe | embark | disembark | scout | stratagem | ingress.
    ``max_distance`` : pour move / advance_move / charge_move, distance maximale par figurine
    (M, M + jet d'Advance, ou jet de charge). ``target_id`` : pour charge_move, la première cible ;
    ``targets`` : toutes les cibles de la charge.
    ``phase`` : pour select_unit, la phase concernée. ``ineligible`` : unités non activables et pourquoi.
    ``window`` : pour stratagem, la fenêtre de réaction (voir :mod:`fortyk.engine.stratagems`).
    """

    kind: str
    side: str
    options: List[Action]
    unit_id: Optional[str] = None
    note: str = ""
    max_distance: Optional[float] = None
    phase: Optional[str] = None
    ineligible: Dict[str, str] = field(default_factory=dict)
    target_id: Optional[str] = None
    targets: tuple = ()
    window: str = ""

    def __post_init__(self):
        if not self.options:
            raise ValueError(f"décision {self.kind} sans option")
