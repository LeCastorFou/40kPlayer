"""État de partie : figurines, unités, plateau, contrôle des objectifs.

Les objets sont mutables (le moteur les modifie en place au fil des phases).
:meth:`GameState.clone` fait une copie rapide pour la recherche arborescente : figurines,
unités, score, dés et curseur de flot sont copiés, tout ce qui est immuable (fiches, armes,
carte, décors, règles) est partagé. :class:`Flow` est le « compteur de programme » de la
machine à états du moteur (:mod:`fortyk.engine.engine`) : où en est la partie, qui décide,
et le contexte de la sous-étape en cours.
Tout ce qui est une *règle* (portées, seuils) vient de :class:`RulesConfig`.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from ..data.models import Datasheet, ModelProfile, Weapon
from .geometry import EPS, Disk, Point, Terrain, disk_gap, dist, extreme_points, point_core_distance, within, within_of_point
from .layout import Layout, ObjectivePoint
from .mission import Scoreboard, UnstoppableForce, controller, disk_in_objective_range, disk_objective_distance, level_of_control
from .rules import DEFAULT_RULES, RulesConfig

__all__ = ["SIDES", "other_side", "Model", "Unit", "GameState", "Flow"]

SIDES = ("attacker", "defender")


def other_side(side: str) -> str:
    return "defender" if side == "attacker" else "attacker"


# ------------------------------------------------------------------ figurines


@dataclass
class Model:
    id: str
    unit_id: str
    name: str  #: « Intercessor Sergeant », « Infractor »…
    profile: ModelProfile
    weapons: Tuple[Weapon, ...]
    x: float = 0.0
    y: float = 0.0
    wounds: int = 0
    alive: bool = True
    is_leader: bool = False  #: figurine d'un personnage qui mène l'unité (protégée à l'allocation)
    angle: float = 0.0  #: orientation (radians) de l'empreinte non ronde (axe long)
    radius: float = field(init=False, default=0.0)  #: rayon du socle rond, ou d'arrondi de l'empreinte, en pouces
    hx: float = field(init=False, default=0.0)  #: demi-longueur du cœur (ovale, coque), en pouces
    hy: float = field(init=False, default=0.0)  #: demi-largeur du cœur (coque), en pouces

    def __post_init__(self):
        base = self.profile.base
        if base is None:
            self.radius = 32 / 25.4 / 2  # « Use model » : remplacé par la coque dans army.build_units_from_list
        elif base.shape == "oval" and base.width_mm != base.depth_mm:
            # ovale → « stade » : demi-petit axe en rayon, segment le long du grand axe
            big, small = max(base.width_mm, base.depth_mm), min(base.width_mm, base.depth_mm)
            self.radius = small / 2 / 25.4
            self.hx = (big - small) / 2 / 25.4
        else:
            self.radius = base.width_mm / 2 / 25.4

    def set_hull(self, length_in: float, width_in: float) -> None:
        """Empreinte rectangulaire (coque de véhicule) : longueur × largeur en pouces."""
        self.radius = 0.0
        self.hx = length_in / 2
        self.hy = width_in / 2

    def set_footprint(self, shape: str, length_in: float, width_in: float) -> None:
        """Empreinte sans socle d'origine : ``rect`` (coque), ``oval`` (stade) ou ``round``
        (``width_in`` = diamètre)."""
        if shape == "rect":
            self.set_hull(length_in, width_in)
        elif shape == "oval" and length_in > width_in:
            self.radius, self.hx, self.hy = width_in / 2, (length_in - width_in) / 2, 0.0
        else:
            self.radius, self.hx, self.hy = width_in / 2, 0.0, 0.0

    @property
    def is_round(self) -> bool:
        return self.hx == 0.0 and self.hy == 0.0

    @property
    def extent(self) -> float:
        """Rayon du cercle circonscrit à l'empreinte."""
        return self.radius + math.hypot(self.hx, self.hy)

    @property
    def disk(self) -> Disk:
        if self.hx == 0.0 and self.hy == 0.0:
            return Disk(self.x, self.y, self.radius)
        return Disk(self.x, self.y, self.radius, self.hx, self.hy, self.angle)

    @property
    def position(self) -> Point:
        return (self.x, self.y)

    @property
    def ranged_weapons(self) -> Tuple[Weapon, ...]:
        return tuple(w for w in self.weapons if not w.is_melee)

    @property
    def melee_weapons(self) -> Tuple[Weapon, ...]:
        return tuple(w for w in self.weapons if w.is_melee)

    def move_to(self, x: float, y: float, angle: Optional[float] = None) -> None:
        self.x, self.y = x, y
        if angle is not None:
            self.angle = angle

    def take_damage(self, amount: int) -> bool:
        """Applique des dégâts ; retourne True si la figurine est détruite."""
        if not self.alive:
            return False
        self.wounds -= amount
        if self.wounds <= 0:
            self.wounds = 0
            self.alive = False
            return True
        return False


# ------------------------------------------------------------------ unités


@dataclass
class Unit:
    id: str
    side: str
    datasheet: Datasheet
    models: List[Model]
    starting_strength: int = 0
    # --- drapeaux de tour (remis à zéro au début du tour de leur camp)
    moved_in: float = 0.0  #: plus grand déplacement d'une figurine ce tour (Heavy : < 3")
    remained_stationary: bool = True
    advanced: bool = False
    fell_back: bool = False
    charged: bool = False
    has_shot: bool = False
    has_fought: bool = False
    engaged_at_turn_start: Set[str] = field(default_factory=set)  #: unités ennemies à portée d'engagement au début du tour
    disembarked: bool = False  #: a débarqué ce tour (ne peut pas ré-embarquer dans la même phase)
    no_charge: bool = False  #: ne peut pas charger ce tour (débarquée d'un transport qui avait bougé, transport détruit)
    # --- statuts persistants
    battle_shocked: bool = False
    fights_first: bool = False
    embarked_in: Optional[str] = None  #: transport dans lequel l'unité est embarquée (hors table)
    last_shot_turn: int = -10  #: numéro du dernier tour de joueur où l'unité a fait des attaques à distance (Hidden)
    # --- composition (listes importées)
    leaders: Tuple[Datasheet, ...] = ()  #: personnages qui mènent l'unité (Leader) : leurs figurines sont dans ``models``
    enhancements: Tuple[str, ...] = ()  #: améliorations portées par l'unité ou ses personnages
    points: int = 0
    is_warlord: bool = False
    list_key: str = ""  #: clé de l'unité dans la liste d'armée (« Infractors[1] »)

    def __post_init__(self):
        if not self.starting_strength:
            self.starting_strength = len(self.models)

    # ----------------------------------------------------------- figurines

    @property
    def alive_models(self) -> List[Model]:
        return [m for m in self.models if m.alive]

    @property
    def strength(self) -> int:
        return len(self.alive_models)

    @property
    def is_destroyed(self) -> bool:
        return self.strength == 0

    @property
    def below_half_strength(self) -> bool:
        """Sous la moitié de son effectif : figurines restantes ≤ moitié de l'effectif de départ ;
        pour une unité d'une seule figurine (monstre, véhicule), PV restants ≤ moitié de ses PV."""
        if self.starting_strength == 1:
            m = self.models[0]
            return m.wounds * 2 <= m.profile.wounds
        return self.strength * 2 <= self.starting_strength

    @property
    def name(self) -> str:
        if self.leaders:
            return f"{self.datasheet.name} + {' + '.join(l.name for l in self.leaders)} [{self.id}]"
        return f"{self.datasheet.name} [{self.id}]"

    @property
    def datasheets(self) -> Tuple[Datasheet, ...]:
        """Fiche des gardes du corps puis celles des personnages qui mènent l'unité."""
        return (self.datasheet,) + tuple(self.leaders)

    @property
    def status(self) -> str:
        """« Intercessor Squad 4/5 » ou, pour une figurine seule, « Redemptor Dreadnought 7/12 PV »."""
        if self.starting_strength == 1:
            m = self.models[0]
            return f"{self.datasheet.name} {m.wounds}/{m.profile.wounds} PV"
        return f"{self.datasheet.name} {self.strength}/{self.starting_strength}"

    @property
    def profile(self) -> ModelProfile:
        return self.datasheet.models[0]

    @property
    def move_in(self) -> float:
        """M de l'unité : celui de la figurine la plus lente (une unité menée avance au pas de tous)."""
        if not self.leaders:
            return self.profile.move_in or 0.0
        return min((m.profile.move_in or 0.0) for m in self.alive_models) if self.alive_models else 0.0

    @property
    def leadership(self) -> int:
        """Meilleur Ld de l'unité (le plus petit seuil)."""
        return min(m.profile.leadership for m in self.alive_models) if self.alive_models else self.profile.leadership

    def disks(self) -> List[Disk]:
        return [m.disk for m in self.alive_models]

    @property
    def centroid(self) -> Point:
        ms = self.alive_models
        return (sum(m.x for m in ms) / len(ms), sum(m.y for m in ms) / len(ms))

    def has_keyword(self, kw: str) -> bool:
        return self.datasheet.has_keyword(kw) or any(l.has_keyword(kw) for l in self.leaders)

    def has_ability(self, name: str) -> bool:
        if self.datasheet.ability(name) is not None or any(l.ability(name) is not None for l in self.leaders):
            return True
        key = name.lower()
        return any(e.lower() == key for e in self.enhancements)

    @property
    def oc_per_model(self) -> int:
        return 0 if self.battle_shocked else self.profile.oc

    def model_oc(self, m: "Model") -> int:
        return 0 if self.battle_shocked else m.profile.oc

    # ----------------------------------------------------------- distances

    def min_gap_to(self, other: "Unit") -> float:
        """Plus petite distance socle-à-socle entre deux unités (inf si l'une est détruite)."""
        best = float("inf")
        mine = [a for a in self.models if a.alive]
        theirs_m = [b for b in other.models if b.alive]
        if not all(m.hx == 0.0 and m.hy == 0.0 for m in mine) or not all(m.hx == 0.0 and m.hy == 0.0 for m in theirs_m):
            # empreintes non rondes : paires rondes en direct, les autres seulement si leurs cercles
            # circonscrits peuvent battre le meilleur écart trouvé
            hypot = math.hypot
            for a in mine:
                a_round = a.hx == 0.0 and a.hy == 0.0
                ea = a.extent
                da = None
                for b in theirs_m:
                    cd = hypot(a.x - b.x, a.y - b.y)
                    if a_round and b.hx == 0.0 and b.hy == 0.0:
                        g = cd - a.radius - b.radius
                    elif cd - ea - b.extent >= best:
                        continue
                    else:
                        da = da or a.disk
                        g = disk_gap(da, b.disk)
                    if g < best:
                        best = g
            return max(0.0, best) if best != float("inf") else best
        hypot = math.hypot
        theirs = [(b.x, b.y, b.radius) for b in theirs_m]
        for a in mine:
            ax, ay, ar = a.x, a.y, a.radius
            for bx, by, br in theirs:
                g = hypot(ax - bx, ay - by) - ar - br
                if g < best:
                    best = g
        return max(0.0, best) if best != float("inf") else best

    def in_engagement_range_of(self, other: "Unit", rules: RulesConfig = DEFAULT_RULES) -> bool:
        return self.min_gap_to(other) <= rules.engagement_range_in

    def models_in_engagement_range(self, other: "Unit", rules: RulesConfig = DEFAULT_RULES) -> List[Model]:
        er = rules.engagement_range_in
        enemies = other.disks()
        return [m for m in self.alive_models if any(within(m.disk, e, er) for e in enemies)]

    def models_in_objective_range(self, objective: ObjectivePoint, rules: RulesConfig = DEFAULT_RULES) -> List[Model]:
        return [m for m in self.alive_models if disk_in_objective_range(m.disk, objective, rules)]

    def coherency_ok(self, rules: RulesConfig = DEFAULT_RULES) -> bool:
        ms = self.alive_models
        if len(ms) <= 1:
            return True
        for m in ms:
            others = [o for o in ms if o is not m]
            if not any(within(m.disk, o.disk, rules.coherency_range_in) for o in others):
                return False
            if any(disk_gap(m.disk, o.disk) > rules.coherency_max_spread_in for o in others):
                return False
        return True

    # ----------------------------------------------------------- tour

    def clone(self) -> "Unit":
        """Copie indépendante (figurines comprises) ; fiche et armes partagées."""
        u = copy.copy(self)
        u.models = [copy.copy(m) for m in self.models]
        u.engaged_at_turn_start = set(self.engaged_at_turn_start)
        return u

    def reset_turn_flags(self) -> None:
        self.moved_in = 0.0
        self.remained_stationary = True
        self.advanced = False
        self.fell_back = False
        self.charged = False
        self.has_shot = False
        self.has_fought = False
        self.fights_first = False
        self.engaged_at_turn_start = set()
        self.disembarked = False
        self.no_charge = False

    def allocate_damage(self, damages: Iterable[int]) -> int:
        """Alloue une liste de dégâts (une entrée par blessure passée) selon la règle :
        d'abord une figurine déjà blessée, puis les autres ; l'excédent est perdu.
        Retourne le nombre de figurines détruites."""
        killed = 0
        for dmg in damages:
            target = self._allocation_target()
            if target is None:
                break
            if target.take_damage(dmg):
                killed += 1
        return killed

    def allocate_mortal_wounds(self, n: int) -> int:
        """Les blessures mortelles débordent d'une figurine sur la suivante."""
        killed = 0
        remaining = n
        while remaining > 0:
            target = self._allocation_target()
            if target is None:
                break
            dealt = min(remaining, target.wounds)
            remaining -= dealt
            if target.take_damage(dealt):
                killed += 1
        return killed

    def mortal_wound_target(self) -> Optional[Model]:
        """Figurine qui subit la prochaine blessure mortelle (06.02) : une non-PERSONNAGE blessée,
        sinon une non-PERSONNAGE, sinon un PERSONNAGE blessé, sinon un PERSONNAGE."""
        return self._allocation_target()

    def _allocation_target(self) -> Optional[Model]:
        alive = self.alive_models
        if not alive:
            return None
        # Les personnages qui mènent l'unité ne prennent les blessures qu'une fois les gardes du corps morts.
        pool = [m for m in alive if not m.is_leader] or alive
        wounded = [m for m in pool if m.wounds < m.profile.wounds]
        if wounded:
            return wounded[0]
        # Le chef d'escouade (première figurine de la liste) est protégé : on retire les autres d'abord.
        return pool[-1]


# ------------------------------------------------------------------ flot


@dataclass
class Flow:
    """Curseur de la machine à états : étape courante et contexte de la sous-étape.

    ``step`` est soit une étape automatique (le moteur l'exécute sans rien demander), soit une
    étape de décision (le moteur attend une action du camp ``decider``). Les listes (et non des
    ensembles) gardent un ordre déterministe, indispensable pour rejouer une partie à l'identique."""

    step: str = "setup"
    deploy: bool = True
    forced_first_player: Optional[str] = None
    turn_index: int = 0  #: 0 = premier joueur du round, 1 = second
    active: str = "attacker"  #: joueur dont c'est le tour (ou qui déploie)
    unit_id: Optional[str] = None  #: unité en cours d'activation
    target_id: Optional[str] = None
    roll: int = 0
    max_distance: float = 0.0
    queues: Dict[str, List[str]] = field(default_factory=dict)  #: déploiement : unités restant à poser
    deployed: List[str] = field(default_factory=list)
    pending: List[str] = field(default_factory=list)  #: mouvement : unités pas encore activées
    done: List[str] = field(default_factory=list)  #: tir : déjà activées ; charge : refusées ou ratées
    eligible: List[str] = field(default_factory=list)
    ineligible: Dict[str, str] = field(default_factory=dict)
    candidates: List[str] = field(default_factory=list)  #: charge : cibles à 12"
    reachable: List[str] = field(default_factory=list)  #: charge : cibles atteignables avec le jet
    targets: List[str] = field(default_factory=list)  #: charge : cibles choisies
    fight_start_engaged: List[str] = field(default_factory=list)  #: combat : unités engagées au début de l'étape de combat
    fight_seen: List[str] = field(default_factory=list)  #: combat : unités éligibles à combattre cette phase (consolidation)
    fights_first: bool = True
    fight_side: str = "attacker"
    last_activator: Optional[str] = None
    over_reason: str = ""
    decision: object = None  #: décision en attente, mise en cache (partagée entre clones : immuable)

    def clone(self) -> "Flow":
        f = copy.copy(self)
        f.queues = {k: list(v) for k, v in self.queues.items()}
        f.deployed = list(self.deployed)
        f.pending = list(self.pending)
        f.done = list(self.done)
        f.eligible = list(self.eligible)
        f.ineligible = dict(self.ineligible)
        f.candidates = list(self.candidates)
        f.reachable = list(self.reachable)
        f.targets = list(self.targets)
        f.fight_start_engaged = list(self.fight_start_engaged)
        f.fight_seen = list(self.fight_seen)
        return f


# ------------------------------------------------------------------ partie


@dataclass
class GameState:
    layout: Layout
    units: Dict[str, Unit]
    rules: RulesConfig = DEFAULT_RULES
    mission: UnstoppableForce = field(default_factory=UnstoppableForce)
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    battle_round: int = 0  #: 0 = déploiement
    side_to_move: str = "attacker"
    first_player: str = "attacker"
    phase: str = "deployment"
    scoreboard: Scoreboard = field(default_factory=Scoreboard)
    sticky_control: Dict[str, str] = field(default_factory=dict)  #: objectif → camp (Objective Secured)
    oath_target: Optional[str] = None  #: unité ennemie désignée par Oath of Moment (attaquant Space Marines)
    controlled_at_turn_start: Set[str] = field(default_factory=set)
    destroyed_this_turn: int = 0  #: unités ennemies détruites pendant le tour en cours
    turn_counter: int = 0  #: tours de joueur commencés depuis le début de la bataille (1 = premier tour)
    cp: Dict[str, int] = field(default_factory=lambda: {"attacker": 0, "defender": 0})  #: points de commandement
    charge_targets_this_phase: Set[str] = field(default_factory=set)
    log: List[str] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)  #: journal structuré (voir Engine._say)
    history: List[tuple] = field(default_factory=list)  #: (camp, action) de chaque décision, pour rejouer
    recording: bool = True  #: False : ni journal ni événements (simulations de l'IA)
    flow: Optional[Flow] = None
    army_lists: Optional[dict] = None  #: listes d'armée importées, par camp (fortyk.data.army_list.ArmyList)
    _terrain: Optional[List[Terrain]] = None
    _los: object = None

    def __post_init__(self):
        self.scoreboard.round_cap = self.rules.vp_round_cap

    # ----------------------------------------------------------- accès

    @property
    def terrain(self) -> List[Terrain]:
        if self._terrain is None:
            self._terrain = self.layout.terrain_objects()
        return self._terrain

    @property
    def los(self):
        """Index vectorisé des lignes de vue (:class:`~fortyk.engine.fastgeo.LineOfSight`), avec cache
        partagé par toutes les parties et copies sur les mêmes décors."""
        if self._los is None:
            from .fastgeo import line_of_sight_for

            self._los = line_of_sight_for(self.terrain)
        return self._los

    def units_of(self, side: str, alive_only: bool = True, include_embarked: bool = False) -> List[Unit]:
        """Unités du camp sur la table (les unités embarquées dans un transport n'y sont pas, sauf
        ``include_embarked``)."""
        return [u for u in self.units.values() if u.side == side and (not alive_only or not u.is_destroyed)
                and (include_embarked or u.embarked_in is None)]

    def on_table_units(self) -> List[Unit]:
        """Toutes les unités présentes sur la table (ni détruites ni embarquées)."""
        return [u for u in self.units.values() if not u.is_destroyed and u.embarked_in is None]

    def passengers(self, transport_id: str) -> List[Unit]:
        """Unités embarquées dans ce transport."""
        return [u for u in self.units.values() if u.embarked_in == transport_id and not u.is_destroyed]

    def enemies_of(self, side: str) -> List[Unit]:
        return self.units_of(other_side(side))

    def unit(self, unit_id: str) -> Unit:
        return self.units[unit_id]

    def all_alive_models(self, exclude_unit: Optional[str] = None) -> List[Model]:
        return [m for u in self.units.values() if u.id != exclude_unit and u.embarked_in is None for m in u.alive_models]

    def enemy_models(self, side: str) -> List[Model]:
        return [m for u in self.enemies_of(side) for m in u.alive_models]

    def enemies_in_engagement_range(self, unit: Unit) -> List[Unit]:
        return [e for e in self.enemies_of(unit.side) if unit.in_engagement_range_of(e, self.rules)]

    def is_engaged(self, unit: Unit) -> bool:
        return bool(self.enemies_in_engagement_range(unit))

    def on_board(self, disk: Disk) -> bool:
        w, h = self.layout.board
        if disk.hx == 0.0 and disk.hy == 0.0:
            return disk.x - disk.r >= -1e-9 and disk.y - disk.r >= -1e-9 and disk.x + disk.r <= w + 1e-9 and disk.y + disk.r <= h + 1e-9
        return all(-1e-9 <= px <= w + 1e-9 and -1e-9 <= py <= h + 1e-9 for px, py in extreme_points(disk))

    def unit_in_objective_range(self, unit: Unit, objective: ObjectivePoint) -> bool:
        return bool(unit.models_in_objective_range(objective, self.rules))

    def model_in_objective_range(self, model: Model, objective: ObjectivePoint) -> bool:
        return disk_in_objective_range(model.disk, objective, self.rules)

    def disk_in_objective_range(self, disk: Disk, objective: ObjectivePoint) -> bool:
        return disk_in_objective_range(disk, objective, self.rules)

    def unit_objective_distance(self, unit: Unit, objective: ObjectivePoint) -> float:
        return min((disk_objective_distance(m.disk, objective, self.rules) for m in unit.alive_models), default=float("inf"))

    def objective_anchor(self, objective: ObjectivePoint, point: Point) -> Point:
        """Point de l'objectif le plus proche de ``point`` (dans son empreinte), vers lequel avancer."""
        if objective.rect is None:
            return objective.center
        x0, y0, x1, y1 = objective.rect
        return (min(max(point[0], x0), x1), min(max(point[1], y0), y1))

    # ----------------------------------------------------------- objectifs

    def levels_of_control(self, objective: ObjectivePoint) -> Dict[str, int]:
        """Somme des OC à portée de l'objectif, par camp (voir :func:`mission.level_of_control`) ; un
        objectif de terrain compte les figurines dont le socle touche son empreinte (V11 14.02)."""
        rules = self.rules
        levels = {side: 0 for side in SIDES}
        for u in self.units.values():
            if u.battle_shocked or u.embarked_in is not None:
                continue
            for m in u.models:
                if m.alive and disk_in_objective_range(m.disk, objective, rules):
                    levels[u.side] += m.profile.oc
        return levels

    def objective_controller(self, objective: ObjectivePoint) -> Optional[str]:
        """Contrôleur d'un objectif : Level of Control, puis contrôle « collant » s'il n'y a pas de vainqueur."""
        who = controller(self.levels_of_control(objective))
        if who is not None:
            return who
        return self.sticky_control.get(objective.id)

    def controlled_objectives(self, side: str) -> Set[str]:
        return {o.id for o in self.layout.objectives if self.objective_controller(o) == side}

    def update_sticky_control(self) -> None:
        """À appeler quand un contrôle est constaté (début/fin de tour, fin de phase de commandement) :
        un objectif pris par le Level of Control chasse le contrôle collant adverse."""
        for o in self.layout.objectives:
            who = controller(self.levels_of_control(o))
            if who is not None and self.sticky_control.get(o.id) not in (None, who):
                del self.sticky_control[o.id]

    def apply_objective_secured(self, side: str) -> None:
        """Fin de la phase de commandement : les unités avec Objective Secured rendent collants
        les objectifs qu'elles contrôlent et dont elles sont à portée."""
        for u in self.units_of(side):
            if not u.has_ability("Objective Secured"):
                continue
            for o in self.layout.objectives:
                if self.objective_controller(o) == side and self.unit_in_objective_range(u, o):
                    self.sticky_control[o.id] = side

    # ----------------------------------------------------------- divers

    def d6(self) -> int:
        return self.rng.randint(1, 6)

    def roll(self, n: int) -> int:
        return sum(self.d6() for _ in range(n))

    def say(self, msg: str) -> None:
        if self.recording:
            self.log.append(msg)

    def clone(self, record: Optional[bool] = None, seed: Optional[int] = None) -> "GameState":
        """Copie rapide et indépendante de l'état (pour la recherche arborescente).

        ``record`` : garder le journal (None = comme l'original ; False = journal vide et plus rien
        n'est enregistré, le plus rapide). ``seed`` : ré-ensemence les dés de la copie (échantillonner
        un autre futur) ; None = mêmes dés que l'original (la copie rejoue exactement le même futur)."""
        new = copy.copy(self)
        new.units = {uid: u.clone() for uid, u in self.units.items()}
        new.rng = random.Random()
        if seed is None:
            new.rng.setstate(self.rng.getstate())
        else:
            new.rng.seed(seed)
        new.scoreboard = Scoreboard(round_cap=self.scoreboard.round_cap, events=list(self.scoreboard.events))
        new.sticky_control = dict(self.sticky_control)
        new.controlled_at_turn_start = set(self.controlled_at_turn_start)
        new.charge_targets_this_phase = set(self.charge_targets_this_phase)
        new.cp = dict(self.cp)
        new.flow = self.flow.clone() if self.flow is not None else None
        keep = self.recording if record is None else record
        new.recording = keep
        new.log = list(self.log) if keep else []
        new.events = list(self.events) if keep else []
        new.history = list(self.history)
        return new

    def copy(self) -> "GameState":
        return self.clone()

    def summary(self) -> str:
        parts = []
        for side in SIDES:
            units = ", ".join(u.status for u in self.units_of(side, alive_only=False))
            parts.append(f"{side}: {self.scoreboard.total(side)} VP — {units}")
        return " | ".join(parts)
