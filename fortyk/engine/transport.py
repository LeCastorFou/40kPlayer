"""Transports et Scouts : capacité lue dans la fiche, embarquer, débarquer, transport détruit, scouts.

Règles V11 (PDF des règles de base, sections 18 et 24.31-24.32) :

* **capacité** (18.01) : lue dans le texte « transport capacity » de la fiche (nombre, mots-clés requis,
  exclusions « excluding … » / « It cannot transport … », places multiples « takes up the space of
  N models », « It can only transport … »). Les clauses spéciales (« can instead transport 1
  Dreadnought », véhicules embarqués) ne sont pas lues : elles restent dans :attr:`Capacity.unparsed` ;
  une unité peut commencer la bataille à bord (Declare Battle Formations) ;
* **embarquer** (18.02) : après un mouvement normal, Advance ou Fall Back, si chaque figurine est à 3"
  du transport et que l'unité n'a pas été posée sur la table ce tour (débarquement) ;
* **débarquer** (18.03-18.04, phase de mouvement, si l'unité était à bord et que le transport n'a fait
  ni Advance ni Fall Back) : *Rapid* (le transport a bougé : 3", pas de charge ce tour), *Tactical*
  (le transport n'a pas encore bougé et l'unité tient à 3" hors portée d'engagement : l'unité fait
  ensuite un mouvement normal ou Advance), sinon *Combat* (6", un jet de danger par figurine, les
  figurines peuvent finir engagées avec les unités qu'engage le transport ; l'unité est battle-shocked
  et ne charge pas ce tour) ;
* **transport détruit** (18.05, emergency disembark) : un jet de danger par figurine, puis chaque
  figurine est posée entièrement à 6" du transport, aussi près que possible (sinon détruite) ;
  l'unité est battle-shocked et ne charge pas ce tour ;
* **Scouts X"** (24.31-24.32) : avant le round 1, une unité dont toutes les figurines ont Scouts et qui
  est entièrement dans sa zone fait un mouvement normal d'au plus X", fini à plus de 8" de toute
  figurine ennemie ; un DEDICATED TRANSPORT entièrement dans sa zone fait ce mouvement si toutes les
  figurines à bord ont Scouts (X = le plus petit des passagers).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fastgeo import deployment_grid_legal, point_box_distances, shape_arrays, touches_any
from .geometry import Disk, Point, disk_gap, disks_overlap, extreme_points, point_core_distance, point_in_polygon, within
from .movement import FormationMove, MoveKind, legal_translations, move_distance, pose_disk, unit_vector
from .state import GameState, Model, Unit

__all__ = [
    "Capacity",
    "capacity_of",
    "is_transport",
    "space_needed",
    "space_used",
    "embark_error",
    "embark_options",
    "disembark_error",
    "disembark_candidates",
    "check_disembark_positions",
    "emergency_positions",
    "disembark_mode",
    "ring_placements",
    "place_formation",
    "scout_distance",
    "scout_moves",
    "check_scout_positions",
]

_TYPE_WORDS = {"infantry", "mounted", "beasts", "swarm", "vehicle", "monster", "walker"}


def _norm(text: str) -> str:
    """Minuscules, apostrophes droites, pluriels simples retirés (« Flawless Blades » = « Flawless Blade »)."""
    t = text.replace("’", "'").replace("‘", "'").lower()
    t = re.sub(r"[^a-z0-9' -]+", " ", t)
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in t.split()]
    return " ".join(words)


def _split_list(text: str) -> List[str]:
    """« A, B and C » / « A, B or C » → [A, B, C]."""
    parts = re.split(r",\s*|\s+and\s+|\s+or\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass(frozen=True)
class Capacity:
    """Capacité de transport lue dans la fiche."""

    capacity: int
    kinds: Tuple[Tuple[str, str], ...]  #: alternatives (mot-clé requis, type requis) — « Emperor's Children » + « Infantry »
    excluded: Tuple[str, ...]  #: mots-clés ou noms qui ne peuvent pas monter
    costs: Tuple[Tuple[str, int], ...]  #: (mot-clé ou nom, places par figurine)
    only: Tuple[str, ...]  #: « It can only transport … » : noms d'unités
    text: str
    unparsed: str = ""  #: clauses non lues (dreadnought à la place, véhicules embarqués…)


_CAP_RE = re.compile(r"transport capacity of (\d+)\s*(.*?)\s*models?\b(.*)", re.I | re.S)
_EXCL_PAREN_RE = re.compile(r"\(excluding ([^)]*?) models?\)", re.I)
_CANNOT_RE = re.compile(r"It cannot transport ([^.(]*?) models?\b", re.I)
_COST_RE = re.compile(r"(?:Each|each) ([^.]*?) models? takes? up the space of (\d+) models?", re.I)
_ONLY_RE = re.compile(r"It can only transport ([^.]*?)(?: models?)?\.", re.I)

_CACHE: Dict[str, Optional[Capacity]] = {}


def capacity_of(datasheet) -> Optional[Capacity]:
    """Capacité de transport de la fiche (None si ce n'est pas un transport)."""
    key = getattr(datasheet, "id", None) or datasheet.name
    if key in _CACHE:
        return _CACHE[key]
    text = (datasheet.transport or "").strip()
    cap = _parse_capacity(text) if text else None
    _CACHE[key] = cap
    return cap


def _parse_capacity(text: str) -> Optional[Capacity]:
    m = _CAP_RE.search(text)
    if m is None:
        return None
    n = int(m.group(1))
    desc = m.group(2).strip()
    rest = m.group(3)
    kinds: List[Tuple[str, str]] = []
    alts = [a.strip() for a in re.split(r"\s+or\s+", desc) if a.strip()] if desc else []
    parsed: List[Tuple[str, Optional[str]]] = []
    for alt in alts:
        words = alt.split()
        if words and words[-1].lower() in _TYPE_WORDS:
            parsed.append((" ".join(words[:-1]), words[-1]))
        else:
            parsed.append((alt, None))
    # « Tacticus or Phobos Infantry » : une alternative sans type hérite du type suivant
    for i, (kw, typ) in enumerate(parsed):
        if typ is None:
            typ = next((t for _, t in parsed[i + 1:] if t), "")
        kinds.append((kw, typ or ""))
    excluded: List[str] = []
    for em in _EXCL_PAREN_RE.finditer(text):
        excluded += _split_list(em.group(1))
    for cm in _CANNOT_RE.finditer(text):
        excluded += _split_list(cm.group(1))
    costs = [(name, int(cm.group(2))) for cm in _COST_RE.finditer(text) for name in _split_list(cm.group(1))]
    only: List[str] = []
    om = _ONLY_RE.search(text)
    if om:
        only = _split_list(om.group(1))
    unparsed = []
    if re.search(r"can (?:instead|also) transport|and 1 \w+ model", text, re.I):
        unparsed.append("capacité spéciale (véhicule / dreadnought) non lue")
    if re.search(r"excluding TACTICUS CHARACTER", text, re.I):
        unparsed.append("exception TACTICUS CHARACTER non lue")
    # l'exception « (excluding TACTICUS CHARACTER models …) » n'est pas une exclusion supplémentaire
    excluded = [e for e in excluded if not _norm(e).startswith("tacticus character")]
    return Capacity(n, tuple(kinds), tuple(dict.fromkeys(excluded)), tuple(costs), tuple(only), text, "; ".join(unparsed))


def is_transport(unit: Unit) -> bool:
    return capacity_of(unit.datasheet) is not None


# ------------------------------------------------------------------ qui peut monter, combien de places


def _datasheet_of(unit: Unit, model: Model):
    if model.is_leader:
        for ds in unit.leaders:
            if any(model.profile is p for p in ds.models):
                return ds
    return unit.datasheet


def _matches(ds, model: Model, term: str) -> bool:
    t = _norm(term)
    if not t:
        return False
    if any(_norm(k.name) == t for k in ds.keyword_entries):
        return True
    return _norm(ds.name) == t or _norm(model.name) == t


def _unit_has(unit: Unit, term: str) -> bool:
    t = _norm(term)
    for ds in (unit.datasheet,) + tuple(unit.leaders):
        if any(_norm(k.name) == t for k in ds.keyword_entries):
            return True
    return False


def space_needed(unit: Unit, cap: Capacity) -> int:
    total = 0
    for m in unit.alive_models:
        ds = _datasheet_of(unit, m)
        cost = 1
        for name, k in cap.costs:
            if _matches(ds, m, name):
                cost = max(cost, k)
        total += cost
    return total


def space_used(state: GameState, transport: Unit) -> int:
    cap = capacity_of(transport.datasheet)
    if cap is None:
        return 0
    return sum(space_needed(p, cap) for p in state.passengers(transport.id))


def _kind_error(unit: Unit, cap: Capacity) -> Optional[str]:
    if cap.only:
        names = {_norm(n) for n in cap.only}
        if _norm(unit.datasheet.name) not in names:
            return f"ce transport ne prend que {', '.join(cap.only)}"
    if cap.kinds:
        for m in unit.alive_models:
            ds = _datasheet_of(unit, m)
            ok = False
            for kw, typ in cap.kinds:
                if (not typ or _matches(ds, m, typ)) and (not kw or _matches(ds, m, kw)):
                    ok = True
                    break
            if not ok:
                wanted = " ou ".join(f"{kw} {typ}".strip() for kw, typ in cap.kinds)
                return f"{m.name} n'est pas {wanted}"
    for m in unit.alive_models:
        ds = _datasheet_of(unit, m)
        for ex in cap.excluded:
            if _matches(ds, m, ex):
                return f"{m.name} ({ex}) ne peut pas monter dans ce transport"
    return None


def embark_error(state: GameState, unit: Unit, transport: Unit) -> Optional[str]:
    """Pourquoi ``unit`` ne peut pas monter dans ``transport`` (capacité, mots-clés) ; None si elle peut.
    Ne vérifie pas les distances (voir :func:`embark_options`)."""
    if transport.is_destroyed or transport.side != unit.side or transport.id == unit.id:
        return "transport indisponible"
    if transport.embarked_in is not None:
        return "transport lui-même embarqué"
    cap = capacity_of(transport.datasheet)
    if cap is None:
        return f"{transport.name} n'est pas un transport"
    if capacity_of(unit.datasheet) is not None:
        return "un transport ne monte pas dans un autre transport"
    err = _kind_error(unit, cap)
    if err:
        return err
    need, used = space_needed(unit, cap), space_used(state, transport)
    if used + need > cap.capacity:
        return f"plus assez de place ({used} + {need} > {cap.capacity})"
    return None


def embark_options(state: GameState, unit: Unit, deployed: Optional[Sequence[str]] = None, check_range: bool = True) -> List[Unit]:
    """Transports amis où ``unit`` peut monter maintenant : au déploiement (``deployed`` = unités déjà
    posées, pas de distance), ou après un mouvement (toutes les figurines à 3" du transport)."""
    out = []
    for t in state.units_of(unit.side):
        if t.id == unit.id or capacity_of(t.datasheet) is None:
            continue
        if deployed is not None and t.id not in deployed:
            continue
        if embark_error(state, unit, t) is not None:
            continue
        if check_range:
            tdisks = t.disks()
            if not all(any(within(m.disk, d, state.rules.embark_range_in) for d in tdisks) for m in unit.alive_models):
                continue
        out.append(t)
    return out


# ------------------------------------------------------------------ débarquer


def disembark_error(state: GameState, unit: Unit) -> Optional[str]:
    """Pourquoi l'unité embarquée ne peut pas débarquer maintenant (phase de mouvement) ; None sinon."""
    if unit.embarked_in is None:
        return "pas embarquée"
    t = state.units.get(unit.embarked_in)
    if t is None or t.is_destroyed:
        return "transport détruit"
    if t.advanced:
        return f"{t.name} a fait une Advance"
    if t.fell_back:
        return f"{t.name} s'est replié"
    return None


def disembark_mode(state: GameState, unit: Unit) -> Tuple[str, float, Tuple[str, ...]]:
    """Mode de débarquement V11 (18.04) : (mode, distance de pose, unités ennemies qu'on peut engager).

    * ``rapid`` : le transport a fait un mouvement normal cette phase → 3", pas de charge ce tour ;
    * ``tactical`` : il n'a pas bougé (immobile ou pas encore activé) et l'unité tient à 3" → 3", puis
      mouvement normal ou Advance ;
    * ``combat`` : sinon → 6", un jet de danger par figurine, pose possible au contact des ennemis
      engagés avec le transport, puis battle-shock et pas de charge ce tour."""
    rules = state.rules
    t = state.units[unit.embarked_in]
    if not t.remained_stationary:
        return "rapid", rules.disembark_range_in, ()
    if ring_placements(state, unit, t, rules.disembark_range_in, starts=4) or disembark_candidates(state, unit, t, rules.disembark_range_in, step=0.25, limit=1):
        return "tactical", rules.disembark_range_in, ()
    engaged_with = tuple(e.id for e in state.enemies_in_engagement_range(t))
    return "combat", rules.combat_disembark_range_in, engaged_with


def _wholly_within_of(d: Disk, tdisks: Sequence[Disk], dist: float) -> bool:
    """Toute l'empreinte ``d`` est à ``dist`` ou moins d'une figurine du transport."""
    for t in tdisks:
        # point le plus éloigné de l'empreinte : on majore par le centre + l'extension de l'empreinte
        far = point_core_distance(d.center, t) - t.r + d.extent
        if far <= dist + 1e-6:
            return True
        if not d.is_round and all(point_core_distance(p, t) - t.r <= dist + 1e-6 for p in extreme_points(d)):
            return True
    return False


def _formation(unit: Unit) -> Tuple[np.ndarray, np.ndarray]:
    ms = unit.alive_models
    cx = sum(m.x for m in ms) / len(ms)
    cy = sum(m.y for m in ms) / len(ms)
    offsets = np.array([(m.x - cx, m.y - cy) for m in ms], dtype=float)
    shapes = shape_arrays([m.disk for m in ms])
    shapes[:, 0] = offsets[:, 0]
    shapes[:, 1] = offsets[:, 1]
    return offsets, shapes


def disembark_candidates(state: GameState, unit: Unit, transport: Unit, dist: Optional[float] = None, step: float = 0.5,
                         limit: int = 40, blockers_extra: Sequence[Disk] = (), allowed_engaged: Sequence[str] = ()) -> List[Point]:
    """Centres de formation légaux pour débarquer : chaque figurine entièrement à ``dist`` (3") du
    transport, sur la table, sans chevaucher personne, hors portée d'engagement des ennemis.
    Sous-échantillonnage régulier (déterministe) au-delà de ``limit``."""
    dist = state.rules.disembark_range_in if dist is None else dist
    tmodels = transport.models if transport.is_destroyed else transport.alive_models
    tdisks = [m.disk for m in tmodels]
    offsets, shapes = _formation(unit)
    n = len(offsets)
    reach = max(t.extent for t in tdisks) + dist
    tx = sum(t.x for t in tdisks) / len(tdisks)
    ty = sum(t.y for t in tdisks) / len(tdisks)
    xs = np.arange(tx - reach, tx + reach + 1e-9, step)
    ys = np.arange(ty - reach, ty + reach + 1e-9, step)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    centers = np.stack([gx.ravel(), gy.ravel()], axis=1)
    pos = centers[:, None, :] + offsets[None, :, :]  # (G, n, 2)
    ext = shapes[:, 2] + np.hypot(shapes[:, 3], shapes[:, 4])
    # entièrement à ``dist`` : centre + extension de l'empreinte à ``dist`` du bord du transport
    ok_within = np.zeros(pos.shape[:2], dtype=bool)
    for t in tdisks:
        d = point_box_distances(pos, tuple(t)) - t.r + ext[None, :]
        ok_within |= d <= dist + 1e-6
    ok = ok_within.all(axis=1)
    if not ok.any():
        return []
    centers = centers[ok]
    board_zone = ((0.0, 0.0), (state.layout.board[0], 0.0), state.layout.board, (0.0, state.layout.board[1]))
    others = [m.disk for u in state.on_table_units() if u.id != unit.id for m in u.alive_models] + tdisks + list(blockers_extra)
    others_s = shape_arrays(others)
    legal = deployment_grid_legal(centers, offsets, shapes[:, 2], board_zone, state.layout.board, np.zeros((0, 2)), np.zeros(0),
                                  shapes=shapes, others_shapes=others_s)
    enemies = shape_arrays(_blocking_enemies(state, unit, allowed_engaged))
    legal &= ~touches_any(centers, offsets, shapes, enemies, grow=state.rules.engagement_range_in + 1e-6)
    out = [(round(float(x), 2), round(float(y), 2)) for x, y in centers[legal]]
    if len(out) > limit:
        stride = len(out) / limit
        out = [out[int(k * stride)] for k in range(limit)]
    return out


def _blocking_enemies(state: GameState, unit: Unit, allowed_engaged: Sequence[str] = ()) -> List[Disk]:
    """Figurines ennemies dont on doit rester hors de portée d'engagement (toutes, sauf celles des
    unités de ``allowed_engaged`` : débarquement de combat au contact des ennemis du transport)."""
    return [m.disk for e in state.enemies_of(unit.side) if e.id not in allowed_engaged for m in e.alive_models]


def check_disembark_positions(state: GameState, unit: Unit, transport: Unit, positions: Dict[str, Sequence[float]],
                              dist: Optional[float] = None, allowed_engaged: Sequence[str] = ()) -> Optional[str]:
    """Valide un débarquement figurine par figurine (toutes les figurines doivent être placées) :
    entièrement à ``dist`` du transport, sur la table, sans chevauchement, hors portée d'engagement
    (sauf des unités de ``allowed_engaged``), en cohérence."""
    dist = state.rules.disembark_range_in if dist is None else dist
    alive = unit.alive_models
    missing = [m.id for m in alive if m.id not in positions]
    if missing:
        return f"figurines non placées : {', '.join(missing)}"
    finals = {m.id: pose_disk(m, positions[m.id]) for m in alive}
    tdisks = [m.disk for m in (transport.models if transport.is_destroyed else transport.alive_models)]
    others = [m.disk for u in state.on_table_units() if u.id != unit.id for m in u.alive_models]
    enemies = _blocking_enemies(state, unit, allowed_engaged)
    for mid, d in finals.items():
        if not state.on_board(d):
            return f"{mid} sort de la table"
        if not _wholly_within_of(d, tdisks, dist):
            return f"{mid} n'est pas entièrement à {dist:g}\" du transport"
        if any(disks_overlap(d, o) for o in others + tdisks):
            return f"{mid} chevauche une autre figurine"
        if any(within(d, e, state.rules.engagement_range_in) for e in enemies):
            return f"{mid} est à portée d'engagement d'un ennemi"
    ids = list(finals)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if disks_overlap(finals[a], finals[b]):
                return f"{a} et {b} se chevauchent"
    if len(alive) > 1:
        rules = state.rules
        for mid, d in finals.items():
            rest = [finals[o] for o in finals if o != mid]
            if not any(within(d, o, rules.coherency_range_in) for o in rest):
                return f"{mid} n'est à 2\" d'aucune autre figurine (cohérence)"
            if any(disk_gap(d, o) > rules.coherency_max_spread_in for o in rest):
                return f"{mid} est à plus de 9\" d'une figurine de l'unité (cohérence)"
    return None


def _ring_sequences(unit: Unit, transport: Unit, dist: float, starts: int) -> List[List[Point]]:
    """Points de pose le long du pourtour du transport, couronne après couronne (tant qu'une figurine
    y reste entièrement à ``dist``), à partir de ``starts`` points de départ différents."""
    from .geometry import perimeter_points

    tmodels = transport.models if transport.is_destroyed else transport.alive_models
    if not tmodels or not unit.alive_models:
        return []
    t = tmodels[0].disk
    r = max(m.extent for m in unit.alive_models)
    rings = []
    off = r + 0.05
    while off + r <= dist + 1e-9 and len(rings) < 6:
        ring = Disk(t.x, t.y, t.r + off, t.hx, t.hy, t.a)
        perim = 2 * math.pi * ring.r + 4 * (t.hx + t.hy)
        rings.append(perimeter_points(ring, max(1, int(perim // (2 * r + 0.1)))))
        off += 2 * r + 0.1
    if not rings:
        return []
    out = []
    for k in range(starts):
        seq: List[Point] = []
        for pts in rings:
            shift = int(round(k * len(pts) / starts))
            seq += pts[shift:] + pts[:shift]
        out.append(seq)
    return out


def _greedy_ring(state: GameState, unit: Unit, transport: Unit, seq: Sequence[Point], allowed_engaged: Sequence[str] = ()) -> Dict[str, Tuple[float, float]]:
    """Pose les figurines une à une sur les points libres de ``seq`` (dans l'ordre) ; celles qui ne
    trouvent pas de place sont absentes du résultat."""
    tmodels = transport.models if transport.is_destroyed else transport.alive_models
    others = [m.disk for u in state.on_table_units() if u.id != unit.id for m in u.alive_models] + [m.disk for m in tmodels]
    enemies = _blocking_enemies(state, unit, allowed_engaged)
    er = state.rules.engagement_range_in
    placed: Dict[str, Tuple[float, float]] = {}
    disks: List[Disk] = []
    i = 0
    for m in unit.alive_models:
        while i < len(seq):
            x, y = seq[i]
            i += 1
            d = m.disk.moved_to(x, y)
            if not state.on_board(d) or any(disks_overlap(d, o) for o in others) or any(disks_overlap(d, o) for o in disks):
                continue
            if any(within(d, e, er) for e in enemies):
                continue
            placed[m.id] = (round(x, 3), round(y, 3))
            disks.append(d)
            break
    return placed


def ring_placements(state: GameState, unit: Unit, transport: Unit, dist: Optional[float] = None, starts: int = 8,
                    allowed_engaged: Sequence[str] = ()) -> List[Dict[str, Tuple[float, float]]]:
    """Débarquements « en couronne » : les figurines sont posées une à une le long du bord du
    transport (première couronne collée à la coque, puis les suivantes tant qu'elles restent à
    ``dist``), en partant de ``starts`` points différents du pourtour. Chaque placement retourné est
    légal (vérifié par :func:`check_disembark_positions`)."""
    dist = state.rules.disembark_range_in if dist is None else dist
    out, seen = [], set()
    for seq in _ring_sequences(unit, transport, dist, starts):
        placed = _greedy_ring(state, unit, transport, seq, allowed_engaged)
        if len(placed) != len(unit.alive_models):
            continue
        key = tuple(sorted(placed.values()))
        if key in seen:
            continue
        seen.add(key)
        if check_disembark_positions(state, unit, transport, placed, dist, allowed_engaged) is None:
            out.append(placed)
    return out


def _coherent_subset(state: GameState, unit: Unit, placed: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """Plus grand groupe cohérent (2" de proche en proche, 9" au plus entre deux figurines) parmi les
    figurines posées ; les autres ne peuvent pas être posées."""
    rules = state.rules
    by_id = {m.id: m for m in unit.alive_models}
    disks = {mid: by_id[mid].disk.moved_to(*xy) for mid, xy in placed.items()}
    best: List[str] = []
    for seed in disks:
        group, todo = [seed], [seed]
        while todo:
            a = todo.pop()
            for b in disks:
                if b not in group and within(disks[a], disks[b], rules.coherency_range_in):
                    group.append(b)
                    todo.append(b)
        # 9" : on retire la figurine la plus éloignée tant que l'écart maximal est trop grand
        while len(group) > 1:
            worst = max(group, key=lambda g: max(disk_gap(disks[g], disks[o]) for o in group if o != g))
            if max(disk_gap(disks[worst], disks[o]) for o in group if o != worst) <= rules.coherency_max_spread_in:
                break
            group.remove(worst)
        if len(group) > len(best):
            best = group
    return {mid: placed[mid] for mid in best}


def emergency_positions(state: GameState, unit: Unit, transport: Unit) -> Dict[str, Tuple[float, float]]:
    """Débarquement d'urgence (V11, 18.05) : chaque figurine entièrement à 6" du transport détruit et
    aussi près que possible de lui (couronnes à partir de la coque) ; celles qu'on ne peut pas poser
    ainsi sont absentes du résultat (elles sont détruites). Le meilleur des points de départ = le plus
    de figurines posées."""
    dist = state.rules.emergency_disembark_range_in
    best: Dict[str, Tuple[float, float]] = {}
    for seq in _ring_sequences(unit, transport, dist, 8):
        placed = _coherent_subset(state, unit, _greedy_ring(state, unit, transport, seq))
        if len(placed) > len(best):
            best = placed
            if len(best) == len(unit.alive_models):
                break
    return best


def place_formation(unit: Unit, x: float, y: float) -> None:
    ms = unit.alive_models
    cx = sum(m.x for m in ms) / len(ms)
    cy = sum(m.y for m in ms) / len(ms)
    for m in ms:
        m.move_to(m.x - cx + x, m.y - cy + y)


# ------------------------------------------------------------------ Scouts


def _scouts_param(ds) -> Optional[float]:
    a = ds.ability("Scouts")
    if a is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", a.parameter or a.name or "")
    return float(m.group(1)) if m else None


def _own_scouts(unit: Unit) -> Optional[float]:
    """Scouts X" si chaque figurine de l'unité (personnages compris) a la capacité."""
    values = [_scouts_param(ds) for ds in (unit.datasheet,) + tuple(unit.leaders)]
    if any(v is None for v in values):
        return None
    return min(values)


def _wholly_in_zone(state: GameState, unit: Unit) -> bool:
    zone = state.layout.deployment_zones[unit.side]
    return all(point_in_polygon(p, zone) for m in unit.alive_models for p in extreme_points(m.disk))


def scout_distance(state: GameState, unit: Unit) -> Optional[float]:
    """Distance de scout de l'unité (None si elle n'en fait pas) : ses figurines ont toutes Scouts, ou
    c'est un DEDICATED TRANSPORT dont toutes les figurines à bord ont Scouts ; entièrement dans sa zone."""
    if unit.is_destroyed or unit.embarked_in is not None or not _wholly_in_zone(state, unit):
        return None
    own = _own_scouts(unit)
    if own is not None:
        return own
    if unit.has_keyword("Dedicated Transport"):
        passengers = state.passengers(unit.id)
        values = [_own_scouts(p) for p in passengers]
        if passengers and all(v is not None for v in values):
            return min(values)
    return None


def _scout_legal(state: GameState, unit: Unit, vecs, x: float) -> np.ndarray:
    a = legal_translations(state, unit, vecs, MoveKind.NORMAL, x)
    b = legal_translations(state, unit, vecs, MoveKind.FALL_BACK, x, engagement_range=state.rules.scout_enemy_distance_in)
    return a & b


def scout_moves(state: GameState, unit: Unit, x: float, tries: int = 8) -> List[FormationMove]:
    """Mouvements de scout candidats (formation) : 8 directions, vers les objectifs, vers / loin de
    l'ennemi, à pleine et demi-distance, raccourcis jusqu'à être légaux (fin à plus de 9" de l'ennemi)."""
    c = unit.centroid
    dirs: List[Tuple[Tuple[float, float], str]] = [((math.cos(math.pi / 4 * k), math.sin(math.pi / 4 * k)), f"cap {k * 45}°") for k in range(8)]
    for o in state.layout.objectives:
        v = unit_vector(c, o.center)
        if v != (0.0, 0.0):
            dirs.append((v, f"vers {o.id}"))
    enemies = state.enemy_models(unit.side)
    if enemies:
        e = min(enemies, key=lambda m: math.hypot(m.x - c[0], m.y - c[1]))
        v = unit_vector(c, e.position)
        if v != (0.0, 0.0):
            dirs.append((v, "vers l'ennemi"))
    wanted = [(v[0] * x * f, v[1] * x * f, f"scout {label} ({x * f:g}\")") for v, label in dirs for f in (1.0, 0.5)]
    fr = 1 - np.arange(tries) / tries
    vecs = np.array([(dx * k, dy * k) for dx, dy, _ in wanted for k in fr], dtype=float)
    legal = _scout_legal(state, unit, vecs, x).reshape(len(wanted), tries)
    out, seen = [], set()
    for i, (dx, dy, label) in enumerate(wanted):
        hits = np.flatnonzero(legal[i])
        if not len(hits):
            continue
        k = fr[hits[0]]
        key = (round(dx * k, 2), round(dy * k, 2))
        if key in seen or math.hypot(*key) < 1e-6:
            continue
        seen.add(key)
        out.append(FormationMove(MoveKind.NORMAL, dx * k, dy * k, label))
    return out


def check_scout_positions(state: GameState, unit: Unit, positions: Dict[str, Sequence[float]], x: float) -> Optional[str]:
    """Mouvement de scout figurine par figurine : mouvement normal de ``x``" au plus, fin à plus de 9"
    de toute figurine ennemie (la validation du mouvement normal est faite par l'appelant)."""
    far = state.rules.scout_enemy_distance_in
    enemies = [m.disk for m in state.enemy_models(unit.side)]
    for m in unit.alive_models:
        d = pose_disk(m, positions.get(m.id))
        if any(disk_gap(d, e) <= far + 1e-9 for e in enemies):
            return f"{m.id} finit à {far:g}\" ou moins d'une figurine ennemie"
    return None
