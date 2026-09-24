"""Traduction des textes de règles Wahapedia (stratagèmes, et à terme capacités et améliorations) en
briques du moteur : effets à durée (:mod:`fortyk.engine.effects`) et effets instantanés.

Les textes sont très formulaires (« Until the end of the phase, each time a model in your unit makes
an attack, add 1 to the Hit roll. »). Le compilateur découpe l'EFFECT en phrases et reconnaît chaque
phrase par un motif ; une phrase reconnue devient une ou plusieurs briques, une phrase inconnue reste
à appliquer à la main (effet manuel). Statut d'une règle :

* ``auto`` — toutes les phrases sont traduites ; le moteur l'applique seul ;
* ``partial`` — une partie seulement (le reste est rappelé au joueur, à faire à la main) ;
* ``manual`` — rien de traduit : le joueur applique l'effet avec les outils manuels.

Le moment (WHEN) est traduit en :class:`Timing` — tour (le sien, l'adverse, les deux), phases,
et moment précis (début / fin de phase, juste après que l'ennemi a choisi ses cibles…) — pour ouvrir
les fenêtres de réaction au bon moment.

    python3 scripts/rules_coverage.py          # couverture par faction
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

__all__ = ["EffectSpec", "Instant", "Timing", "Compiled", "compile_stratagem", "compile_text", "parse_timing", "PHASES"]

PHASES = ("command", "movement", "shooting", "charge", "fight")
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


@dataclass(frozen=True)
class EffectSpec:
    """Un effet à poser : ``on`` = unit (l'unité ciblée par le stratagème), target (l'unité ennemie
    choisie), vs_target (effet de l'unité seulement contre l'unité ennemie choisie)."""

    kind: str
    value: object = None
    until: str = "phase"
    scope: str = "all"
    on: str = "unit"
    cond: Optional[str] = None  #: condition (voir fortyk.engine.effects.CONDITIONS)


@dataclass(frozen=True)
class Instant:
    """Effet immédiat : reserve, mortal (dés + seuil + BM ou expression), heal, battleshock_test,
    move_roll (distance à lancer, déplacement à la main), secure (l'unité rend « collants » les
    objectifs qu'elle contrôle)."""

    kind: str
    value: object = None
    on: str = "unit"


@dataclass(frozen=True)
class Timing:
    turn: str = "either"  #: own | opponent | either
    phases: Tuple[str, ...] = PHASES
    moment: str = "any"  #: any | start | end | targets_selected | selected | after_attacks | after_move | after_charge | charge_declared | other


@dataclass
class Compiled:
    status: str  #: auto | partial | manual
    effects: List[EffectSpec] = field(default_factory=list)
    instants: List[Instant] = field(default_factory=list)
    timing: Timing = field(default_factory=Timing)
    needs_enemy: bool = False  #: l'effet vise une unité ennemie à choisir (« Select one enemy unit »)
    choices: Tuple[str, ...] = ()  #: « Select either the [LETHAL HITS] or [SUSTAINED HITS 1] ability » : valeur de « $choice »
    target_keywords: Tuple[str, ...] = ()
    unparsed: List[str] = field(default_factory=list)

    @property
    def auto(self) -> bool:
        return self.status == "auto"

    def summary(self) -> str:
        from .engine.effects import KINDS

        bits = []
        for e in self.effects:
            v = f" {e.value}" if e.value not in (None, True) else ""
            bits.append(f"{KINDS.get(e.kind, e.kind)}{v}" + ({"ranged": " (tir)", "melee": " (mêlée)"}.get(e.scope, "")))
        for i in self.instants:
            bits.append({"reserve": "retour en réserve", "mortal": "blessures mortelles", "heal": "soins",
                         "battleshock_test": "test de battle-shock", "move_roll": "distance de mouvement"}.get(i.kind, i.kind))
        return ", ".join(bits)


# ------------------------------------------------------------------ moment (WHEN)

_PHASE_WORDS = {"command": "command", "movement": "movement", "shooting": "shooting", "charge": "charge", "fight": "fight"}


def parse_timing(when: str, turn_field: str = "", phase_field: str = "") -> Timing:
    w = (when or "").lower()
    t = (turn_field or "").lower()
    if t.startswith("your"):
        turn = "own"
    elif t.startswith("opponent"):
        turn = "opponent"
    else:
        turn = "either"
    # « Your opponent's Shooting phase or the Fight phase » : le tour vient du champ Wahapedia
    phases = tuple(p for p in PHASES if re.search(rf"\b{p}\b phase", w)) or tuple(p for p in PHASES if p in (phase_field or "").lower())
    if "any phase" in w or not phases:
        phases = PHASES
    if re.search(r"selected (its|their) targets|targets? (a|one or more) friendly|when an enemy unit targets", w):
        moment = "targets_selected"
    elif re.search(r"\b(start|beginning) of\b", w):
        moment = "start"
    elif re.search(r"\bend of\b", w):
        moment = "end"
    elif re.search(r"(has|have) (shot|fought|resolved its attacks)|finished making its attacks", w):
        moment = "after_attacks"
    elif re.search(r"is selected to (shoot|fight|attack)|selected to shoot or fight", w):
        moment = "selected"
    elif re.search(r"ends? a charge move|ended a charge move", w):
        moment = "after_charge"
    elif re.search(r"declares? a charge|declared a charge", w):
        moment = "charge_declared"
    elif re.search(r"ends? (a|its) (normal|advance|fall.back|move)|ends a move|falls? back|advances?\b", w):
        moment = "after_move"
    elif re.fullmatch(r"\s*(your |the |your opponent’s |your opponent's )?(\w+ )(or (the |your )?\w+ )?phase\.?\s*", w) or re.fullmatch(r"\s*any phase\.?\s*", w):
        moment = "any"
    else:
        moment = "other"
    return Timing(turn, phases, moment)


# ------------------------------------------------------------------ effets (EFFECT)

_DUR = [
    (re.compile(r"^until the end of the (phase|turn|battle|battle round)\b,?\s*", re.I), None),
    (re.compile(r"^until the attacking unit has finished making its attacks,?\s*", re.I), "phase"),
    (re.compile(r"^until the start of your next (command phase|turn)\b,?\s*", re.I), "owner_next_turn"),
    (re.compile(r"^until the end of your opponent’s next turn,?\s*|^until the end of your opponent's next turn,?\s*", re.I), "owner_next_turn"),
]


def _duration(sentence: str) -> Tuple[str, str]:
    s = sentence.strip()
    for rx, fixed in _DUR:
        m = rx.match(s)
        if m:
            until = fixed or {"phase": "phase", "turn": "turn", "battle": "battle", "battle round": "round"}[m.group(1).lower()]
            return until, s[m.end():]
    m = re.search(r",?\s*until the end of the (phase|turn|battle)\.?$", s, re.I)
    if m:
        return {"phase": "phase", "turn": "turn", "battle": "battle"}[m.group(1).lower()], s[:m.start()]
    return "phase", s


def _num(tok: str) -> Optional[int]:
    tok = tok.lower().strip()
    if tok.isdigit():
        return int(tok)
    return _WORD_NUM.get(tok)


def _scope(text: str) -> str:
    t = text.lower()
    if re.search(r"\branged (weapons?|attacks?)\b|makes? a ranged attack|shooting attacks", t):
        return "ranged"
    if re.search(r"\bmelee (weapons?|attacks?)\b|makes? a melee attack", t):
        return "melee"
    return "all"


_ATTACK_SUBJECT = re.compile(r"(each time (a|an \w+) model in your unit makes an? (ranged |melee )?attack|weapons equipped by models in your unit|"
                             r"your unit’s (ranged |melee )?attacks|your unit's (ranged |melee )?attacks|models in your unit|your unit)", re.I)
_DEFENSE_SUBJECT = re.compile(r"(each time an? (ranged |melee )?attack targets your unit|attacks that target your unit|each time an attack is allocated to a model in your unit|"
                              r"ranged attacks that target your unit)", re.I)
_CONDITION = re.compile(r"\bif\b|\bthat targets? an? (enemy )?unit (that|within)|\bwithin engagement range of one or more\b|"
                        r"\bthat is (below|at) half|\bthat (is|are) (within|wholly)|\bthat targets? a (battle-shocked|character|monster|vehicle)", re.I)
_KW_RE = re.compile(r"\[(\$?[A-Za-z][A-Za-z0-9 \-+]+)\]")

#: phrases sans effet mécanique dans ce moteur (décors franchissables, pas d'actions…) : reconnues
_NOOP = re.compile(r"^(designer’s note|designer's note)|can move horizontally through terrain features|can move through terrain features|"
                   r"does not prevent your unit from being eligible to (shoot|declare a charge|start an action|shoot/declare a charge)|"
                   r"^that move does not prevent|vertical distance", re.I)


def _condition(low: str) -> Tuple[Optional[str], Optional[str]]:
    """(condition reconnue, texte restant) ; (None, None) si la phrase a une condition inconnue."""
    m = re.search(r"that targets? an? ((?:[a-z’' -]+?)(?: or [a-z’' -]+?)*) unit\b", low)
    cond = None
    if m and not re.search(r"that (is|are|was|has|had) ", m.group(1)):
        words = [w.strip() for w in re.split(r"\bor\b|,", m.group(1)) if w.strip()]
        if words and all(re.fullmatch(r"[a-z’' -]+", w) for w in words) and not any(w in ("enemy", "friendly") for w in words):
            cond = "kw:" + "|".join(w.title() for w in words)
            low = low[:m.start()] + "that targets that unit" + low[m.end():]
    if re.search(r"that targets? a unit that is battle-shocked|that targets? a battle-shocked unit", low):
        cond, low = "battleshocked", re.sub(r"that targets? (a unit that is battle-shocked|a battle-shocked unit)", "that targets that unit", low)
    elif re.search(r"that targets? a unit (that is )?below half-strength|that targets? a below half-strength unit", low):
        cond, low = "below_half", re.sub(r"that targets? (a unit (that is )?below half-strength|a below half-strength unit)", "that targets that unit", low)
    elif re.search(r"that targets? (a|an enemy) unit (that is )?within range of (an|one or more) objective markers?", low):
        cond, low = "objective", re.sub(r"that targets? (a|an enemy) unit (that is )?within range of (an|one or more) objective markers?", "that targets that unit", low)
    m = re.search(r"if the strength characteristic of that attack is greater than (your|this) unit’s toughness characteristic,?\s*|if the strength characteristic of that attack is greater than (your|this) unit's toughness characteristic,?\s*", low)
    if m:
        cond, low = "s_gt_t", low[:m.start()] + low[m.end():]
    m = re.match(r"if your unit remained stationary this turn,?\s*(then\s*)?", low)
    if m:
        cond, low = "self:stationary", low[m.end():]
    m = re.match(r"if your unit made a charge move this turn,?\s*(then\s*)?", low)
    if m:
        cond, low = "self:charged", low[m.end():]
    return cond, low


def _weapon_abilities(text: str) -> List[str]:
    out = []
    for kw in _KW_RE.findall(text):
        k = kw.strip().lower()
        if k == "$choice":
            out.append(k)
            continue
        if k.startswith(("lethal hits", "sustained hits", "devastating wounds", "precision", "ignores cover", "lance", "assault", "heavy",
                         "twin-linked", "anti-", "pistol", "blast", "torrent", "hazardous", "melta", "rapid fire")):
            out.append(k)
    return out


def _sentence(sentence: str, ctx: Dict) -> Optional[Tuple[List[EffectSpec], List[Instant]]]:
    """Traduit une phrase ; None si elle n'est pas reconnue."""
    if _NOOP.search(sentence):
        ctx["noop"] = True
        return [], []
    low0 = sentence.strip().lower()
    m = re.match(r"select (either )?the (\[[^\]]+\])( ability)? or the (\[[^\]]+\])( abilities| ability)?", low0) \
        or re.match(r"select (either )?the (\[[^\]]+\]) or (\[[^\]]+\]) abilit", low0)
    if m:
        ctx["choices"] = tuple(k.lower() for k in _KW_RE.findall(sentence))
        return [], []
    cond_pre, rest = _condition(low0)
    if cond_pre in ("self:stationary", "self:charged"):  # « If your unit Remained Stationary this turn, then until… »
        sentence = sentence.strip()[len(sentence.strip()) - len(rest):]
    until, body = _duration(sentence)
    b = body.strip().rstrip(".").strip()
    cond, low = _condition(b.lower())
    cond = cond or cond_pre
    b_for_kw = b
    if "selected ability" in low and ctx.get("choices"):
        b_for_kw = b + " [$CHOICE]"
    effects: List[EffectSpec] = []
    instants: List[Instant] = []
    vs_target = bool(re.search(r"that targets? (that|the selected|those) enemy units?", low))

    if not b:
        return [], []
    if re.match(r"select (one|up to \w+) (visible )?enemy units?", low) or re.match(r"select one enemy unit", low):
        ctx["needs_enemy"] = True
        return [], []
    if re.match(r"(remove your unit from the battlefield and )?place (it|your unit|them) (in|into) strategic reserves", low) \
            or re.match(r"remove your unit from the battlefield and place it into strategic reserves", low) \
            or re.fullmatch(r"remove your unit from the battlefield", low):
        return [], [Instant("reserve")]
    if re.search(r"set (your unit|it) up anywhere on the battlefield that is more than (8|9)\" horizontally away from all enemy (models|units)", low) \
            or re.fullmatch(r"your unit can be set up anywhere on the battlefield that is more than (8|9)\" horizontally away from all enemy (models|units)", low):
        return [EffectSpec("deep_strike", True, "battle")], []
    if re.match(r"select one objective (marker )?(your unit is controlling|you control that your unit is within range of)", low) or re.fullmatch(r"that objective (marker )?is secured", low):
        return [], [Instant("secure")]
    m = re.search(r"if your unit advances, do not make an advance roll( for it)?", low)
    if m:
        ctx["advance_fixed"] = True
        return [], []
    if low0.startswith("instead"):
        m = re.search(r"add (\d)\" to the move characteristic of models in your unit", low0)
        if m and ctx.get("advance_fixed"):
            return [EffectSpec("advance_fixed", int(m.group(1)), "phase")], []
        return None  # « Instead, … » : dépend d'une phrase précédente non traduite
    m = re.fullmatch(r"your unit[’']s (ranged |melee )?attacks have \+(\d) (a|attacks|s|strength|ap|d|damage)", low)
    if m:
        kind = {"a": "attacks_mod", "attacks": "attacks_mod", "s": "strength_mod", "strength": "strength_mod", "ap": "ap_mod", "d": "damage_mod", "damage": "damage_mod"}[m.group(3)]
        return [EffectSpec(kind, int(m.group(2)), until, _scope(b), cond=cond)], []
    m = re.fullmatch(r"your unit has \+(\d) sv", low)
    if m:
        return [EffectSpec("save_mod", int(m.group(1)), until)], []
    m = re.fullmatch(r"re-roll (hit|wound) rolls( of 1)?", low)
    if m:
        return [EffectSpec("reroll_hits" if m.group(1) == "hit" else "reroll_wounds", "ones" if m.group(2) else "fails", until, cond=cond)], []
    # blessures mortelles
    m = re.match(r"roll (one|two|three|four|five|six|\d+) d6(?: for each model in your unit)?:? for each (\d)\+, (that|the selected) enemy unit suffers (1|one) mortal wound", low)
    if m:
        ctx["needs_enemy"] = True
        return [], [Instant("mortal", ("dice", _num(m.group(1)), int(m.group(2)), 1), on="target")]
    m = re.match(r"roll one d6:? on a (\d)\+, (that|the selected) enemy unit suffers (d3|d6|\d+|one) mortal wounds?", low)
    if m:
        ctx["needs_enemy"] = True
        return [], [Instant("mortal", ("check", int(m.group(1)), m.group(3)), on="target")]
    m = re.match(r"(that|the selected) enemy unit suffers (d3|d6|\d+|one) mortal wounds?", low)
    if m:
        ctx["needs_enemy"] = True
        return [], [Instant("mortal", ("fixed", m.group(2)), on="target")]
    m = re.match(r"(one model in )?your unit (heals|regains) (up to )?(d3|d6|\d+|one) (lost )?wounds?", low) or re.match(r"each selected unit heals (\d+) wounds", low)
    if m and "each selected" not in low:
        return [], [Instant("heal", m.group(4))]
    if re.match(r"(that|the selected) enemy unit must take a battle-shock test", low):
        ctx["needs_enemy"] = True
        return [], [Instant("battleshock_test", on="target")]
    m = re.match(r"your unit can (make|immediately make) an? (normal|surge) move of up to (d3\+\d|d6\+\d|d6|d3|\d+)\"", low)
    if m:
        return [], [Instant("move_roll", m.group(3))]
    if _CONDITION.search(low):
        return None  # condition que le moteur ne sait pas évaluer : à la main
    # --- défense (attaques qui ciblent l'unité)
    if _DEFENSE_SUBJECT.search(b):
        scope = _scope(b)
        if re.search(r"with a s greater than your unit[’']s t", low):
            cond = "s_gt_t"
        m = re.search(r"subtract (\d|one) from the (hit|wound) roll", low)
        if m:
            effects.append(EffectSpec("hit_mod_against" if m.group(2) == "hit" else "wound_mod_against", -_num(m.group(1)), until, scope, cond=cond))
        m = re.search(r"have -(\d) to wound rolls", low)
        if m:
            effects.append(EffectSpec("wound_mod_against", -int(m.group(1)), until, scope, cond=cond))
        m = re.search(r"worsen the armour penetration characteristic of that attack by (\d|one)|have -(\d) ap\b", low)
        if m:
            effects.append(EffectSpec("ap_worsen", _num(m.group(1) or m.group(2)), until, scope))
        m = re.search(r"subtract (\d|one) from the damage characteristic", low)
        if m:
            effects.append(EffectSpec("damage_reduction", _num(m.group(1)), until, scope))
        m = re.search(r"add (\d|one) to (the|any) (armour )?saving throw", low)
        if m:
            effects.append(EffectSpec("save_mod", _num(m.group(1)), until, scope))
        return (effects, instants) if effects else None
    # --- capacités de l'unité
    m = re.search(r"(?:have|has) the feel no pain (\d)\+ ability(?! against)", low) or re.search(r"(?:have|has) feel no pain (\d)\+(?! against)", low)
    if m and "against" not in low:
        return [EffectSpec("fnp", int(m.group(1)), until)], []
    m = re.search(r"(?:have|has) the feel no pain (\d)\+ ability against mortal wounds|(?:have|has) feel no pain (\d)\+ against mortal wounds", low)
    if m:
        return [EffectSpec("fnp_mortal", int(m.group(1) or m.group(2)), until)], []
    m = re.search(r"add (\d|one) to the toughness characteristic of models in your unit", low)
    if m:
        return [EffectSpec("toughness_mod", _num(m.group(1)), until)], []
    m = re.search(r"(?:have|has) an? (\d)\+ invulnerable save", low)
    if m:
        return [EffectSpec("invuln", int(m.group(1)), until)], []
    if re.search(r"(?:have|has) the stealth ability|your unit has stealth", low):
        return [EffectSpec("hit_mod_against", -1, until, "ranged")], []
    if re.search(r"(?:have|has) the benefit of cover", low):
        return [EffectSpec("cover", True, until, "ranged")], []
    if re.search(r"(?:have|has) the lone operative ability", low):
        return [EffectSpec("lone_operative", True, until)], []
    if re.search(r"(?:have|has) the fights first ability|has fights first", low):
        return [EffectSpec("fights_first", True, until)], []
    m = re.search(r"eligible to (shoot|declare a charge|shoot and declare a charge) in a turn in which it (fell back|advanced)", low)
    if m and len(re.findall(r"in a turn in which", low)) == 1:
        what, when = m.group(1), m.group(2)
        pre = "fall_back" if when == "fell back" else "advance"
        if "shoot" in what:
            effects.append(EffectSpec(f"{pre}_and_shoot", True, until))
        if "charge" in what:
            effects.append(EffectSpec(f"{pre}_and_charge", True, until))
        return effects, []
    m = re.search(r"add (\d)\" to the move characteristic", low)
    if m:
        return [EffectSpec("move_mod", int(m.group(1)), until)], []
    m = re.search(r"(?:has|have|add) \+?(\d) to (advance and charge|charge|advance) rolls", low) or re.search(r"add (\d) to (advance and charge|charge|advance) rolls", low)
    if m:
        n = int(m.group(1))
        if "advance" in m.group(2):
            effects.append(EffectSpec("advance_mod", n, until))
        if "charge" in m.group(2):
            effects.append(EffectSpec("charge_mod", n, until))
        return effects, []
    m = re.search(r"(?:improve|add (\d) to) the objective control characteristic|oc characteristic of models in your unit by (\d)", low)
    if m and re.search(r"by (\d)|add (\d)", low):
        n = int(re.search(r"by (\d)|add (\d)", low).group(1) or re.search(r"by (\d)|add (\d)", low).group(2))
        return [EffectSpec("oc_mod", n, until)], []
    # --- attaque
    if _ATTACK_SUBJECT.search(b):
        scope = _scope(b)
        on_ = "vs_target" if vs_target else "unit"
        kws = _weapon_abilities(b_for_kw)
        for k in kws:
            effects.append(EffectSpec("weapon_ability", "$choice" if k == "$choice" else k, until, scope, on_, cond))
        for m in re.finditer(r"add (\d|one) to the (hit|wound) roll", low):
            effects.append(EffectSpec("hit_mod" if m.group(2) == "hit" else "wound_mod", _num(m.group(1)), until, scope, on_, cond))
        for m in re.finditer(r"re-roll (a|the) (hit|wound) roll( of 1)?|re-roll (hit|wound) rolls( of 1)?", low):
            which = m.group(2) or m.group(4)
            ones = bool(m.group(3) or m.group(5))
            effects.append(EffectSpec("reroll_hits" if which == "hit" else "reroll_wounds", "ones" if ones else "fails", until, scope, on_, cond))
        m = re.search(r"improve the armour penetration characteristic of (that attack|those attacks|\w+ weapons equipped by models in your unit|weapons equipped by models in your unit) by (\d|one)", low)
        if m:
            effects.append(EffectSpec("ap_mod", _num(m.group(2)), until, scope, on_, cond))
        m = re.search(r"improve the strength characteristic of (that attack|\w+ weapons equipped by models in your unit|weapons equipped by models in your unit) by (\d|one)", low)
        if m:
            effects.append(EffectSpec("strength_mod", _num(m.group(2)), until, scope, on_, cond))
        m = re.search(r"add (\d|one) to the (damage|attacks|strength) characteristic", low)
        if m:
            effects.append(EffectSpec({"damage": "damage_mod", "attacks": "attacks_mod", "strength": "strength_mod"}[m.group(2)], _num(m.group(1)), until, scope, on_, cond))
        m = re.search(r"improve the (ballistic|weapon) skill characteristic of (that attack|\w+ weapons equipped by models in your unit|weapons equipped by models in your unit) by (\d|one)", low)
        if m:
            effects.append(EffectSpec("skill_mod", _num(m.group(3)), until, "ranged" if m.group(1) == "ballistic" else "melee", on_, cond))
        m = re.search(r"(?:successful )?unmodified hit roll of (\d)\+ scores a critical hit", low)
        if m:
            effects.append(EffectSpec("crit_hit_on", int(m.group(1)), until, scope, on_, cond))
        if effects:
            # une phrase avec des armes nommées (« storm bolter weapons ») : appliqué à toutes les armes de la portée
            return effects, instants
    return None


def compile_text(effect: str, when: str = "", turn: str = "", phase: str = "", target: str = "") -> Compiled:
    ctx: Dict = {"needs_enemy": False, "choices": ()}
    effects: List[EffectSpec] = []
    instants: List[Instant] = []
    unparsed: List[str] = []
    text = re.sub(r"\s+", " ", effect or "").strip()
    sentences = [x for x in re.split(r"(?<=[.])\s+(?=[A-Z])", text) if x.strip()]
    for sen in sentences:
        try:
            res = _sentence(sen, ctx)
        except Exception:  # noqa: BLE001 — une phrase mal formée ne doit pas casser la compilation
            res = None
        if res is None:
            unparsed.append(sen)
            continue
        e, i = res
        effects += e
        instants += i
    parsed = bool(effects or instants or ctx.get("noop") or ctx.get("choices") or ctx.get("needs_enemy"))
    if not parsed:
        status = "manual"
    elif unparsed:
        status = "partial"
    else:
        status = "auto"
    kws = tuple(k.strip() for k in re.findall(r"\b([A-Z][A-Z’' \-]{2,}[A-Z])\b", target or "") if k.strip() not in ("TARGET",))
    return Compiled(status, effects, instants, parse_timing(when, turn, phase), ctx["needs_enemy"], ctx["choices"], kws, unparsed)


@lru_cache(maxsize=4096)
def _cached(effect: str, when: str, turn: str, phase: str, target: str) -> Compiled:
    return compile_text(effect, when, turn, phase, target)


def compile_stratagem(st) -> Compiled:
    """Traduction (mise en cache) d'un :class:`fortyk.data.detachments.Stratagem`."""
    return _cached(st.effect or st.text, st.when, st.turn, st.phase, st.target)
