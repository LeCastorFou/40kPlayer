"""Parsing des chaînes de caractéristiques Wahapedia vers des types Python.

Toutes les fonctions acceptent les variantes observées dans l'export V11 :
espaces parasites (``"D6 "``), tirets pour « sans objet » (``"-"``), ``"N/A"``,
casse incohérente dans les mots-clés d'arme (``"IGNORES COvER"``), etc.
Une valeur non applicable est rendue par ``None``, jamais par une exception,
sauf si la chaîne est manifestement d'un format inconnu (``ParseError``).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Optional, Union

__all__ = [
    "ParseError",
    "DiceExpr",
    "BaseSize",
    "WeaponKeyword",
    "KNOWN_WEAPON_KEYWORDS",
    "clean",
    "strip_html",
    "parse_int",
    "parse_dice",
    "parse_skill",
    "parse_save",
    "parse_inches",
    "parse_base_size",
    "parse_weapon_keywords",
]


class ParseError(ValueError):
    """Chaîne d'un format non reconnu."""


_WS = re.compile(r"\s+")
_NA = {"", "-", "–", "—", "n/a", "na", "none"}


def clean(text: str) -> str:
    """Normalise les espaces (dont les espaces insécables) et coupe les bords."""
    return _WS.sub(" ", text.replace(" ", " ")).strip()


def _is_na(text: str) -> bool:
    return clean(text).lower() in _NA


# --------------------------------------------------------------------------- dés


@dataclass(frozen=True)
class DiceExpr:
    """Expression de dés du type ``2D6+3``. ``n_dice == 0`` pour une valeur fixe."""

    n_dice: int = 0
    sides: int = 0
    flat: int = 0

    @classmethod
    def fixed(cls, value: int) -> "DiceExpr":
        return cls(0, 0, value)

    @property
    def is_fixed(self) -> bool:
        return self.n_dice == 0

    @property
    def mean(self) -> float:
        return self.n_dice * (self.sides + 1) / 2 + self.flat

    @property
    def min(self) -> int:
        return self.n_dice + self.flat

    @property
    def max(self) -> int:
        return self.n_dice * self.sides + self.flat

    def roll(self, rng) -> int:
        """Tire la valeur avec un ``random.Random`` (ou tout objet ayant ``randint``)."""
        return sum(rng.randint(1, self.sides) for _ in range(self.n_dice)) + self.flat

    def __str__(self) -> str:
        if self.is_fixed:
            return str(self.flat)
        dice = f"{self.n_dice if self.n_dice > 1 else ''}D{self.sides}"
        return f"{dice}+{self.flat}" if self.flat else dice


_FIXED_RE = re.compile(r"^\d+$")
_DICE_RE = re.compile(r"^(\d*)D(\d+)(?:\+(\d+))?$", re.IGNORECASE)


def parse_dice(text: str) -> Optional[DiceExpr]:
    """``"3"`` → fixe 3 ; ``"D6+1"`` → 1D6+1 ; ``"2D6"`` ; ``"-"`` → None."""
    s = clean(text).replace(" ", "")
    if _is_na(s):
        return None
    if _FIXED_RE.match(s):
        return DiceExpr.fixed(int(s))
    m = _DICE_RE.match(s)
    if not m:
        raise ParseError(f"expression de dés inconnue : {text!r}")
    n = int(m.group(1)) if m.group(1) else 1
    return DiceExpr(n, int(m.group(2)), int(m.group(3) or 0))


# ------------------------------------------------------------- entiers & jets


_INT_RE = re.compile(r"^[+-]?\d+$")


def parse_int(text: str) -> Optional[int]:
    """``"-2"`` → -2, ``"-0"`` → 0, ``"7 "`` → 7, ``"-"`` → None."""
    s = clean(text)
    if _is_na(s):
        return None
    if not _INT_RE.match(s):
        raise ParseError(f"entier attendu : {text!r}")
    return int(s)


_ROLL_RE = re.compile(r"^(\d)\+?\*?$")


def parse_skill(text: str) -> Optional[int]:
    """BS/WS : ``"3"`` ou ``"3+"`` → 3 ; ``"N/A"`` (Torrent) ou ``"-"`` → None."""
    s = clean(text)
    if _is_na(s):
        return None
    m = _ROLL_RE.match(s)
    if not m:
        raise ParseError(f"compétence de tir/combat inconnue : {text!r}")
    return int(m.group(1))


def parse_save(text: str) -> Optional[int]:
    """Sv / invul / Ld : ``"3+"`` → 3, ``"4"`` → 4, ``"4*"`` → 4, ``"-"`` → None."""
    s = clean(text)
    if _is_na(s):
        return None
    m = _ROLL_RE.match(s)
    if not m:
        raise ParseError(f"jet de sauvegarde inconnu : {text!r}")
    return int(m.group(1))


_INCHES_RE = re.compile(r'^(\d+(?:\.\d+)?)\+?\s*(?:"|”|in(?:ches)?)?$', re.IGNORECASE)


def parse_inches(text: str) -> Optional[float]:
    """``'6"'`` → 6.0, ``"24"`` → 24.0, ``'20+"'`` → 20.0, ``"Melee"``/``"-"`` → None."""
    s = clean(text)
    if _is_na(s) or s.lower() == "melee":
        return None
    m = _INCHES_RE.match(s)
    if not m:
        raise ParseError(f"distance en pouces inconnue : {text!r}")
    return float(m.group(1))


# ------------------------------------------------------------------- socles


@dataclass(frozen=True)
class BaseSize:
    """Socle d'une figurine. ``shape`` vaut ``"round"`` ou ``"oval"``. Dimensions en mm."""

    shape: str
    width_mm: float
    depth_mm: float
    flying: bool = False
    raw: str = ""

    @property
    def is_round(self) -> bool:
        return self.shape == "round"

    @property
    def radius_mm(self) -> float:
        """Rayon pour un socle rond ; pour un ovale, demi-grand axe (approximation prudente)."""
        return max(self.width_mm, self.depth_mm) / 2

    @property
    def radius_in(self) -> float:
        return self.radius_mm / 25.4


_BASE_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(?:x\s*(\d+(?:\.\d+)?))?\s*mm(?:\s*(flying)\s*base)?$",
    re.IGNORECASE,
)


def parse_base_size(text: str) -> Optional[BaseSize]:
    """``"32mm"`` → rond 32 ; ``"120 x 92mm flying base"`` → ovale volant ; ``"Use model"`` → None."""
    s = clean(text)
    if not s or s.lower() in {"use model", "no official base size"}:
        return None
    m = _BASE_RE.match(s)
    if not m:
        raise ParseError(f"taille de socle inconnue : {text!r}")
    w = float(m.group(1))
    d = float(m.group(2)) if m.group(2) else w
    return BaseSize("oval" if m.group(2) else "round", w, d, flying=bool(m.group(3)), raw=s)


# ------------------------------------------------------------- mots-clés d'arme


#: Mots-clés d'arme des règles de base (nom normalisé). Tout autre mot-clé est
#: conservé tel quel avec ``known == False`` (capacité spécifique à une fiche).
KNOWN_WEAPON_KEYWORDS = frozenset(
    {
        "assault",
        "heavy",
        "pistol",
        "torrent",
        "blast",
        "rapid fire",
        "melta",
        "sustained hits",
        "lethal hits",
        "devastating wounds",
        "twin-linked",
        "ignores cover",
        "indirect fire",
        "precision",
        "lance",
        "hazardous",
        "extra attacks",
        "one shot",
        "psychic",
        "anti",
        "conversion",
        "cleave",  # V11 : [CLEAVE N], référencé par des règles de détachement et stratagèmes
        "close-quarters",  # V11 : [CLOSE-QUARTERS], cité dans les règles de base
    }
)


@dataclass(frozen=True)
class WeaponKeyword:
    """Un mot-clé d'arme normalisé.

    ``name`` est en minuscules (``"sustained hits"``, ``"anti"``…).
    ``value`` porte le paramètre : entier (``rapid fire 2``, ``anti-infantry 4+`` → 4)
    ou :class:`DiceExpr` (``sustained hits D3``). ``plus`` indique la notation ``4+``.
    ``target`` n'est rempli que pour ``anti`` (``"infantry"``, ``"monster/vehicle"``…).
    ``condition`` reprend un éventuel suffixe ``": non-monster/vehicle"``.
    """

    name: str
    value: Union[int, DiceExpr, None] = None
    plus: bool = False
    target: Optional[str] = None
    condition: Optional[str] = None
    raw: str = ""

    @property
    def known(self) -> bool:
        return self.name in KNOWN_WEAPON_KEYWORDS

    def __str__(self) -> str:
        head = f"anti-{self.target}" if self.name == "anti" else self.name
        if self.value is not None:
            head += f" {self.value}{'+' if self.plus else ''}"
        return f"{head}: {self.condition}" if self.condition else head


_KW_RE = re.compile(r"^(?P<name>[a-z][a-z'’\-/ ]*?)\s*(?P<value>\d*d\d+(?:\+\d+)?|\d+\+?)?$")


def _parse_one_keyword(part: str) -> WeaponKeyword:
    raw = part
    s = clean(part).lower().replace("’", "'")
    condition: Optional[str] = None
    if ":" in s:
        s, condition = (p.strip() for p in s.split(":", 1))
    m = _KW_RE.match(s)
    if not m:
        return WeaponKeyword(name=s, condition=condition, raw=raw)
    name = clean(m.group("name"))
    value_txt = m.group("value")
    value: Union[int, DiceExpr, None] = None
    plus = False
    if value_txt:
        if value_txt.endswith("+"):
            plus = True
            value = int(value_txt[:-1])
        elif "d" in value_txt:
            value = parse_dice(value_txt)
        else:
            value = int(value_txt)
    target: Optional[str] = None
    if name.startswith("anti-") or name.startswith("anti "):
        target = name[5:].strip()
        name = "anti"
    return WeaponKeyword(name=name, value=value, plus=plus, target=target, condition=condition, raw=raw)


def parse_weapon_keywords(text: str) -> tuple:
    """``"assault, heavy"`` → deux mots-clés ; ``""`` → tuple vide."""
    return tuple(_parse_one_keyword(p) for p in text.split(",") if clean(p))


# -------------------------------------------------------------------- HTML


_BLOCK_TAG = re.compile(r"</?(?:p|br|li|ul|ol|div|tr|table|h[1-6])\b[^>]*>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")


def strip_html(text: str) -> str:
    """Texte brut d'une description HTML Wahapedia, une ligne par bloc."""
    s = _BLOCK_TAG.sub("\n", text)
    s = _ANY_TAG.sub("", s)
    s = html.unescape(s)
    lines = [clean(line) for line in s.split("\n")]
    return "\n".join(line for line in lines if line)
