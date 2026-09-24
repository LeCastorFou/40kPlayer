"""Paramètres et petites fonctions de règles du toy model (Warhammer 40,000 V11).

Tout ce qui est un *nombre issu des règles* vit ici, dans :class:`RulesConfig`, pour
être vérifié en un seul endroit. Les valeurs ci-dessous ont été confirmées par
Valentin (règles de base V11) sauf celles explicitement marquées « à confirmer »,
qui concernent la mission.

Points V11 qui diffèrent de la V10 et qui structurent le moteur :

* Engagement Range à 2" (horizontal) ;
* cohérence : à 2" d'au moins une autre figurine ET à 9" de toutes les autres ;
  la règle « deux voisines à partir de 7 figurines » n'existe plus ;
* le couvert n'améliore plus la sauvegarde : il inflige -1 au jet de touche du tireur.
  Une unité bénéficie du couvert contre une figurine qui tire si chacune de ses
  figurines est soit dans l'empreinte d'une ruine, soit pas pleinement visible de
  cette figurine. Ça s'évalue tireur par tireur ;
* Heavy : +1 à la touche si l'unité a bougé de moins de 3" ce tour ;
* Hazardous : sur 1 ou 2, 1 blessure mortelle (Infanterie) ou 3 (Véhicule / Monstre) ;
* la portée d'un pion d'objectif (3") se mesure depuis le bord du pion de 40 mm ;
* les empreintes de décor bloquent les lignes de vue (pas seulement les murs) ; toutes les
  empreintes (ruines, barricades) sont des zones de terrain pour le couvert et Hidden ;
* Hidden : une unité INFANTRY ou BEASTS dont chaque figurine touche une empreinte de décor (« un
  orteil suffit ») ne peut être visée au tir que par des figurines situées à 15" ou moins ; jamais
  pour un MONSTER / VEHICLE ;
* ruines : un socle qui touche l'empreinte est « dedans » : il voit à travers la ruine et se fait voir ;
* monstres et véhicules (validé) : Big Guns Never Tire ; couvert seulement s'ils sont partiellement
  occultés ou entièrement dans une ruine ; ils franchissent les empreintes mais pas les murs (pas de
  murs sur la carte pour l'instant) ;
* transports et Scouts selon le PDF des règles de base V11 (sections 18 et 24) ;
* phase de combat : seules les figurines à portée d'engagement (2") d'une figurine ennemie
  peuvent frapper. Une unité qui a chargé gagne Fights First. Toutes les unités Fights First
  s'activent d'abord, en alternance en commençant par le joueur actif ; puis les autres, en
  alternance en commençant par le joueur qui n'a pas activé en dernier ;
* éligibilité au tir : après une Advance seules les armes Assault tirent ; après un Fall Back
  ni tir ni charge (sauf capacité du type Thrill Seekers) ; une unité à portée d'engagement ne
  tire qu'avec ses Pistols, et uniquement sur des unités à portée d'engagement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = ["RulesConfig", "DEFAULT_RULES", "wound_roll_needed", "save_needed", "clamp_modifier", "clamp_roll"]


@dataclass(frozen=True)
class RulesConfig:
    # --- table et mission -------------------------------------------------
    board_in: Tuple[float, float] = (44.0, 30.0)  #: table Combat Patrol — choix du toy model
    battle_rounds: int = 5
    objective_range_in: float = 3.0  #: mesuré depuis le bord du pion
    objective_marker_radius_in: float = 20.0 / 25.4  #: pion de 40 mm

    # --- distances de base ------------------------------------------------
    engagement_range_in: float = 2.0
    coherency_range_in: float = 2.0  #: à ≤ 2" d'au moins une autre figurine de l'unité
    coherency_max_spread_in: float = 9.0  #: et à ≤ 9" de toutes les autres
    charge_declare_range_in: float = 12.0
    pile_in_in: float = 3.0
    consolidate_in: float = 3.0  #: 12.08 : aussi la portée des modes « engaging » / « objective »
    pile_in_target_range_in: float = 5.0  #: 12.03 : unité non engagée : cibles de pile-in à 5"
    fight_pass_range_in: float = 5.0  #: annexe : une unité éligible à plus de 5" de tout ennemi peut passer
    reinforcements_distance_in: float = 9.0  #: à confirmer (pas utilisé par le toy model)

    # --- dés et modificateurs ---------------------------------------------
    critical_roll: int = 6  #: un 6 naturel est un critique (touche / blessure) et réussit toujours
    modifier_cap: int = 1  #: modificateurs de touche et de blessure plafonnés à ±1 (net)
    natural_one_always_fails: bool = True
    best_possible_save: int = 2  #: une sauvegarde ne peut jamais être meilleure que 2+

    # --- couvert, Hidden et mots-clés d'arme ---------------------------------
    cover_skill_penalty: int = 1  #: 13.08 : le couvert dégrade de 1 la CT de l'attaque (ce n'est pas un modificateur de touche)
    cover_keywords: Tuple[str, ...] = ("Infantry", "Beasts", "Swarm")  #: 13.08 : à couvert « dans une zone de terrain »
    hidden_range_in: float = 15.0  #: portée de détection (13.09) : une figurine cachée n'est visible que des ennemis à 15" ou moins
    hidden_keywords: Tuple[str, ...] = ("Infantry", "Beasts", "Swarm")  #: 13.09 : dans une zone contenant un décor dense, sans avoir tiré ce tour ni au précédent
    indirect_fail_below: int = 6  #: 10.07 : un jet de touche non modifié de 1-5 rate…
    indirect_fail_below_spotted: int = 4  #: … 1-3 si l'unité est restée immobile et que la cible est visible d'une unité amie
    los_clearance_in: float = 0.25  #: marge minimale d'une ligne de vue par rapport au bord d'un décor opaque
    heavy_move_threshold_in: float = 3.0  #: [HEAVY] (24.16) : +1 touche si aucune figurine n'a bougé de plus de 3", unité désengagée et pas posée ce tour
    heavy_hit_bonus: int = 1
    blast_models_per_extra_attack: int = 5
    hazardous_fail_on: Tuple[int, ...] = (1, 2)  #: jet de danger (06.03) raté sur 1-2 : 1 BM, ou 3 si toutes les figurines sont MONSTER / VEHICLE
    hazardous_mortal_wounds_infantry: int = 1
    hazardous_mortal_wounds_vehicle_monster: int = 3

    # --- monstres et véhicules ------------------------------------------------
    big_guns_never_tire: bool = True  #: validé V11 : un MONSTER / VEHICLE tire même à portée d'engagement (cibles engagées, -1 touche hors Pistol) et peut être visé quand il est engagé avec une unité amie du tireur (-1 touche hors Pistol)
    big_guns_hit_penalty: int = 1
    deadly_demise_trigger: int = 6  #: à la destruction, D6 = 6 → X blessures mortelles à chaque unité à 6"
    deadly_demise_range_in: float = 6.0
    lone_operative_range_in: float = 12.0  #: Lone Operative : ciblable au tir seulement à 12" ou moins

    # --- transports et Scouts (règles de base V11, sections 18 et 24.31-24.32) --------------
    # Débarquer (18.04) : mode imposé — « rapid » si le transport a fait un mouvement normal cette
    # phase (3", pas de charge ce tour) ; « tactical » s'il n'a pas bougé et que l'unité tient à 3"
    # (puis mouvement normal ou Advance) ; sinon « combat » (6", un jet de danger par figurine, pose
    # possible au contact des ennemis engagés avec le transport, puis battle-shock et pas de charge).
    # Interdit si le transport a fait une Advance ou un Fall Back, ou si l'unité a embarqué cette phase.
    disembark_range_in: float = 3.0
    combat_disembark_range_in: float = 6.0
    embark_range_in: float = 3.0  #: 18.02 : chaque figurine à 3" du transport après un mouvement normal / Advance / Fall Back
    # Transport détruit (18.05) : un jet de danger par figurine, puis chaque figurine entièrement à 6"
    # et aussi près que possible de l'épave (sinon détruite) ; battle-shock et pas de charge ce tour ;
    # Deadly Demise se résout ensuite.
    emergency_disembark_range_in: float = 6.0
    scout_enemy_distance_in: float = 8.0  #: 24.32 : fin du mouvement de scout à plus de 8" de toute unité ennemie

    # --- sous-phases -------------------------------------------------------
    advance_dice: str = "D6"
    charge_dice: str = "2D6"
    battle_shock_dice: str = "2D6"  #: en phase de commandement, si effectif ≤ moitié
    charger_fights_first: bool = True
    only_models_in_engagement_range_fight: bool = True  #: V11 : pas de « contact de socle avec un ami engagé »
    vp_round_cap: int = 15

    @property
    def objective_control_radius_in(self) -> float:
        """Distance centre du pion → bord du socle en dessous de laquelle une figurine
        est « à portée » de l'objectif (3" depuis le bord d'un pion de 40 mm)."""
        return self.objective_range_in + self.objective_marker_radius_in


DEFAULT_RULES = RulesConfig()


def clamp_modifier(modifier: int, rules: RulesConfig = DEFAULT_RULES) -> int:
    """Plafonne un modificateur net de jet de touche / blessure à ±cap."""
    return max(-rules.modifier_cap, min(rules.modifier_cap, modifier))


def clamp_roll(needed: int) -> int:
    """Un résultat requis est toujours entre 2+ et 6+ (le 1 échoue, le 6 réussit)."""
    return max(2, min(6, needed))


def wound_roll_needed(strength: int, toughness: int) -> int:
    """Jet de blessure minimal (sur D6) pour une Force donnée contre une Endurance donnée.

    S ≥ 2T → 2+ ; S > T → 3+ ; S = T → 4+ ; S < T → 5+ ; S ≤ T/2 → 6+.
    """
    if strength >= 2 * toughness:
        return 2
    if strength > toughness:
        return 3
    if strength == toughness:
        return 4
    if 2 * strength <= toughness:
        return 6
    return 5


def save_needed(
    save: int,
    ap: int,
    invuln: Optional[int] = None,
    rules: RulesConfig = DEFAULT_RULES,
) -> Optional[int]:
    """Jet de sauvegarde minimal après PA ; None si aucune sauvegarde n'est possible.

    * la sauvegarde d'armure est dégradée par la PA (``ap`` est négatif ou nul : AP -1 → +1 au jet) ;
    * la sauvegarde invulnérable ignore la PA ; on garde la meilleure des deux ;
    * un résultat requis supérieur à 6 est impossible → None (l'invulnérable peut encore sauver).

    En V11 le couvert ne modifie pas la sauvegarde (il pénalise la touche), d'où l'absence
    de paramètre ``in_cover`` ici.
    """
    armour = max(save - ap, rules.best_possible_save)  # ap ≤ 0 : AP -2 sur une 3+ donne 5+
    candidates = [v for v in (armour, invuln) if v is not None and v <= 6]
    return min(candidates) if candidates else None
