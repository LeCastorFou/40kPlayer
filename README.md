# 40kPlayer

Toy model 2D de Warhammer 40,000 (V11) avec une IA adverse. Les figurines sont des
disques de la taille de leur socle, les décors des polygones ; le moteur calcule
lignes de vue, distances, charges, et l'IA cherche à gagner aux points de victoire.

## État

| Brique | État |
|---|---|
| `fortyk.data` — lecture de l'export Wahapedia, fiches d'unité typées | fait, testé |
| `fortyk.engine.geometry` — disques, polygones, lignes de vue, déplacements | fait, testé (vérifié contre shapely) |
| `fortyk.engine.rules` — paramètres de règles V11 (validés), table de blessure, sauvegardes | fait, testé |
| `fortyk.engine.attack` — séquence d'attaque aux dés et en espérance (Torrent, Lethal, Sustained, DW, Anti-X, FNP…) | fait, testé (dés ≡ espérance) |
| `fortyk.engine.layout` — carte Layout A (44"×60", zones, 5 objectifs, 17 décors) + rendu PNG | fait, testé (positions des objectifs à confirmer) |
| `fortyk.engine.mission` — Unstoppable Force : score, plafond 15 VP/round, Level of Control | fait, testé |
| `fortyk.engine.state` / `army` — figurines, unités (équipement lu sur la fiche), contrôle d'objectifs collant | fait, testé |
| `fortyk.engine.movement` — mouvements en formation, légalité, candidats discrets, charge, pile-in, consolidation | fait, testé |
| `fortyk.engine.combat` — tir et mêlée depuis les positions (portée, LdV, couvert par tireur, mots-clés, capacités du duo) | fait, testé |
| `fortyk.engine.engine` — machine à états : `decision` / `step` / clone / replay, journal structuré | fait, testé (rejouabilité, clones indépendants) |
| `fortyk.engine.fastgeo` — géométrie vectorisée numpy (lignes de vue avec cache, mouvements, déploiement) | fait, testé (identique à la référence Python) |
| `fortyk.engine.game` — pilote : relie le moteur aux agents (humain, bots), mode pas à pas | fait, testé (bot vs bot, interface web) |
| `fortyk.engine.record` — sauvegarde / relecture des parties en JSON | fait, testé |
| `fortyk.data.army_list` — import de listes d'armée (export texte NewRecruit), contrôle des points | fait, testé |
| `fortyk.data.detachments` — détachements V11, règles, stratagèmes, améliorations | données lues ; effets pas encore joués |
| `fortyk.engine.coverage` — ce qu'une liste utilise et ce que le moteur en joue | fait, testé |
| `fortyk.agents` — bot aléatoire, joueur humain en terminal | fait ; heuristique et recherche arborescente à venir |
| `fortyk.web` — service web : parties à deux (liens secrets) ou contre le bot, sauvegardées, annulables ; glisser-déposer des socles, journal en direct | fait, testé (API + Playwright à deux navigateurs) |
| `fortyk.training` — parties sauvegardées → export d'entraînement (état avant chaque décision, action, résultat) | fait, testé |

## Moteur pour l'IA

Le moteur est une **machine à états** : toute la progression de la partie vit dans l'état
(`GameState.flow`), pas dans la pile d'appels. On peut donc cloner une partie à n'importe quelle
décision, explorer des coups sur des copies, et rejouer une partie à l'identique depuis sa graine et
la liste de ses actions — ce qu'exigent la recherche arborescente et le self-play.

```python
from fortyk.engine import Engine
from fortyk.toy import new_toy_state

engine = Engine()
state = new_toy_state(seed=1)
engine.start(state)                      # déploiement, jet d'initiative… jusqu'à la 1re décision
while not engine.is_over(state):
    d = engine.decision(state)           # qui décide (d.side), quoi (d.kind), options légales (d.options)
    engine.step(state, d.options[0])     # applique, puis avance jusqu'à la décision suivante
engine.result(state)                     # vainqueur, scores

copy = state.clone(record=False)         # copie rapide (~90 µs) sans journal, pour les simulations
copy = state.clone(seed=42)              # même position, autres dés : un autre futur possible
engine.replay(initial_state, state.history)   # rejoue la partie à l'identique
```

`state.log` garde le journal texte, `state.events` le même journal en événements structurés (dés,
tirs, morts, scores…), `state.history` les actions `(camp, action)` ; `fortyk.engine.record` les
sauvegarde en JSON. Construire une décision ne consomme jamais de dé : une partie ne dépend que de sa
graine et de ses actions.

Vitesse (bot aléatoire contre bot aléatoire, 3 unités par camp) : **~95 ms par partie**, ~90 décisions,
soit ~1 ms par décision (contre ~1,4 s par partie avant la vectorisation). `python3 scripts/bench.py`
le mesure. Le moteur demande numpy (`pip install numpy`).

## Listes d'armée

Une liste exportée en texte depuis NewRecruit (`data/lists/ec_mercurial_host_2000.txt` en exemple) est
lue puis rattachée à l'export Wahapedia : fiches, figurines, armes figurine par figurine (`2x Excruciator
cannon` compte double), personnages attachés (`Leading …` / `Attached to …`, vérifiés contre les fiches
Leader), améliorations, détachements (DP, Force Disposition), règle de détachement choisie, stratagèmes
disponibles (les 10 stratagèmes core V11 + ceux des détachements, découpés en QUAND / CIBLE / EFFET).
Les points sont recalculés (paliers d'exemplaires, équipement payant, améliorations) et toute
incohérence est signalée.

**Importer une autre liste** — le plus simple, dans le navigateur : sur l'accueil du service, section
« Importer une liste », colle l'export texte de NewRecruit, « Vérifier » (points recalculés, points à vérifier,
règles jouées par le moteur), « Enregistrer » (la liste est rangée dans `data/lists/NOM.txt`, ou dans
`FORTYK_LISTS_DIR` sur un serveur), puis choisis-la dans « Nouvelle partie ». Depuis le terminal, directement
depuis le presse-papiers :

```bash
pbpaste | python3 scripts/import_list.py - --save ma_liste        # vérifie et range data/lists/ma_liste.txt
python3 scripts/serve.py --vs-bot --attacker-list ma_liste --defender-list ec_mercurial_host_2000   # par leur nom
python3 scripts/import_list.py ec_mercurial_host_2000              # fiche d'armée + couverture
python3 scripts/import_list.py ma_liste --weapons --json ma_liste.json   # profils d'arme, export JSON
```

```python
from fortyk.toy import new_state_from_lists
state = new_state_from_lists("ma_liste", "ec_mercurial_host_2000")   # partie 2000 pts (noms ou fichiers)
state = new_state_from_lists("ma_liste", None)                      # None : roster du toy model pour ce camp
```

Dans le moteur, un personnage qui mène une unité la rejoint : ses figurines sont dans l'unité, protégées
à l'allocation des blessures (les gardes du corps d'abord), l'unité prend ses capacités et mots-clés, son
M est celui de la figurine la plus lente, son Ld le meilleur, l'OC se compte figurine par figurine. Les
figurines sans socle (« Use model » : Rhino, Land Raider, Predator, motos…) reçoivent l'empreinte de
`data/hulls.json` : coque rectangulaire (Rhino 117 × 88 mm) orientée vers l'adversaire, ovale pour les
motos. Les dimensions sont approximatives : corrige-les dans ce fichier (relu à chaque import).

**Couverture** : le rapport de `import_list.py` classe chaque règle de la liste en jouée / partielle /
non jouée. Pour la liste d'exemple : 14 jouées, 4 partielles, 47 non jouées — dont tous les stratagèmes,
les règles et les améliorations de détachement (toutes les règles des détachements pris sont actives dans
la liste, pas encore dans le moteur), Infiltrators, Deep Strike, Firing Deck, Precision et la plupart des
capacités de fiche. C'est la liste de travail des prochaines étapes.

**Transports** : la capacité est lue dans la fiche (nombre de places, mots-clés requis, exclusions
« excluding … » / « It cannot transport … », places doubles « takes up the space of 2 models »). Une unité
peut commencer la partie à bord d'un transport déjà déployé (bouton « Commencer à bord » au déploiement),
débarquer en phase de mouvement (couronne autour de la coque proposée, ou placement à la main) et
ré-embarquer en fin de mouvement à 3" du transport. Les unités à bord sont hors table : ni ciblables, ni
comptées sur les objectifs. **Scouts** : avant le round 1, mouvement de X" fini à plus de 9" de l'ennemi ; un
transport dédié en profite si toutes les figurines à bord ont Scouts.

## Données

Les règles et fiches viennent de l'export CSV officiel de [Wahapedia](https://wahapedia.ru/wh40k11ed/the-rules/data-export/)
(spécification : `Export Data Specs.xlsx`). Données © Games Workshop, compilées par Wahapedia —
« powered by Wahapedia ».

```bash
python3 scripts/fetch_wahapedia.py            # télécharge les 21 CSV dans data/wahapedia/raw/
python3 scripts/fetch_wahapedia.py --check    # l'export distant a-t-il changé ?
```

## Utilisation

```python
from fortyk.data import load_catalog

cat = load_catalog()                       # ~1 s pour 1660 fiches
ds = cat.get("Intercessor Squad")          # cat.get("Leman Russ Commander", faction="AM") si ambigu
m = ds.models[0]
m.move_in, m.toughness, m.save, m.wounds, m.oc, m.base.radius_in
rifle = ds.weapon("Bolt rifle")
rifle.attacks.mean, rifle.skill, rifle.strength, rifle.ap, rifle.damage, rifle.has("assault")
ds.cost_for(10), ds.keywords, ds.faction_keywords, [a.name for a in ds.abilities]
```

```bash
python3 scripts/show_datasheet.py "Intercessor Squad"        # fiche lisible
python3 scripts/show_datasheet.py --faction EC --list        # toutes les fiches d'une faction
python3 scripts/show_datasheet.py Infractors --json          # export JSON
python3 scripts/mathhammer.py "Intercessor Squad" "Bolt rifle" "Infractors" --count 5 --cover --reroll-hits fails
python3 scripts/render_layout.py layout_a          # PNG de contrôle de la carte (Pillow requis)
python3 scripts/play.py --seed 3 --png out/        # une partie aléatoire vs aléatoire, journal + PNG par tour
python3 scripts/play.py --quiet --games 20         # statistiques sur 20 parties
python3 scripts/play_human.py --side attacker      # toi contre le bot, dans le terminal (plateau dans board.png)
python3 scripts/serve.py                            # service web : http://127.0.0.1:8040/ (parties à deux ou contre le bot)
python3 scripts/serve.py --vs-bot --side defender   # crée tout de suite une partie contre le bot et l'ouvre
python3 scripts/export_games.py data/games parties.jsonl   # parties sauvegardées → données d'entraînement
python3 scripts/bench.py --games 50                 # vitesse du moteur (parties aléatoires, clonage)
```

### Service web : parties à deux, sauvegardées

`python3 scripts/serve.py` lance le service (bibliothèque standard, pas de dépendance) et ouvre l'accueil :

- **Nouvelle partie** : ton nom, ton camp, ta liste (ou le toy model), et l'adversaire — le bot (tu choisis
  sa liste, la partie démarre tout de suite) ou un joueur. Contre un joueur, tu reçois un **code de partie**
  (`K7F-3QX`, sans 0/O ni 1/I pour se dicter facilement) et un lien d'invitation ; la partie attend.
- **Rejoindre une partie** : l'adversaire tape le code sur l'accueil (ou ouvre le lien d'invitation), donne
  son nom et **choisit sa propre liste** ; la partie démarre aussitôt (la page du créateur se met à jour
  toute seule). Le code ne sert qu'une fois.
- Chaque joueur a son **lien secret** `/g/<partie>?t=<jeton>`, qui ne donne accès qu'à son camp et rouvre la
  partie depuis n'importe quel appareil ; sans jeton, `/g/<partie>` est le lien des spectateurs. Pas de compte.
- **Mes parties** (mémorisées par le navigateur) et **toutes les parties** du serveur : état, round, score,
  à qui de jouer ; « Reprendre » rouvre la partie là où elle en était.
- **Importer une liste** : coller l'export texte NewRecruit, vérifier (points recalculés, points à vérifier,
  règles jouées), enregistrer ; elle apparaît dans les menus.

**La partie est sauvegardée après chaque action** (un fichier JSON par partie dans `data/games`, ou le dossier
`FORTYK_DATA_DIR`) : on peut fermer la page ou redémarrer le serveur, la partie reprend à l'identique. Le plateau
de chaque joueur se met à jour à chaque action adverse ; l'onglet affiche « ● À toi » quand c'est ton tour.

**Annuler** : n'importe quel joueur peut revenir en arrière à tout moment — « ↶ Annuler » (ou Ctrl/Cmd + Z)
annule la dernière action d'un joueur, quelle qu'elle soit (placement, jet d'Advance, charge, tir, combat…) ;
le panneau « Historique » liste toutes les actions et « revenir avant » (deux clics) revient avant n'importe
laquelle, en annulant aussi toutes les suivantes. L'adversaire voit un bandeau (« Paul a annulé 3 actions —
retour à round 2, tir ») et le journal est réécrit. **Après une annulation, les dés sont nouveaux** : rejouer
la même action donne un autre jet (la graine est re-tirée et enregistrée, pour que la partie reste rejouable
à l'identique). Les actions annulées restent dans le fichier de la partie.

Contre le bot, le mode « pas à pas » (coché par défaut) arrête chaque action du bot sur « Continuer »
(ou Espace), et « Annuler » revient à ta dernière décision (le bot ne rejoue pas aussitôt).

Sur le plateau : en phase de mouvement, **glisse directement une de tes unités** (plus besoin de la choisir
d'abord) — une figurine, une sélection encadrée, ou toute l'unité (Maj, ou le bouton « toute l'unité » au
doigt). Pendant le glissement, le serveur vérifie le placement à blanc : le socle fautif passe en rouge avec
la raison (distance, table, chevauchement, socle ennemi traversé, portée d'engagement, cohérence), « ✓
placement légal » sinon. Advance en deux temps (le D6, puis le placement déjà glissé), repli ordonné ou
Desperate Escape quand tu es engagé ; tir, charge, combat et serment par bouton ou clic sur l'unité ennemie,
avec les **dégâts et pertes attendus** affichés sur chaque cible ; clic sur une unité (ou I en survol) : sa
**fiche** (profils, armes, capacités). Les unités qui ne peuvent pas agir sont listées avec la raison.

Fluidité (menu « Réglages », par joueur) : **actions automatiques** (combat à une seule option, tir vers la
seule cible possible, serment à cible unique : joués d'office), **déploiement automatique** des unités
restantes, **stratagèmes** (fenêtres proposées ou non), pas à pas contre le bot, notifications du navigateur
et son quand c'est à toi, animations. Raccourcis : Entrée valider, S immobile (ou « non » à un stratagème),
A Advance, N unité suivante, E fin de phase, R pivoter, I fiche, 1…9 stratagème, Ctrl/Cmd + Z annuler.
La page se met à jour dès que l'adversaire agit (long-polling) ; ses mouvements sont animés et un fil résume
tirs, charges et jets. En revenant, un bandeau « **Pendant ton absence** » liste ses actions et peut les
**rejouer en animation** (aussi depuis l'historique : « Rejouer les 30 dernières »). Sur tablette et
téléphone : glisser au doigt, pincer pour zoomer, glisser le fond pour se déplacer (boutons + − ⤢ aussi).

**Stratagèmes** (15) : le moteur ouvre une fenêtre au moment exact où un stratagème de base peut servir, et
seulement s'il est utilisable (assez de CP, pas déjà utilisé dans la phase, une cible éligible et utile) ;
sinon la partie continue sans rien demander. Joués : Command Re-roll (jets d'Advance et de charge), Epic
Challenge, Insane Bravery (une fois par bataille), Explosives (proposé parmi les options de la phase de
tir), Crushing Impact, Fire Overwatch (tir d'opportunité : 6 non modifié), Smokescreen, Heroic Intervention
(Leap to Defend, ou Into the Fray pour +1 CP) et Counteroffensive. Limites de 15.01 : même stratagème une fois
par phase, une unité ciblée par un seul stratagème par phase, jamais une unité battle-shocked.

**Latitudes des règles** (réserves, soins, figurines qui reviennent, bonus de détachement…) :

- **Réserves stratégiques** (20) : au déploiement, « Placer en réserve stratégique » (50 % du format au plus ;
  AIRCRAFT toujours). À partir du round 2, l'unité est proposée dans la phase de mouvement : on la glisse
  (bande de 6" le long des bords en vert, 8" autour des ennemis en rouge, zone adverse interdite avant le
  round 3) ou elle reste en réserve ; **Deep Strike** : n'importe où à plus de 8" de l'ennemi. Les réserves
  jamais arrivées sont détruites à la fin du round 3 (sauf celles remises en réserve pendant la bataille).
  Rapid Ingress (15.07) et Infiltrators (24.20) sont joués ; les AIRCRAFT retournent en réserve à la fin du tour
  adverse.
- **Panneau « Stratagèmes / règles »** : tous les stratagèmes de ta liste (base et détachements), avec leur
  texte, leur coût et s'ils sont jouables maintenant (CP, tour et phase, limites de 15.01, cible). Le moteur
  traduit lui-même le texte Wahapedia quand il le comprend (`fortyk/rules_compiler.py` : +1 touche,
  relances, PA, mots-clés d'arme, Feel No Pain, invulnérable, couvert, Stealth, Fights First, Advance et
  charge, retour en réserve, blessures mortelles…) et l'applique ; sinon il rappelle dans le journal ce qui
  reste à faire. Les stratagèmes de détachement sont aussi **proposés au bon moment** : juste après que
  l'ennemi a choisi ses cibles, quand ton unité est choisie pour tirer ou combattre, au début et à la fin
  des phases.
- **Effet manuel** (onglet du panneau) : pour tout ce qui n'est pas encore traduit, n'importe quel joueur,
  à tout moment — règle invoquée, CP, soigner des PV, **ramener des figurines** (placées au glisser sur le
  plateau, vérifiées en direct), blessures mortelles, retirer des figurines, retour en réserve, poser ou
  déplacer une unité, battle-shock, ou un **effet à durée** (+1 touche, relance, FNP, invulnérable,
  mot-clé d'arme, +2" de mouvement… jusqu'à la fin de la phase, du tour, du round). C'est journalisé,
  l'adversaire le voit et peut l'annuler ; les données d'entraînement le gardent.

Couverture de la traduction automatique, par faction : `python3 scripts/rules_coverage.py` (vague 1 :
40 % des 1 324 stratagèmes de détachement entièrement automatiques, 15 % en partie, le reste en effet
manuel ; `--unparsed 40` liste les phrases les plus fréquentes qui restent à traduire).

Options : `--host 0.0.0.0` (réseau local), `--port`, `--data-dir`,
`--vs-bot --side defender --attacker-list <nom>` (partie immédiate contre le bot). N'importe qui ayant
l'adresse du service peut créer une partie : c'est le code de partie qui protège l'entrée d'un adversaire.

### Données d'entraînement

Chaque partie est un document `fortyk-game/2` : configuration (graine, carte, listes **en texte**, pour que la
partie ne dépende pas d'un fichier qui changerait), joueurs, chaque décision dans l'ordre (camp, humain ou bot,
genre de décision, round, phase, libellé lisible, action JSON, horodatage), re-tirages des dés et annulations
(avec les actions annulées). Téléchargeable depuis l'accueil (« JSON », sans les jetons) ou
`GET /api/g/<id>/record`.

L'export d'entraînement (`fortyk-training/1`, bouton « Entraînement » ou `GET /api/g/<id>/export`) rejoue la
partie et donne pour chaque décision l'**état compact du plateau avant la décision** (figurines : position,
orientation, PV ; statuts ; score, CP, contrôle des objectifs), la décision posée et l'action choisie, puis les
événements du moteur (dés, dégâts, scores) et le résultat. Pour tout un dossier :
`python3 scripts/export_games.py /data/games parties.jsonl [--finished] [--no-states]` (une partie par ligne).

### Mise en ligne sur Dokploy

Le dépôt contient un `Dockerfile` (Python 3.12 slim + numpy ; fiches Wahapedia, carte et listes copiées dans
l'image ; utilisateur non root ; healthcheck `/healthz`) et un `docker-compose.yml`. Dans Dokploy :

1. **Create Application** → source GitHub (le dépôt 40kPlayer, branche principale) → Build Type **Dockerfile**.
2. Aucune variable d'environnement n'est nécessaire.
3. **Advanced → Volumes** : un *Volume Mount* (volume nommé, par ex. `fortyk-data`) monté sur `/data` — c'est
   là que vivent les parties (`/data/games`) et les listes importées (`/data/lists`). Un volume nommé garde les
   bons droits pour l'utilisateur du conteneur ; avec un *Bind Mount*, donne le dossier à l'uid 10001.
4. **Domains** : ton domaine, port **8040**, HTTPS (Let's Encrypt).
5. **Deploy**. Pour récupérer les données d'entraînement : bouton « Entraînement » de chaque partie, ou
   `docker exec <conteneur> python scripts/export_games.py /data/games /data/parties.jsonl`.

Variante *Compose* : Dokploy peut aussi utiliser `docker-compose.yml` (même volume).

**Automatisé** : le workflow « Dokploy (test) » (`.github/workflows/dokploy.yml`, lancé depuis l'onglet
Actions) fait tout cela par l'API de Dokploy avec `scripts/dokploy_setup.py` — projet `other`, application
`40kplayer`, source GitHub, Dockerfile, volume `/data`, domaine (généré, ou celui passé en entrée), déploiement
et contrôle de `/healthz` — à partir des secrets du dépôt `DOKPLOY_URL` et `DOKPLOY_API_KEY`. Relançable sans
rien dupliquer. Dokploy redéploie ensuite tout seul à chaque push sur `main`.
Les parties sont chargées en mémoire par un seul processus : ne pas lancer plusieurs réplicas.

### Jouer dans le terminal

`scripts/play_human.py` te fait jouer un camp : à chaque décision le plateau est redessiné dans
`board.png` (à garder ouvert dans Aperçu, ton unité active est en jaune, marges graduées en pouces),
le terminal affiche la situation et les options numérotées. Saisie libre, validée par le moteur :
`x y` pour déployer (centre de la formation), `m ANGLE DIST` pour bouger (0° = droite, 90° = bas),
`v DX DY` (vecteur), `a ANGLE` (Advance), `r ANGLE DIST` (repli), `s` (immobile), `i` (situation),
`?` (aide), `q` (abandon). Les unités bougent en formation ; le déplacement figurine par figurine
viendra avec l'interface web.

## Simplifications du toy model (v0)

- Une unité se déplace en formation (translation commune), ce qui garantit la cohérence et réduit
  l'espace d'actions ; charge, pile-in et consolidation déplacent les figurines une à une vers l'ennemi.
- Décors tous franchissables, pas d'étages ni de murs ; ce sont les empreintes qui bloquent la vue.
- Une unité tire tout sur une seule cible ; une figurine tire ses pistolets ou ses autres armes (le lot le
  plus rentable) ; pour une arme à plusieurs profils, le profil aux dégâts attendus maximaux.
- Allocation V11 par groupes (voir plus bas) ; stratagèmes de base joués (service web). Placement d'urgence d'un
  transport détruit : automatique.
- Rosters : 2 × 5 Intercessors (Oath of Moment, Objective Secured, Hail of Bolts) + 1 Redemptor Dreadnought
  (Deadly Demise D3, Duty Eternal) contre 2 × 5 Infractors (Thrill Seekers, Excessive Assault) + 1 Daemon Prince
  of Slaanesh (Deadly Demise D3, Lord of Excess, Excessive Vigour ; Ecstatic Death pas encore joué). Le véhicule
  et le monstre servent à tester les règles qui leur sont propres. Tous les décors restent franchissables, y
  compris pour eux, faute de murs sur la carte.

## Règles V11 retenues (PDF des règles de base, validées avec Valentin)

Le PDF (`eng_01-06_warhammer40k_new40k_core_rules`, 88 pages) a été relu en entier et le moteur aligné dessus.
Le texte extrait est rangé localement dans `data/rules/` (ignoré par git : c'est le texte de Games Workshop, à
ne pas publier).

- Engagement Range 2" ; cohérence 2" d'au moins une figurine et 9" de toutes les autres ; pile-in et
  consolidation 3". **Cohérence de fin de tour** (03.03) : une unité hors cohérence retire des figurines
  (le moteur garde le plus grand groupe cohérent).
- **Mouvement** (03.01, 09, 17.01) : on traverse les figurines amies, jamais le socle d'un ennemi ; un
  MONSTER / VEHICLE passe par-dessus les figurines ennemies non M/V en mouvement normal / Advance ; on peut
  passer à portée d'engagement pendant le mouvement, seule la fin doit être désengagée ; pivoter ne compte
  pas (distance = déplacement du centre, aucune partie ne bouge plus que le mouvement autorisé).
  **Fall Back** : *ordered retreat* (on ne traverse pas les ennemis) ou *desperate escape* (on traverse,
  un jet de danger par figurine avant de bouger, puis test de battle-shock) — une unité battle-shocked n'a
  que le desperate escape.
- **Charge** (11.04) : on déclare, on lance 2D6, puis on choisit **une ou plusieurs** cibles à 12" qu'une
  figurine peut atteindre avec ce jet ; un double 1 échoue toujours. Réussite : l'unité finit **engagée (2")
  avec chacune des cibles** et **au moins une figurine socle à socle** avec une cible (choix de Valentin,
  plus strict que le PDF) ; toute figurine qui peut arriver à 1" d'une cible le fait, sinon à 2", sinon elle
  finit plus près ; jamais engagée avec une non-cible ; cohérence. Placement à la main ou automatique.
- **Phase de combat en étapes** (12) : tous les pile-in (joueur actif puis adversaire), puis les combats
  (Fights First en alternance, puis les autres ; une unité désengagée mais à 5" d'un ennemi fait un *overrun
  fight* : pile-in supplémentaire puis frappe), puis toutes les consolidations dans le mode imposé
  (*ongoing* vers les unités engagées, *engaging* vers l'ennemi le plus proche à 3", *objective* vers un
  objectif à 3"). Chaque figurine frappe une unité qu'elle engage (la cible choisie d'abord).
- **Couvert** (13.08) : la CT de l'attaque est dégradée de 1 (3+ → 4+ ; ce n'est plus un modificateur de
  touche, le +1 de Heavy s'y ajoute). Chaque figurine visée est INFANTRY / BEASTS / SWARM dans une zone de
  terrain (un orteil suffit), ou pas pleinement visible du tireur à cause des décors. Les cartes ne donnent
  pas encore les murs : une figurine **entièrement dans une ruine** est considérée comme masquée par ses murs
  (c'est ainsi qu'un monstre ou un véhicule y a le couvert).
- **Hidden** (13.09), figurine par figurine : INFANTRY / BEASTS / SWARM qui touche une zone de terrain
  contenant un **décor dense** (la carte dit lesquels : D / L sur l'aperçu), si son unité n'a pas fait
  d'attaque à distance ce tour ni au tour précédent. Une figurine cachée n'est visible que des ennemis à
  15" (portée de détection). L'unité reste ciblable dès qu'une de ses figurines est visible — une figurine
  hors décor dense « casse » la protection (l'interface la marque en orange). Un MONSTER / VEHICLE n'est
  jamais caché.
- **Obscuring** (13.10) : toute zone de terrain (dense ou légère) bloque les lignes de vue qui la traversent,
  sauf celles des figurines qui s'y trouvent (le prince derrière la barricade légère T8m reste invisible).
- **Tir indirect** (10.07) : une unité désengagée qui n'a pas fait d'Advance peut viser avec ses armes
  [INDIRECT FIRE] une cible qu'elle ne voit pas ; la cible a le couvert, pas de relance de touche, un jet
  non modifié de 1-5 rate (1-3 si l'unité est restée immobile et qu'une unité amie voit la cible).
- **Allocation** (05.03-05.04) : groupes d'allocation (un par PERSONNAGE, un par W / Sv / InSv pour les
  autres) ; le groupe blessé d'abord, les PERSONNAGES en dernier ; les jets de sauvegarde sont résolus du
  plus faible au plus fort contre le groupe courant (1 naturel rate toujours). [PRECISION] : l'attaquant
  vise un groupe PERSONNAGE visible. [DEVASTATING WOUNDS] : BM égales à D, une figurine au plus par
  blessure critique, après les dégâts normaux. E d'une unité menée = meilleure E des gardes du corps (19.02).
  Ordre des groupes non imposé : heuristique du défenseur (meilleure sauvegarde contre la PA d'abord).
- **Objectifs** (14.01-14.02) : chaque objectif du Layout A **est l'empreinte de décor désignée par la carte**
  (T1, T3, T9, T3m, T1m) ; une figurine le contrôle dès que son socle touche l'empreinte. Un objectif sans
  décor reste un pion de 40 mm avec portée de 3". Contrôle évalué à la fin de chaque phase et de chaque tour.
- **Points de commandement** (08.02) : +1 CP à chaque joueur à chaque phase de commandement (affichés à côté
  des VP), dépensés en stratagèmes de base (15, voir « Service web »). Une unité battle-shocked ne peut pas
  être ciblée par un stratagème (01.07), y compris par Insane Bravery.
- **Drapeaux « ce tour »** (Advance, repli, charge, tir…) : remis à zéro pour les deux camps au début de
  chaque tour — une unité qui a chargé à son tour n'est plus « chargeante » au tour adverse (révision 2 du
  moteur ; les parties enregistrées avant se rejouent avec l'ancien comportement).
- **Battle-shock** (08.03) : test pour chaque unité battle-shocked ou à moitié de son effectif ou moins ; l'état
  persiste jusqu'à un test réussi. **[CLOSE-QUARTERS]** = [PISTOL]. **[HEAVY]** : +1 touche si l'unité est
  désengagée, n'a pas été posée ce tour et qu'aucune figurine n'a bougé de plus de 3".
- **Monstres et véhicules** (validé) : Big Guns Never Tire — un MONSTER / VEHICLE tire même à portée
  d'engagement (toutes ses armes, sur les unités qu'il engage, -1 à la touche hors Pistol) et reste ciblable
  quand il est engagé avec une unité amie du tireur (-1 à la touche hors Pistol) ; Deadly Demise — à sa
  destruction, D6 : sur 6, chaque unité à 6" subit X blessures mortelles ; jets de danger ratés (1-2) → 3 BM
  si toutes les figurines sont M/V, sinon 1 ; sous la moitié de son effectif = PV restants ≤ moitié. Duty
  Eternal (-1 aux dégâts alloués, minimum 1), Lone Operative (dont Lord of Excess : ciblable à 12" ou moins),
  aura Excessive Vigour (+1 PA en mêlée pour une unité Slaanesh qui a chargé, à 6" du prince) et Rapid Fire
  (+X attaques à mi-portée) sont joués.
- **Empreintes** : socle rond, ovale (« stade ») ou coque rectangulaire de véhicule. Une coque pivote
  librement pendant son mouvement. Dans le navigateur, R / Maj + R fait pivoter une coque de 15°.
- **Transports** (18) : débarquer se fait dans un mode imposé — *rapide* si le transport a fait un mouvement
  normal cette phase (entièrement à 3", pas de charge ce tour) ; *tactique* s'il n'a pas bougé et que l'unité
  tient à 3" (puis mouvement normal ou Advance) ; sinon *de combat* (6", un jet de danger par figurine, pose
  possible au contact des ennemis engagés avec le transport, puis battle-shock et pas de charge). Interdit si
  le transport a fait une Advance ou un Fall Back. Embarquer : chaque figurine à 3" du transport après un
  mouvement normal, une Advance ou un Fall Back, pas si l'unité a été posée ce tour. **Transport détruit**
  (18.05) : un jet de danger par figurine, puis chaque figurine entièrement à 6" de l'épave et aussi près que
  possible (sinon détruite) ; battle-shock et pas de charge ce tour ; Deadly Demise ensuite.
- **Scouts X"** (24.31-24.32) : avant le round 1, mouvement de X" au plus, fini à plus de 8" de toute unité
  ennemie ; l'unité doit être entièrement dans sa zone ; un DEDICATED TRANSPORT le fait si toutes les figurines
  à bord ont Scouts (X le plus petit).
- Ligne de vue : vraie ligne de vue socle à socle, échantillonnée ; une ligne doit passer à au moins 0,25"
  du bord d'un décor opaque (hors du premier et du dernier pouce du trait) pour éviter les lignes « au
  rasoir » entre deux décors dont les bords sont alignés. Tout est au sol (pas de hauteurs).
- Table de blessure inchangée ; modificateurs de touche/blessure plafonnés à ±1 ; 1 naturel échoue,
  6 naturel = critique. Blast : +1 A par tranche de 5 figurines.
- Tir : après Advance seules les armes Assault ; après Fall Back ni tir ni charge (sauf Thrill Seekers) ;
  engagé = armes [CLOSE-QUARTERS] seulement, sur une unité engagée. Une unité ennemie à portée d'engagement
  d'une unité amie ne peut pas être visée (sauf par l'unité qui l'engage).
- Mission Unstoppable Force : 3 VP unité détruite, 4 VP par objectif hors home dès le round 2 (fin de
  phase de commandement, fin de tour au round 5), 3 VP objectif nouvellement pris, 5 VP central en fin de
  partie, plafond 15 VP par round hors fin de partie.
- Toy model : table 44"×60" avec le **Layout A** du mission deck 2026-27 (`data/layouts/layout_a.json`,
  aperçu `layout_a.png` : objectifs = empreintes cerclées de couleur), mission **Unstoppable Force**, 5 rounds.

## Écarts restants avec le PDF V11

- **Décors denses et M/V** (13.06) : un MONSTER / VEHICLE ne traverse pas un décor dense de plus de 2" de haut.
  La carte ne donne ni murs ni hauteurs : aujourd'hui les M/V franchissent toutes les empreintes. À ajouter
  au Layout A (murs des ruines), ce qui remplacera aussi l'approximation « entièrement dans une ruine =
  masqué par ses murs » du couvert.
- **Stratagèmes** : les 10 stratagèmes de base sont joués ; Command Re-roll relance les jets d'Advance et de
  charge (toujours proposé sous 12 : on peut vouloir une charge plus longue), pas encore les jets de touche,
  blessure, sauvegarde, dégâts, danger et nombre d'attaques (effet manuel en attendant). Tir d'opportunité :
  les armes [TORRENT] touchent automatiquement (validé). Stratagèmes de détachement : traduits en partie
  (voir plus haut), le reste en effet manuel ; capacités de fiches et règles de détachement : effet manuel.
- **Réserves** : le choix « en réserve » se fait en déployant l'unité, pas dans une étape « Declare Battle
  Formations » séparée et cachée à l'adversaire.
- **Révisions du moteur** : une partie enregistrée se rejoue avec les règles de sa révision (1 : d'origine,
  2 : drapeaux « ce tour » pour les deux camps et stratagèmes de base, 3 : Feel No Pain contre les blessures
  mortelles, stratagèmes de détachement, relance de charge élargie).
- **Choix laissés à une heuristique** : ordre des groupes d'allocation hors contraintes (défenseur),
  [PRECISION] toujours utilisé quand un personnage est visible, répartition des attaques de mêlée
  (chaque figurine frappe la cible choisie si elle l'engage). L'IA pourra en faire de vraies décisions.

## Tests

```bash
python3 -m unittest discover -s tests      # numpy requis
pytest                                     # si installé
```

## Ce que la couche données sait déjà faire

- CSV en `|` avec BOM et colonne vide finale ; HTML des descriptions converti en texte.
- Caractéristiques : `6"`, `3+`, `D6+1`, `2D6`, `N/A` (Torrent), `-`, `20+"`, `4*`.
- Socles : `32mm`, `28.5mm`, `120 x 92mm flying base`, `Use model` (→ inconnu).
- Mots-clés d'arme normalisés malgré la casse aléatoire de l'export (`IGNORES COvER`) :
  nom, valeur (`rapid fire 2`, `sustained hits D3`), cible (`anti-infantry 4+`), condition
  (`lethal hits: non-monster/vehicle`). Les mots-clés hors règles de base sont conservés
  avec `known == False`.
- Coûts V11 par exemplaire (« YOUR 1ST TO 2ND UNITS COST » / « YOUR 3RD + UNIT COSTS »),
  coûts d'équipement, contextes Agents of the Imperium.
- Capacités Core / Faction résolues via `Abilities.csv` (Deep Strike, Scouts 6", Oath of Moment…).
- Leaders : qui peut mener qui.
