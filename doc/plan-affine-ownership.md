# Plan — "piste 1" : n'allouer un header refcompté que si la valeur est réellement aliasée

> **Statut (2026-09-17) : ce plan est terminé, pas actif.** §11-§14 ont atterri, testés par exécution réelle, régression zéro (voir `doc/backlog.md`, l'entrée consolidée sur le sujet). `CLEAVE_AFFINE_STRUCTS` a depuis été **retirée** — le mécanisme est actif par défaut, l'opt-out est `CLEAVE_NO_AFFINE_STRUCTS` (`mlir_lower.rs::lower_program`). Les mentions plus bas de « désactivé par défaut »/« gate » reflètent l'état à l'écriture de chaque section, pas l'état actuel — ne pas les prendre pour une consigne encore valide. Ce fichier reste en place (pas archivé dans `backlog-done.md`, pas supprimé) parce que des dizaines de commentaires réels dans `cleave/src/alias_analysis.rs`, `mlir_lower.rs`, `pipeline.rs`, `cleave-rt/src/lib.rs` et plusieurs fichiers de test le citent nommément comme la justification de leur propre conception (`§11`, `§13`, `§14.x`) — c'est un document de référence pour le code existant, pas une proposition en attente.

## 0. Ce que ce plan doit à ce qui précède

Ce document part de trois faits établis dans cette même session, sur preuve, pas par argument :

1. **Le mécanisme exact de la fuite `b = bump(b)`** ([plan-region-arena.md](plan-region-arena.md) §10, affiné en discussion) : le CPS réel (à N=200000000, jamais exécuté — seul `--dump-cps-optimized` a tourné) ne contient **qu'un seul `release`** dans toute la fonction, dans la branche de sortie de boucle. `refcount.rs::walk_var_info` calcule déjà, à chaque saut, ce qui est vivant (`live_set`), mais ne s'en sert que pour décider quoi *garder* — jamais pour émettre une release à ce saut. La branche `CExpr::App { .. } => {}` de `walk_var_info` ([refcount.rs:1129](../cleave/src/refcount.rs#L1129)) ne marque jamais une valeur de retour comme possédée. Résultat : sur N itérations, une seule valeur est jamais libérée — pas différée, jamais ciblée par aucune instruction.
2. **`escape.rs` (commit `277942b`) répond à une question différente et plus grossière.** Il classe un `CVar` lié par `PrimOp::Struct` comme *escaping* s'il traverse ne serait-ce qu'un saut à l'intérieur de sa fonction englobante — ce qui, en CPS, inclut le `return` (un retour est un saut vers le paramètre-continuation). Donc quasiment toute valeur qui compte un jour comme résultat est *escaping* par ce critère. Ce n'est pas une lacune du module — c'est exactement la question qu'il pose (« a-t-elle besoin d'un tas du tout, ou peut-elle être un temporaire d'arène ») — mais ce n'est **pas** la question « est-elle jamais partagée », et `escape.rs` ne franchit jamais la frontière d'un `App` vers une autre fonction top-level pour la poser.
3. **`region_analysis.rs` est le précédent exact de la forme qu'il faut construire** : un résumé par fonction, propagé à point fixe sur le graphe d'appel — rendu fini et concret par la monomorphisation systématique de cleave, donc garanti de terminer.

## 1. L'énoncé de la piste

Pour toute valeur qui a besoin d'une identité tas (donc pas un temporaire d'arène — `escape.rs` tranche déjà ça) : le refcount n'est nécessaire **que si plus d'une référence vivante à cette allocation peut exister simultanément, n'importe où dans le programme entier.** Sinon, c'est une chaîne de transferts de propriété à un seul propriétaire — pas de header, pas de retain/release runtime, un alloc et une free déterministe, au bon point.

Ceci généralise le Tier 1 déjà écrit dans le plan de mode-plan antérieur (`C:\Users\chris\.claude\plans\ok-je-me-demande-swift-bunny.md`) : ce plan-là éliminait des paires retain/release *déjà insérées*, provablement redondantes. Celui-ci décide, en amont, si le header doit même exister.

**La frontière ne change pas** — elle est déjà correctement posée dans ce plan antérieur : un partage à multiplicité fixe et énumérable statiquement reste décidable sans compteur ; un partage à multiplicité dépendant d'une donnée runtime (ex. `push` dans une boucle à borne runtime) ne l'est pas, et retombe sur le mécanisme actuel, intact.

## 2. Prérequis dur, non négociable — et une correction de conception importante trouvée en le creusant

**Sans le point de release correct, la notion même de "libérer au bon point" n'existe pas.** Une chaîne jamais aliasée a quand même besoin d'un endroit où la `free` a lieu — et c'est exactement ce qui manque aujourd'hui (fait 1 ci-dessus). Stage 0 n'est donc pas une étape de la piste 1, c'est ce qui rend la piste 1 formulable.

### 2.1 Pourquoi un simple patch de plus ne suffit pas — trouvé en cherchant le mécanisme exact

En creusant *pourquoi* `b = bump(b)` fuit, la cause n'est pas « le compilateur ne reconnaît pas un saut de boucle comme un vrai retour » (une hypothèse testée et rejetée : `rewrite_body`'s `CExpr::App` arm traite un saut de boucle et un vrai retour **de façon identique** pour la décision de libérer — `at_true_return` ne sert qu'à un cas de bord étroit, la dé-duplication d'un alias de paramètre, pas à autoriser la release elle-même).

La vraie cause, trouvée par lecture directe de `rewrite_body`'s `Fix` arm ([refcount.rs](../cleave/src/refcount.rs)) :

```rust
if (!is_loop || !is_entry_arg) && !aliases_ret_param && seen.insert(*v) {
    seed.push((*v, ty.clone()));
}
```

`is_loop` (dérivé de `def.carried_types.is_some()`, un marqueur posé dès `cps.rs`) déclenche une exclusion **volontaire** : un argument d'entrée de boucle n'est **pas** transmis comme « possédé » au corps de la boucle. Le motif est compréhensible — une variable portée par une boucle représente un objet *différent à chaque tour*, et le mécanisme de suivi (une marche à passe unique, qui sème `owned` une fois puis le fait descendre) suppose implicitement qu'on parle toujours du même objet. Plutôt que de résoudre ce cas, quelqu'un l'a coupé.

Rien n'a jamais été construit pour reprendre le relais. Le résultat frais que produit chaque tour (ici, le retour de `bump`) n'est de toute façon jamais marqué possédé nulle part (`CExpr::App { .. } => {}` dans `walk_var_info`) — donc même sans l'exclusion, il ne serait tracé par rien. Les deux trous se recouvrent exactement sur la variable portée par la boucle.

**Le point important, qui change la portée de ce Stage** : la marche à passe unique actuelle n'est structurellement pas le bon outil pour un cycle. Une boucle (ou une récursion) pose une question dont la réponse dépend d'elle-même — « cette valeur porte-t-elle encore » dépend de ce qui se passe au tour suivant, qui dépend de la même question. Une passe descendante unique ne peut résoudre ça correctement ; elle ne peut que la couper à la main quand le cas se présente — exactement ce que `is_loop`/`is_entry_arg` fait ici, et exactement le genre de rustine ponctuelle que ce plan entier existe pour arrêter d'empiler (`aliases_ret_param`, `param_leaf_key` sont deux autres compensations du même genre, déjà accumulées dans le même mécanisme, pour des variantes voisines du même problème).

**Un point fixe résout un cycle par construction, sans avoir besoin de savoir que c'en est un.** Ni annotation `is_loop`, ni cas particulier pour la récursion mutuelle : la même procédure de convergence traite les trois de façon uniforme, parce qu'elle itère jusqu'à stabilité au lieu de descendre une seule fois. Ce n'est pas « zéro information structurelle utilisée » — un `Fix` qui s'appelle lui-même reste visible dans le CPS, exactement comme `region_analysis::analyze` découvre déjà les cycles du graphe d'appel sans qu'on le lui dise. C'est l'absence d'une **règle écrite à la main** qui traite les boucles différemment des appels ordinaires.

**Conséquence pour la portée de ce Stage — révisée** : patcher une règle `App` de plus dans la marche existante referme la fuite mesurée, mais très probablement en réinventant sa propre exclusion `is_loop`, ce qui empile un hack de plus au lieu d'en retirer un. **Stage 0 et Stage 1 (§3) ne sont donc pas deux mécanismes séparés à construire l'un après l'autre — c'est le même calcul.** Un point fixe unique, sur tout le programme, répondant à la fois à « cette valeur est-elle encore vivante » (ce que fait `live_set`/`releases_for_app` aujourd'hui, correctement, mais seulement au sein d'une passe unique) et « cette valeur est-elle jamais dupliquée » (§3), remplace la marche actuelle et son empilement de cas particuliers.

### 2.2 Ce que ça change concrètement

- Le module de §3 (`alias_analysis.rs`) n'est plus « à construire après Stage 0 » — il **est** Stage 0, étendu pour aussi porter la question de vivacité, pas seulement celle du partage.
- Le résumé par fonction n'est plus seulement `aliases(f, i): bool` — il porte aussi, pour chaque `Fix`-def de `f` (boucle ou continuation locale), l'ensemble des `CVar` encore vivantes à chaque saut sortant, calculé à point fixe plutôt que semé une fois.
- L'exclusion `is_loop`/`is_entry_arg` de `rewrite_body` devient inutile et doit être retirée une fois le point fixe en place — pas gardée « au cas où » à côté.

**Vérification, sans jamais exécuter un programme non borné** ([mémoire](../../.claude/projects/i--Dev-cleave/memory/feedback_kill_unbounded_repro_processes.md)) :
- `--dump-cps-optimized` sur `b = bump(b)` doit montrer une `release` **dans la branche "continue"** de la boucle, ciblant l'ancien `b` — pas seulement dans la branche de sortie. Lecture statique, zéro exécution.
- Si une mesure d'exécution est vraiment nécessaire : une borne **finie et petite** (ex. 1000 itérations), lancée au premier plan (pas en arrière-plan, pas de boucle de sondage), qui se termine seule en une fraction de seconde — jamais le reproducteur non borné.
- `cargo test -p cleave --release --no-fail-fast` vert, y compris `a_struct_passed_to_a_genuinely_identity_shaped_function_is_released_only_once` (le cas d'alias par identité, qui doit continuer à passer).
- Un test qui construit une récursion mutuelle entre deux fonctions top-level portant chacune une valeur refcomptée, pour confirmer que le point fixe converge et libère correctement sans qu'aucun `is_loop` n'ait été nécessaire pour ce cas — la généralisation que ce Stage promet, pas seulement le cas boucle `for`.

**Cette étape referme la fuite mesurée à elle seule**, indépendamment de tout le reste de ce plan — elle mérite d'être mesurée et acceptée séparément.

### 2.3 Statut réel — toujours une verrue, pas la solution générique décrite ci-dessus

Confirmé explicitement en revue : ce qui a effectivement refermé la fuite `b = bump(b)`, c'est §12 (`analyze_identity`) — un point fixe réel, mais qui répond à une question étroite et spécifique (« cette fonction retourne-t-elle ce paramètre inchangé ») typée à `println`/`Print::print`, pas le point fixe unique « vivacité + aliasing, sur tout le programme » que §2.1/§2.2 décrivent. **`is_loop`/`is_entry_arg` (`refcount.rs::rewrite_body`) n'a jamais été retiré** — il tourne toujours, intact, à côté du reste. §2.2 le dit noir sur blanc : « devient inutile et doit être retiré une fois le point fixe en place » — ce point fixe-là n'est pas en place. Ce Stage reste donc une aspiration documentée, pas un travail terminé — à ne pas confondre avec §3/§11.4 (l'analyse d'aliasing elle-même, elle, généralisée et livrée) ni avec §12 (l'identité, livrée aussi, mais pour une question plus étroite).

**Priorité explicite, décidée en revue** : §13 (la cascade sans header pour le pool allocator) passe devant la généralisation de ce Stage — §13 a un impact direct et mesurable sur la performance du benchmark de référence ([[project_parallelization_is_the_viability_test]], le go/no-go de toute l'approche native de cleave), alors que finir §2 fermerait une verrue existante sans qu'aucun bénéfice de performance n'en découle directement — la fuite qu'elle visait à l'origine est déjà fermée par ailleurs (§12).

## 3. Stage 1 — l'analyse d'aliasing interprocédurale, maintenant unifiée avec Stage 0 (lecture seule, aucun changement de codegen tant que §5 n'est pas atteint)

Nouveau module, `cleave/src/alias_analysis.rs`, même forme que `region_analysis::analyze` : un point fixe monotone sur `HashMap<(String, usize), bool>` (nom de fonction, index de paramètre) → « ce paramètre est-il jamais aliasé quand cette fonction est appelée ».

**Règles locales par fonction (aucune connaissance interprocédurale requise pour celles-ci)** :
- Le paramètre `p_i` est la cible d'un `Retain` déjà inséré par `refcount.rs` (les deux sites existants : embarquement protégé, lecture de champ à durée de vie indépendante) → `aliases(f, i) = true`. On réutilise le marquage existant comme vérité terrain, on ne le redérive pas.
- `p_i` est passé en argument à un callee `g` en position `j`, et `aliases(g, j) = true` (propagation transitive) → `aliases(f, i) = true`.
- `p_i` est retourné **inchangé** par `f` (identité pure) : **ceci ne marque pas `aliases(f, i)`.** Le partage, dans ce cas, existe seulement si l'appelant garde sa propre copie vivante après l'appel — et c'est déjà exactement ce que `refcount.rs` détecte aujourd'hui pour insérer le retain compensatoire côté appelant (le test identité déjà cité). Pas de nouvelle règle interprocédurale à inventer ici ; le fait est déjà capturé côté appelant, à ne pas dupliquer côté callee sous peine de faux positifs systématiques sur toute fonction passthrough.
- Point fixe standard : tout part à `false`, on itère jusqu'à stabilité — le domaine est fini (programme monomorphisé), donc la terminaison est garantie, comme pour `region_analysis`.

**Tests, avant tout changement de codegen** — mirroir du style `region_analysis.rs` :
- Une chaîne jamais aliasée (`bump`-like) → `false`.
- Un embarquement protégé → `true`.
- Une fonction identité pure → **ne marque pas** son propre paramètre (le cas à ne pas sur-approximer).
- Une fonction récursive/mutuellement récursive → le point fixe converge, résultat correct.

Diagnostic assorti : `CLEAVE_TRACE_ALIASES=1`, même format que `CLEAVE_TRACE_REGION_LOCAL`.

## 4. Stage 1.5 — finir le branchement d'`escape.rs` lui-même (dette déjà identifiée, indépendante de piste 1 mais sur le même chemin)

`escape.rs` calcule déjà *temp vs slot*, mais **rien ne consulte ce résultat dans `mlir_lower.rs::alloc_llvm_value`** — vérifié directement (`grep escaping_struct_vars cleave/src/mlir_lower.rs` → rien). Aujourd'hui il n'alimente que l'élargissement de `rc_opt::eliminate_redundant_retain_release` à travers un `Fix`. Le brancher sur le choix d'allocateur est un prérequis propre pour Stage 2 (qui a besoin des trois classes — temp / slot-partagé / slot-affine — au même point de décision), et c'est un morceau petit, isolé, testable seul.

## 5. Stage 2 — consommer le verdict : allocation sans header pour les slots jamais aliasés

Pour une construction `PrimOp::Struct` classée *slot* (escape.rs) et dont chaque paramètre en aval est prouvé `aliases(...) = false` sur toute la chaîne : router vers une nouvelle paire runtime **sans header** — pas de `RcHeader`, pas de `retain`/`release` atomiques, un alloc et une free unique au point calculé par Stage 0.

**Risque à nommer explicitement** : si l'analyse se trompe dans le sens optimiste, ce n'est plus une fuite — c'est une corruption silencieuse ou un double-free (un pointeur sans header qui atteint un `retain`, ou un `is_in_arena`/cascade générique qui suppose un header). C'est la classe d'erreur la plus chère qui existe dans ce projet — c'est littéralement `8a748f8`. En conséquence :
- Départ derrière un env var (`CLEAVE_AFFINE_STRUCTS`, même schéma que `CLEAVE_STRICT_REGIONS`), jamais par défaut avant mesure complète.
- Pas d'élimination de code mort agressive au premier passage — préférer sur-classer *slot-partagé* (comportement actuel, sûr) en cas de doute, jamais l'inverse.

## 6. Stage 3 — mesure, protocole déjà écrit ailleurs, à ne pas réinventer

Même quatre axes que tout ce qui a précédé dans cette session : `cargo test --release --no-fail-fast` vert, mémoire sous [§6 du plan région/arène](plan-region-arena.md), accuracy inchangée, wall-clock remis à l'utilisateur pour sa propre mesure. Boucles courtes ([mémoire](../../.claude/projects/i--Dev-cleave/memory/feedback_short_experiment_loops.md)), baseline prise avant tout rejet ([mémoire](../../.claude/projects/i--Dev-cleave/memory/feedback_measure_against_a_baseline.md)), jamais de reproducteur non borné laissé sans supervision ([mémoire](../../.claude/projects/i--Dev-cleave/memory/feedback_kill_unbounded_repro_processes.md)).

## 7. Ce que ce plan ne fait délibérément pas — pour ne pas répéter `8a748f8`

- **Ne touche pas aux temporaires introduits par le lowering MLIR** (l'accumulateur de réduction) — c'est l'axe `--promote-buffers-to-stack` / émission destination-passing, orthogonal, déjà noté dans `backlog.md`.
- **Ne rouvre pas automatiquement le rêve "light struct = type valeur"** — prouver "jamais aliasé" ne prouve pas "n'a jamais besoin d'une adresse stable" (une valeur peut être seule propriétaire et pourtant avoir son pointeur pris ailleurs). C'est un pas de plus, distinct, à ne pas confondre avec celui-ci.
- **Une seule chose par commit.** `8a748f8` a soudé un vrai correctif, un élargissement `is_rc` non sûr, et une passe de compensation dans un même commit de 2000 lignes — trois jours de diagnostic en ont payé le prix. Stage 0 seul, mesuré et accepté seul, avant Stage 1 ; Stage 1 seul (lecture seule, zéro risque de codegen) avant Stage 2 ; Stage 2 seul, derrière un gate, avant tout défaut.

## 8. Ordre des dépendances, explicite — révisé après §2.1

```
Stage 0+1, unifiés (point fixe : vivacité + aliasing) ──┐
                                                          ├──> Stage 2 (alloc sans header) ──> Stage 3 (mesure, rollout gated)
Stage 1.5 (brancher escape.rs) ─────────────────────────┘
```

Stage 0 et Stage 1 ne sont plus deux chantiers indépendants (§2.1) — c'est un seul point fixe à construire, qui répond aux deux questions à la fois. Stage 1.5 reste indépendant, nécessaire à Stage 2 ; Stage 2 a besoin des deux.

## 9. Bénéfice en aval, hors scope pour l'instant — ce point fixe isole déjà "partagé vs privé" pour une future parallélisation

Remarque faite en discussion, qui renforce l'intérêt de bien construire le point fixe plutôt que de le voir comme un coût isolé pour cette seule piste.

**Une boucle, en CPS, est déjà une vraie fonction avec une interface explicite.** `loop$34 (v469 v468)` : les seuls noms qui traversent d'un tour à l'autre sont les paramètres/`carried_types` du `CFunDef`. Tout ce qui est construit à l'intérieur du corps et n'apparaît pas dans cette liste est privé à ce tour-là, par construction — pas à redériver.

**La même classification "ne s'échappe jamais d'un tour" sert déjà, ou servirait, trois consommateurs distincts** :
- `region_analysis` (existant) : arène vs tas.
- Ce plan (piste 1) : header vs pas de header.
- Une future parallélisation : privé-par-thread vs partagé-séquentiel — même prédicat.

**Ce que ça donnerait de neuf, concrètement** : aujourd'hui, quand OpenMP sort une boucle en régions parallèles, ça se joue au niveau MLIR, sans aucune visibilité sur ce que le CPS savait déjà de la propriété — d'où le verrou (`POOL_LOCK`) sur `cleave_alloc_rc`/`cleave_release`, une précaution générale plutôt qu'une décision informée. Avec le point fixe de ce plan, la question "qu'est-ce qui est réellement partagé ici" serait tranchée **avant** la sortie en MLIR, à l'endroit où l'information existe déjà.

**Ce que ça ne donne PAS, pour ne pas survendre** : le partitionnement (quoi est partagé) n'est pas la réponse à "cette dépendance portée est-elle parallélisable". `b = bump(b)` est le contre-exemple exact : `b` est porté donc partagé par cette classification, mais chaque tour dépend strictement du résultat du précédent — ni parallélisable, ni une réduction. Reconnaître qu'une dépendance portée *est* une réduction (associative/commutative) est une question séparée, plus dure (reconnaissance de motif de réduction / analyse de dépendance de boucle) — mais elle ne peut même se poser qu'une fois qu'on sait *quoi* est porté, ce que ce point fixe donne gratuitement.

**Hors scope pour ce plan.** Rien à construire maintenant pour ça — juste un argument de plus, avec [[project_parallelization_is_the_viability_test]] en tête, pour investir correctement dans le point fixe plutôt que de le voir comme jetable une fois piste 1 mesurée.

## 10. Frontière FFI (extern/export) — un invariant de type, pas une conséquence d'aliasing

Question soulevée en discussion : comment garantir qu'une valeur qui traverse la frontière `extern`/`export` garde toujours son header, une fois que piste 1 sait allouer sans header quand c'est sûr.

**Vérifié directement, pas supposé** : le struct-export n'est pas encore implémenté — `cleave.exe` refuse explicitement de compiler un `export fn` dont un paramètre/retour est un struct (« struct/array/tensor export bindings aren't implemented yet »). Rien à corriger en urgence — mais la règle doit être posée **avant** que ce chantier démarre, pas après.

**Le principe** : le risque à cette frontière n'est pas « cette valeur précise est-elle aliasée » — `alias_analysis` répond déjà correctement « inconnu, donc aliasé » pour tout appel extern (règle 4), et c'est déjà suffisant pour ce cas précis. Le vrai risque est plus large : **du code Rust/C, entièrement invisible au compilateur, peut retenir ce pointeur indéfiniment et appeler retain/release dessus à son propre rythme.** C'est une propriété du *type*, pas d'un site d'appel — donc ça doit être un ensemble de noms exclu une fois pour toutes, jamais une conséquence d'un calcul d'aliasing par valeur.

**Le précédent existe déjà et marche** : `collect_extern_boundary_struct_names` (`refcount.rs`) — tout struct nommé au point d'un appel `extern fn` (côté cleave appelant dehors) est déjà exclu de l'optimisation « light » via `is_light_struct`. Exactement le patron à réutiliser, pas à réinventer.

**À faire, au moment où le struct-export sera construit, pas avant** :
1. Étendre le même collecteur (ou lui adjoindre un second ensemble consulté aux mêmes points) pour aussi capturer tout struct nommé au paramètre/retour d'un `CTopLevelFn` dont `is_export` est vrai — le champ existe déjà, threadé partout (`cps.rs`), jamais consulté pour une décision de représentation aujourd'hui.
2. Ce même ensemble doit exclure de **deux** choses, pas une : l'optimisation « light » existante (déjà correct) **et** l'allocation sans header de piste 1 (Stage 2) — un seul ensemble, deux consommateurs, pas deux listes à tenir synchronisées.
3. Le glue Rust généré par `cleave-build` pour un tel type peut alors supposer, sans jamais le vérifier à l'exécution, qu'un header existe — l'invariant ci-dessus le garantit à la compilation. Recommandé pour le premier jet : **move-only côté Rust** (pas de `Clone`, un seul `Drop` appelant `cleave_release`) — pas de `cleave_retain` exposé en FFI tant que ce n'est pas nécessaire, pour ne pas rouvrir côté Rust exactement la question « qui retient, quand » que cette session a mis trois jours à cerner côté cleave.

## 11. Stage 2, premier essai réel — un vrai bug de conception trouvé par le test, pas encore corrigé

Construit et testé jusqu'au bout : `cleave_alloc_pool`/`cleave_release_pool` (runtime, `cleave-rt`, testés), `alias_analysis::affine_struct_vars` (décide quelles constructions sont éligibles), le branchement dans `alloc_llvm_value` et dans l'émission de `PrimOp::Release`, derrière `CLEAVE_AFFINE_STRUCTS=1` (désactivé par défaut). La suite complète passe, gate désactivée : 833/833, aucune régression.

**Avec la gate activée, sur le cas réel `b = bump(b)` : `STATUS_HEAP_CORRUPTION` confirmé, pas une instabilité de test.**

### 11.1 Le mécanisme exact

`refcount::insert_refcounting` insère la release sur `v469` — le **paramètre porté par la boucle** (`loop$33`'s own second param) — jamais sur le site de construction lui-même. C'est correct et voulu : c'est exactement le point que Stage 0 a corrigé plus tôt dans cette session.

Mais `affine_struct_vars` ne classe que les sites `PrimOp::Struct` — `v469` n'en est jamais un, ce n'est qu'un paramètre. Donc `ctx.affine_structs.contains(v469)` est toujours faux, et l'émission de release pour `v469` tombe systématiquement sur `cleave_release` ordinaire (avec header).

Le problème : **selon le tour de boucle, ce que `v469` contient physiquement change d'origine**. Au premier tour, c'est `v468` (construit avant la boucle) — correctement jugé aliasé par mon analyse (passé comme argument d'entrée à `loop$33`, dont le nom n'est jamais dans `AliasSummary` puisque ce n'est pas une vraie fonction top-level — retombe donc, à raison, sur « inconnu → aliasé »). Header réel, cohérent avec `cleave_release`. À partir du second tour, `v469` contient le résultat de `bump` — inliné plus tard dans le pipeline (`--inline`, au niveau MLIR, après que `insert_refcounting`/`alias_analysis` ont déjà tranché) — correctement jugé **jamais aliasé du point de vue de `bump` seul**, donc alloué sans header via `cleave_alloc_pool`.

**Un seul site de release textuel, deux origines physiques différentes selon le tour — et la release ne peut pas savoir laquelle elle a en face d'elle.** `cleave_release(v469)` lit un header qui existe au premier tour et n'existe pas ensuite. Corruption confirmée, isolée par exécution réelle (pas un dump statique) sur un cas borné (`0..2`, JIT, verrouillé par mutex) — exactement la discipline de test que ce plan a établie depuis le début.

### 11.2 Pourquoi ce n'est pas patchable en surface

Le vrai défaut : **mon analyse classe par site de construction (`v465`/`v468`), mais la décision d'allocateur doit tenir au niveau de l'identité de valeur qui traverse un paramètre de boucle ou un retour d'appel** — exactement le même problème structurel que Stage 0 a résolu pour la question « faut-il libérer », mais cette fois pour la question « quel allocateur ». `walk_var_info`'s own Fix arm (`refcount.rs`) sait déjà propager une propriété (`owned_origin`) à travers un paramètre de boucle, en regardant si l'argument d'entrée ET l'argument du dos-de-boucle sont tous deux possédés. La même propagation manque pour « est-ce que ce paramètre est **toujours** backé par le pool, sur tous les chemins qui l'alimentent ».

### 11.3 Première extension réelle — les paramètres de résomption d'un vrai appel, et un piège évité en le faisant

Fait cette session, réutilisant directement `alias_analysis.rs`'s propre discipline de point fixe (`propagate()`, déjà partagé par `analyze`/`analyze_identity`) : `affine_struct_vars` classe désormais aussi le **paramètre de résomption d'un vrai appel** (`Fix{defs:[def à un seul paramètre, carried_types: None], body: App{Label(callee), ..}}` — exactement la forme que `analyze_identity::collect_identity_facts` reconnaît déjà) comme éligible au pool, dès lors que `callee` retourne **toujours** une valeur déjà affine sur tout chemin de retour tracable (`collect_return_vars`, un nouveau petit helper : poison — jamais affine — dès qu'un retour ne se réduit pas à une seule `CVar` tracée). Un point fixe, pas une seule passe : une fonction peut forwarder tel quel le résultat d'un autre appel affine, en chaîne.

**Le piège trouvé en le faisant, pas anticipé** : `refcount::insert_refcounting` relâche **directement** le paramètre de résomption d'un vrai appel dès que son résultat n'est pas ensuite reporté par un `Fix` de boucle/jonction — un cas réel, différent de celui décrit en §11.1/11.2 (qui, lui, concerne un paramètre *porté par une boucle*). Mais `region_analysis::find_region_local_functions` peut classer le *callee* lui-même comme *region-local* (site d'appel unique, résultat confiné à une seule itération) — auquel cas sa propre construction est allouée dans l'**arène** (`cleave_alloc_local`), jamais dans le pool, quoi que dise `affine_struct_vars` sur elle en isolation. `affine_struct_vars` a donc été corrigée pour **ignorer entièrement le corps de toute fonction region-local** — trouvé par exécution réelle (`many_short_lived_affine_constructions_run_correctly`, `cleave/tests/affine_pool_alloc.rs`, où `bump` a exactement ce profil), pas par relecture du code a priori.

**Une fausse alerte en chemin, corrigée avant de conclure** : le diagnostic initial (`CLEAVE_DEBUG_POOL: popped block ... was not marked parked`) a d'abord semblé indiquer un second bug, *non gardé par aucun flag* — un pointeur d'arène relâché via le `cleave_release` ordinaire. Faux : `cleave_release` a déjà son propre garde-fou (`is_in_arena`, `cleave-rt/src/lib.rs`), qui détecte ce cas exact et court-circuite le free-list — vérifié directement via `CLEAVE_TRACE_SIZE`, qui montre `in_arena=true` géré correctement. Le message restant est un vrai, mais inoffensif, trou de diagnostic pré-existant : `cleave_release_pool` n'appelle jamais `parked_insert`, donc *tout* bloc légitimement échangé entre `cleave_alloc_rc` et le pool (une fonctionnalité réelle et déjà testée, `pool_and_rc_allocations_freely_interchange_the_same_size_class_blocks`) déclenche ce message au pop, corruption réelle ou non. Un nouveau test dédié (`region_local_result_released_by_caller_with_the_gate_off`, gate désactivée, forte pression de réutilisation) confirme qu'aucun vrai signal de double-release n'apparaît jamais dans ce cas.

`many_short_lived_affine_constructions_run_correctly` n'est donc plus `#[ignore]` — les deux correctifs ci-dessus suffisent à le rendre correct, vérifié par exécution réelle et par `CLEAVE_DEBUG_POOL`.

### 11.4 Le cas porté par une boucle — root-cause complète, puis corrigé

`bump_shaped_loop_runs_correctly_through_the_pool_allocator` reproduisait exactement le mécanisme de §11.1/11.2 — reconfirmé cette session par exécution réelle (`STATUS_HEAP_CORRUPTION`), **à une borne mise à jour** : l'e-graph sait maintenant replier entièrement `0..2` à la compilation (zéro variable affine, boucle disparue — vérifié via `CLEAVE_TRACE_AFFINE_STRUCTS`), ce qui avait rendu le test original silencieusement caduc (un `ok` qui ne prouvait plus rien). `0..2000` résiste au repliement et a reproduit le crash tel quel, avant correctif.

**Pourquoi §11.3 seul ne suffisait pas** : la règle envisagée alors (« un paramètre porté est éligible seulement si toutes ses sources — argument d'entrée ET chaque dos-de-boucle — le sont aussi ») était correcte, mais **l'argument d'entrée (`v468`, la construction avant la boucle) ne pouvait, jusque-là, structurellement jamais être jugé affine** : `value_is_ever_aliased` traitait tout passage à un label *local* (boucle ou jonction `if`, jamais dans `AliasSummary` puisque ce n'est pas une vraie fonction top-level) comme « inconnu → aliasé », sans jamais regarder ce que ce label fait réellement de son paramètre. Une règle sûre en soi (aucun faux négatif), mais qui, appliquée telle quelle, fermait **par construction** toute boucle porteuse à cette optimisation.

**La vraie extension, faite cette session, discutée d'abord** : question posée directement à ce moment — « c'est bien la cible avoir alias analysis qui s'occupe même des boucles avec un point fixe, et d'ailleurs... les boucles doivent être en CPS aussi, pourquoi c'est pas déjà le cas ? » — clarification faite que les boucles **sont** déjà en CPS (un `Fix{defs:[CFunDef{carried_types:Some(..)}], body}`, structurellement identique à un vrai appel avec sa résomption — §2.1/§9 l'établissaient déjà) ; ce qui manquait était uniquement dans `alias_analysis::analyze()` elle-même, pas dans la représentation. Fait :

1. `analyze()`/`collect_facts` traite désormais **tout `CFunDef` nommé — local ou top-level — comme sa propre unité analysée**. `collect_local_defs` (nouveau) découvre chaque def `Fix` local à n'importe quelle profondeur et lui fait tourner `collect_facts` avec ses propres paramètres et son propre corps, exactement comme `analyze()` le fait déjà pour chaque fonction top-level. La distinction `known_functions.contains(callee)` dans le bras `App` de `collect_facts` a été **retirée entièrement** — tout `CVal::Label` nommé produit désormais une arête vers le résumé de sa cible, résolue par le même point fixe, jamais un seed « inconnu → aliasé » (cette dernière règle ne s'appliquait de toute façon jamais à un vrai `extern fn`, qui n'est jamais un `CExpr::App` — seulement `PrimOp::Extern`, déjà géré séparément). Une généralisation, pas une exception de plus : elle retire une distinction artificielle plutôt que d'en ajouter une. Vérifié : les 34 tests existants (`alias_analysis.rs` + `refcount.rs`) passent sans changement, et `v468` (l'argument d'entrée de la boucle) apparaît désormais correctement dans l'ensemble affine (`CLEAVE_TRACE_AFFINE_STRUCTS`, avant même la seconde partie ci-dessous).
2. `affine_struct_vars` classe désormais aussi le **paramètre porté par un `Fix` local** (boucle ou jonction `if`, `carried_types.is_some()`) — éligible une fois que *toutes* ses sources (l'argument d'entrée, et chaque argument de dos-de-boucle, où qu'ils se trouvent textuellement dans la même fonction top-level) sont déjà affines. `collect_carried_param_facts` extrait, en un seul passage par fonction top-level, chaque appel `App{Label(nom), args}` et chaque def `Fix` local porteur ; `collect_affine_carried_params` fait l'intersection (AND, pas OR — à la différence de `propagate()`'s propre union), dans le même point fixe partagé que la propagation par résomption de §11.3 (un dos-de-boucle peut lui-même être alimenté par un paramètre de résomption qui ne devient affine que dans cette même passe).

**Vérification, par exécution réelle, pas seulement statique** : `bump_shaped_loop_runs_correctly_through_the_pool_allocator` (`0..2000`, la borne qui résiste au repliement) — **5 exécutions consécutives sous `CLEAVE_DEBUG_POOL=1`, zéro warning de toute nature**, y compris le bruit inoffensif que `many_short_lived_affine_constructions_run_correctly` montre encore (cette boucle-ci ne mélange plus jamais `cleave_alloc_rc`/`cleave_alloc_pool` pour la même valeur — tout le cycle de vie de `b`, boucle après boucle, est affine de bout en bout). Suite complète du workspace : verte, gate désactivée par défaut (`cargo test --release --no-fail-fast`).

`CLEAVE_AFFINE_STRUCTS` reste désactivé par défaut malgré tout — les deux bugs confirmés (§11.1/11.2 et le cas résomption de §11.3) sont fermés, mais la mesure Stage 3 (impact réel sur `mnist-interop`, cf. §6) n'a pas encore été refaite avec la gate activée ; l'activer par défaut est une décision séparée, à prendre après cette mesure.

**Ce qui a été gagné** : le runtime (`cleave_alloc_pool`/`cleave_release_pool`), l'analyse d'aliasing unifiée (locale et top-level dans un seul point fixe), la propagation par résomption d'appel, l'exclusion region-local, et maintenant la propagation par paramètre porté — les trois bugs confirmés cette session (résomption/region-local, et boucle porteuse) sont clos, testés par exécution réelle, zéro régression sur 833+ tests.

## 12. Les verrues extirpées — `alias_analysis::analyze_identity` remplace `refcount::collect_identity_param_positions`, et corrige au passage un vrai double-release réel (`mnist-interop`)

Suite directe de la demande explicite : remplacer les mécanismes ad hoc de `refcount.rs` par la même discipline de point fixe déjà posée en §3 (`propagate()`, un seul walk puis un worklist, consultation en O(1)) plutôt que d'empiler un second mécanisme parallèle à côté du premier.

### 12.1 Le bug réel trouvé en le faisant

`insert_refcounting` protège une valeur transférée de la release si la fonction qui la reçoit est « identity-shaped » (`fn(x) -> x`, exactement la forme de `println(x) -> x`, un `extern`). L'ancien mécanisme (`refcount::collect_identity_param_positions`/`find_identity_returns`) ne détectait que la forme **directe** : le corps de la fonction retourne littéralement, par nom, son propre paramètre.

`println<(A,B)>` (stdlib `io`) ne retourne pas directement son paramètre — il appelle `Print::print<(A,B)>` (qui, elle, est réellement `fn(x) -> x`) et retourne **le résultat de cet appel**, une `CVar` différente de son propre paramètre. La forme est transitivement identity-shaped, mais pas directement — invisible pour un walk à une seule passe.

Conséquence réelle, pas hypothétique : dans `examples/mnist-interop/src/kernel.cleave`, `println(("Epoch=", epoch))` (ligne 247) construit un tuple 12 octets, le passe à `println`, qui elle-même le passe à `Print::print`. `println` n'étant pas reconnue comme identity-shaped par l'ancien mécanisme, `insert_refcounting` insérait une seconde release sur l'argument déjà transféré et relâché par `Print::print` — double-release, confirmé par `CLEAVE_DEBUG_POOL=1` (« release on parked block », site fautif `train_and_evaluate @ kernel.cleave:247`), puis reproduit en clair par `STATUS_ACCESS_VIOLATION` en AOT quelques secondes après `Epoch=0`.

### 12.2 Le mauvais réflexe évité

Le premier réflexe (avant d'être corrigé) était de construire un **second** point fixe séparé, propre à `refcount.rs`, pour détecter l'identité transitive — dupliquant `propagate()` sans le réutiliser. Question posée directement à ce moment : « en fait j'ai du mal a comprendre pourquoi ca n'est pas bien traité par ce qu'on a déja dude ? » — juste : il n'y avait aucune raison structurelle d'avoir deux mécanismes de point fixe côte à côte pour deux questions de même forme (« est-ce que cette position est aliasée » / « est-ce que cette position est retournée inchangée »). Le bon geste : ajouter un second `analyze_*` dans `alias_analysis.rs`, réutilisant le `propagate()` déjà existant, exactement comme `analyze` (aliasing) le fait déjà.

### 12.3 Ce qui a été fait

- `alias_analysis::IdentitySummary`/`analyze_identity(program) -> IdentitySummary` : même schéma en deux passes que `analyze` — `collect_identity_facts` (un seul walk, règle directe inchangée + nouvelle règle transitive : un `Fix` dont le corps de résomption appelle directement une fonction connue et en retourne le résultat sans transformation ajoute une **arête**, pas un fait direct), puis `propagate()` (le même worklist que Stage 1, sans aucune duplication).
- `refcount::RefcountCtx` consulte désormais `identity_summary: &IdentitySummary` (méthode `returns_unchanged(fn_name, param_index) -> Option<bool>`) au lieu de l'ancien champ `identity_param_positions: &HashMap<...>`.
- `collect_identity_param_positions`/`find_identity_returns` **supprimées** de `refcount.rs` — remplacées par un simple pointeur de commentaire vers `alias_analysis::analyze_identity`.

### 12.4 Vérification

- Régression dédiée ajoutée : `a_wrapper_around_a_genuinely_identity_shaped_function_is_also_identity_shaped` (`cleave/tests/refcount.rs`) — reproduit la forme exacte `wrapper(y) -> identity(y)`, exécute réellement le programme (`run_i32`), et vérifie qu'une seule release est insérée pour l'allocation partagée entre `a`/`r` (pas deux, pas zéro).
- Suite complète du workspace : `cargo test --release --no-fail-fast` — tout vert, aucune régression (y compris les deux tests d'identité déjà existants dans `refcount.rs`, toujours verts avec le nouveau mécanisme).
- Vérification end-to-end réelle, pas seulement au niveau test : rebuild AOT forcée de `examples/mnist-interop` (`touch build.rs` + `cargo build --release`, « Compiling mnist-interop » confirmé), run complet des 10 epochs — plus de crash, `test accuracy: 0.9342` (valeur de référence déjà connue), sortie propre (`EXIT_CODE=0`).

## 13. Pourquoi Stage 2, même correct, ne change rien à `mnist-interop` — trouvé après coup, pas anticipé

Les trois bugs de §11 fermés, gate activée sur le vrai kernel (`CLEAVE_AFFINE_STRUCTS=1 CLEAVE_TRACE_AFFINE_STRUCTS=1`, `examples/mnist-interop/src/kernel.cleave`) : **9 constructions** jugées éligibles au pool sur l'ensemble du programme. Pas zéro — mais loin des structs qui comptent.

**La raison est structurelle, pas un oubli** : `Network`, `Dense`, `NetworkState` (et toute la hiérarchie de couches) embarquent chacun au moins un champ `Tensor`. `affine_struct_vars::struct_has_no_cascade_fields` — la seconde des deux conditions requises, à côté de « jamais aliasé » — exclut explicitement tout struct dont un champ (ou le struct lui-même) est tensor-taggé, précisément parce qu'aucune histoire de cascade n'existe encore pour une release sans header (`mlir_lower.rs::lower_release_cascade` ne sait cascader qu'à partir d'une release *avec* header aujourd'hui). C'est la restriction que §5/§11.2 documentaient déjà comme « ce premier atterrissage, pas une limite fondamentale de l'analyse » — confirmé ici : ce n'est pas un manque de couverture de l'analyse (elle marche, elle est même correcte sur boucle portée depuis §11.4), c'est une frontière volontaire posée à la construction du mécanisme.

**Conséquence directe pour §6 (Stage 3, mesure)** : mesurer l'impact de Stage 2 *sur ce kernel précis, tel qu'il est aujourd'hui* ne dirait rien d'utile sur la valeur du mécanisme — les 9 sites éligibles sont nécessairement de petits structs scalaires ailleurs dans le pipeline, pas le chemin chaud de l'entraînement. Refaire cette mesure n'a de sens qu'après avoir construit la cascade sans header (ci-dessous) — pas avant.

**Ce que ça voudrait dire de construire, si un jour ça vaut le coût mesuré** : donner à `cleave_release_pool` la même capacité de cascade que `lower_release_cascade` a déjà pour une release avec header — sauf qu'un bloc du pool n'a pas de header pour retrouver son propre type à l'exécution ; le type (donc la disposition des champs à cascader) doit être **entièrement résolu à la compilation**, exactement comme `cleave_alloc_pool`/`cleave_release_pool` résolvent déjà `data_size` sans jamais le lire depuis la mémoire. `affine_struct_vars` sait déjà, statiquement, quels champs existent (`struct_field_types`) — la cascade recursive existerait donc au même endroit, generée au site de release plutôt que lue dynamiquement. Pas commencé, pas scopé plus finement que ça — à reprendre seulement si la mesure post-implémentation (une fois construite) montre un vrai gain sur un kernel qui a réellement des structs tensor-porteurs dans son chemin chaud.

## 14. Le vrai modèle sous-jacent — classes d'équivalence de valeur, par champ, pas quatre mécanismes séparés

Retour en arrière discuté en revue, après §11/§13 : `AliasSummary`, `IdentitySummary`, la propagation par résomption d'appel (§11.3), la propagation par paramètre porté (§11.4), et le dédoublonnage `is_loop`/`is_entry_arg` (§2.3) répondent tous, sous des formes différentes, à la **même** question. Ce paragraphe pose le modèle qui les unifierait, et pourquoi il est réellement fermé (décidable, pas juste "probablement correct") dans ce compilateur d'une façon qui ne le serait pas en C — avant de committer quoi que ce soit dessus.

### 14.1 L'insight

Une **memory location** (une allocation de struct/array, `PrimOp::Struct`/`PrimOp::Array`) est une entité stable, créée une seule fois. Un `CVar` n'est jamais qu'un **nom** pour elle à un point du programme — passer un paramètre, retourner une valeur, porter par une boucle sont tous la même opération : *renommer* la même location, jamais en créer une nouvelle. L'aliasing véritable n'est pas « cette valeur traverse-t-elle une frontière de fonction » (le critère `known_functions`/`is_loop` qu'on vient de retirer en §11.4) — c'est : **deux noms différents de la même location ont-ils des live ranges qui se chevauchent**, exactement le critère d'interférence qu'un allocateur de registres calcule pour décider si deux valeurs peuvent partager un slot.

### 14.2 Pourquoi c'est fermé ici et ne le serait pas en C

En C, l'aliasing est indécidable en général — arithmétique de pointeur, cast, union, `&` sur n'importe quel objet : une analyse doit rester conservative partout, elle ne peut jamais fermer le monde. Ici :
- **Aucune façon de fabriquer une référence hors des constructions du langage.** Pas d'adresse-de, pas d'arithmétique de pointeur, pas de réinterprétation de type. Le seul moyen de faire exister un alias est un ensemble **fini et énumérable** de formes CPS : passage de paramètre/argument à un saut (`App`), retour (`App{Var(k_ret), [v]}`), portage de boucle, projection de champ (`Field`), indexation de tableau à taille statique (`Load`). Chacune est visible et nommée dans l'IR.
- **Le stockage n'est jamais un choix de l'utilisateur.** Un struct « lourd » est toujours une référence canonique unique vers un bloc géré par le compilateur — jamais un layout que le programmeur réinterprète. Pas de « même objet vu à travers deux types différents » à envisager.
- **On ne s'intéresse qu'aux valeurs qui ont déjà un slot** (`is_rc(ty)`/`needs_seed` filtrent déjà exactement ça avant d'entrer dans cette machinerie) — le scalaire et le temporaire d'arène (`escape.rs`) sont hors sujet par construction, pas juste par convention.

Cette fermeture est ce qui rend un vrai union-find tractable et *complet* (pas juste "conservative et espérons que ça suffise") — un luxe que l'analyse C-like équivalente n'a jamais.

### 14.3 Ce qui serait un plain union-find — le renommage pur, *sauf une frontière*

**Corrigé après un vrai essai raté cette session — voir §14.8.** Une union directe entre deux `CVar` fait *que* renommer, sans jamais créer de branchement, seulement quand **aucune des deux extrémités n'est le paramètre ou le retour partagé d'une fonction top-level** — ces deux-là sont des noms *symboliques*, réutilisés tels quels par tous les appels à cette fonction, jamais une identité physique unique. Unir directement l'argument d'un appelant avec eux fusionnerait transitivement les arguments de deux appels sans rapport dès que la fonction a plus d'un site d'appel — un vrai bug de précision, pas une approximation acceptable (§14.8 en donne la preuve par un test qui casse).

**Ce qui reste un union-find direct, sans risque** — jamais partagé au-delà d'une seule fonction top-level englobante :
- **Argument d'entrée/dos-de-boucle → paramètre porté d'un `Fix` local** (boucle, jonction `if`) — toutes ses sources vivent, par construction, dans la *même* fonction englobante. Un paramètre de boucle porté finit dans la même classe que son argument d'entrée ET chacun de ses arguments de dos-de-boucle — l'AND-join que `collect_affine_carried_params` (§11.4) calcule aujourd'hui à la main, gratuit ici.

**Ce qui reste port + `propagate()`, comme aujourd'hui** — toute frontière de fonction top-level (paramètre *ou* retour, un vrai appel *ou* une résomption qui en reçoit le résultat) : un fait booléen par port (« ce paramètre/retour est-il jamais aliasé »), une arête à sens unique par appel (« si le port du callee est aliasé, le mien l'est aussi »), résolu par le même `propagate()` que `analyze()`/`analyze_identity()` utilisent déjà. C'est déjà exactement ce que §11.3/§11.4 ont construit et validé cette session — pas à réinventer, à **partager** entre `AliasSummary` et `IdentitySummary` plutôt que dupliqué dans deux fonctions séparées (`analyze`/`analyze_identity`), ce qui reste une vraie simplification même sans toucher à la frontière inter-fonctions.

### 14.4 Ce qui est plus dur — les champs sont mutables, pas du SSA

Là où ça dépasse un simple union-find : un champ de struct n'est pas assigné une fois. `PrimOp::FieldStore { struct_ty, field }`, `args = [base, value]` — le même `base` (le même nom, la même location) voit le contenu de son champ changer avec le temps. La classe d'équivalence « qu'est-ce qui occupe le champ `field` de cette location, ici, à ce point du programme » n'est donc pas une constante — c'est une analyse *sensible au flot*, une forme de points-to must-alias, pas juste une fermeture transitive statique une fois pour toutes.

**C'est exactement ce point qui débloquerait §13** (cascade sans header pour un struct porteur de tensor — `Network`/`Dense`/`NetworkState`) et l'ambition « granularité champ » du plan Tier-1 d'avant cette session (`C:\Users\chris\.claude\plans\ok-je-me-demande-swift-bunny.md`) — les deux sont la même extension, pas deux travaux distincts : une fois qu'on sait, pour chaque champ, quelle classe l'occupe et si elle est sûre, la cascade à la libération devient une simple récursion sur cette information déjà calculée, résolue à la compilation (`struct_field_types` connaît déjà la disposition), jamais lue dynamiquement.

Mais ce reste **fermé** de la même façon que §14.2 : le nombre de sites `FieldStore` pour un struct donné est fini et connu statiquement (déclaré par l'utilisateur), contrairement à C où une mutation à travers un pointeur arbitraire rend ce genre d'analyse généralement intraitable.

### 14.5 Le point d'évasion — déjà là, rien à construire

`DynArray<T>` a besoin d'un vrai monde ouvert (`push` dans une boucle à borne runtime — le nombre de références vivantes n'est pas énumérable statiquement, la frontière que le plan Tier-1 avait déjà posée). Vérifié directement : son buffer brut (`stdlib/dynarray/dynarray.cleave`, `struct RawBuf {}`, zéro champ) n'est **jamais** touché autrement que via des `extern fn` (`dynarray_alloc_ptr`/`get`/`set`/`grow`) — donc la règle 4 déjà existante (« passé à un extern, aucune visibilité, aliasé sans condition ») l'exclut *automatiquement* du monde fermé, sans code spécial. Le point d'évasion que ce modèle a besoin d'avoir existe déjà, nommé nulle part comme tel jusqu'à cette conversation, mais bien réel : **tout type dont chaque interaction passe par `extern` sort du modèle fermé pour de bon**, par la règle la plus ancienne de ce fichier (§3, règle 4), pas par une addition.

### 14.8 Le premier essai de Phase A, raté, et pourquoi — trouvé par le test, pas anticipé

Une première implémentation complète a été écrite : un seul union-find global sur tous les `CVar`, unissant *sans distinction* argument↔paramètre et retour↔résomption, à travers *toute* frontière (boucle locale ET fonction top-level). Elle a compilé, et 14 tests sur 16 sont passés du premier coup.

**Les 2 échecs ont révélé le vrai défaut, pas un détail à corriger** : `bump(a: Boxed) -> Boxed { Boxed(v: a.v+1, tag:[a.tag[0]]) }`, appelée depuis `main` avec `opaque_sink(bump(a)) + opaque_sink2(a)` — `a` est passé à `bump` ET, séparément, à `opaque_sink2` (un extern). Le hasard extern sur `a` (règle 4) marque sa classe aliasée — et comme `bump(a)` avait uni `a` avec le paramètre *propre* de `bump`, ce hasard remonte directement contaminer « `bump` est-elle sûre », un fait qui doit être **intrinsèque au corps de `bump`**, jamais dépendant de ce qu'un appelant particulier fait ailleurs avec sa propre copie. Le test attendait `!is_aliased("bump", 0)` — l'union-find global répondait `true`.

**Pourquoi ce n'est pas juste "un peu moins précis"** : un paramètre (ou un retour) de fonction top-level est un nom *symbolique*, partagé tel quel par tous les appels à cette fonction — pas une identité physique. Unir directement l'argument d'un appelant avec lui fusionne transitivement, dès que la fonction a plus d'un site d'appel, les arguments de deux appels sans rapport entre eux. Une fonction aussi ordinaire que `bump`, appelée plusieurs fois avec des arguments aux usages différents, verrait son propre verdict d'aliasing dépendre du comportement du pire de ses appelants — dans un vrai programme, ça revient à rendre Stage 2 quasiment inutile (déjà seulement 9 sites éligibles sur le kernel réel avec le mécanisme précis actuel, §13 — un mécanisme moins précis en trouverait probablement zéro).

**La correction, §14.3 mise à jour** : le union-find direct ne s'applique qu'*à l'intérieur* d'une seule fonction top-level englobante (paramètres/retours de `Fix` locaux — boucles, jonctions `if` — jamais partagés au-delà). Toute frontière de fonction top-level reste port + `propagate()`, exactement le mécanisme que §11.3/§11.4 avaient déjà construit et validé plus tôt cette session, avant cet essai. Revert fait (`git checkout` sur les deux fichiers touchés, retour au commit de checkpoint) — rien de cassé n'est resté dans l'arbre.

**Ce qui reste vrai malgré l'échec** : l'insight (mémoire = classe d'équivalence, aliasing = interférence de live ranges) tient — il s'applique juste à une portée plus fine (une fonction à la fois pour le renommage direct) qu'espéré, avec la frontière inter-fonctions gérée par le mécanisme déjà existant plutôt que remplacée.

### 14.6 Phasage révisé

1. **Phase A, restreinte (§14.3 corrigée)** : partager `seed`/`edges`/`propagate()` entre `AliasSummary` et `IdentitySummary` (aujourd'hui deux implémentations séparées mais isomorphes) et remplacer `is_loop`/`is_entry_arg` par le union-find *local* (boucle/jonction, jamais à travers une frontière de fonction top-level). Une vraie simplification, mais plus modeste que l'ambition initiale — la frontière inter-fonctions garde son mécanisme actuel, déjà correct. Pas commencé.
2. **Phase B — les champs mutables (§14.4)** : inchangée, c'est *cette* phase qui débloquerait un gain mesurable sur `mnist-interop` (§13) — Phase A seule ne change rien à la mesure Stage 3.

### 14.7 Discipline, reprise de §7

Même règle que tout ce fichier : une seule chose par commit, jamais `CLEAVE_AFFINE_STRUCTS` par défaut avant mesure complète, jamais de reproducteur non borné laissé sans supervision. Phase A avant Phase B, strictement — Phase A doit être acceptée seule (suite verte, verdicts identiques à l'ancien mécanisme) avant que Phase B ne touche au moindre fichier.
