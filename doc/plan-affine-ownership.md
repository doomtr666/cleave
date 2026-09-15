# Plan — "piste 1" : n'allouer un header refcompté que si la valeur est réellement aliasée

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
