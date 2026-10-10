# Literature

Full text and notes live in paperpipe; use the `papi` name to look them up.

## Current focus

The papers agents should work from until the user changes or requests a change to this section. Read these (via `papi`) before proposing designs for the listed cards.

- **Card / rung:** [card 094](experiments/094-object-files-and-layout/card.md),
  belief as object files and a layout. Rung 1 (P2, P17). Set 2026-10-09
  with the user, after framing the architecture as four memories
  (ARCHITECTURE.md, "Memory systems").
- **Why these:** how working memory holds a few objects with their
  places, how the brain codes where (path integration, places, vectors
  to objects and walls), and agents that keep a set of object vectors or
  a spatial map as memory.
- **Until:** card 094.2 has a decision (094: revise; 094.1: stop, 2026-10-09).

| Paper (papi name) | What to take from it |
| ----------------- | -------------------- |
| Kahneman, Treisman and Gibbs 1992, object files (not in papi) | An object file binds an object's features to a place-and-time index and persists as the object moves; a newly seen object is matched to a file by where it is expected, not only by how it looks. Card 094: a file is indexed by its token id (its where at first sight), so two doors of one colour are two files |
| Pylyshyn and Storm 1988, tracking (not in papi); Cowan 2001, capacity (not in papi) | People track about 4–5 objects at once; working memory's focus holds about 4 chunks. Card 094 sets no limit yet and says so; a limit needs the layout or long-term memory to hold the rest |
| Whittington et al. 2020, the Tolman–Eichenbaum machine (not in papi); `2112_04035` (Whittington, Warren and Behrens 2022) | Where is a code updated by an action-specific transformation (path integration; grid-cell-like), kept apart from what is sensed; memories bind what to where and are retrieved by attention on position (place-cell-like). Card 094's placement (one learned transformation per move) is this split; object files and the layout are memories indexed by it |
| Høydal et al. 2019, object-vector cells; Lever et al. 2009, boundary-vector cells; O'Keefe and Dostrovsky 1971; Hafting et al. 2005 (none in papi) | The entorhinal cortex codes the vector from the animal to objects, whatever the object, and to boundaries. Card 094: an object file's where is its vector from the agent; the layout's blocked places are boundaries |
| `1911_07141` (Loynd et al. 2020, working memory graphs) | An agent attending over a dynamic set of factored vectors (one "memo" replaced per step) learns faster on BabyAI levels and Sokoban than recurrent baselines: attention over a small set of object vectors works as working memory in worlds like ours. There attention is learned end to end by reward; card 094's is a rule over learned predictions |
| `neural-map` (Parisotto and Salakhutdinov 2017) | A 2D memory written at the agent's place and read by attention and local convolution beats recurrent memories in mazes. The agent's position is given there; card 094's layout is indexed by the learned placement |
| `1803_02155` (Shaw et al. 2018, relative position representations); `1706_01427` (relation networks) | Attention that reads each element's offset from another, learned per offset; objects given as features with their coordinates. Card 094.1: the router reads each object file's offset from the agent |
| `object-centric-learning-with-slot-attention` (Locatello et al. 2020) | A fixed number of slots compete for the parts of a scene. Not needed while tokens are given per tile; for Crafter and beyond, where units must be found (P7) |

Previous focus (card 093, set 2026-10-09; set aside by the user):
working memory as an intention held between steps (Fikes, Hart and
Nilsson 1972; Bratman 1987; `rao-georgeff-bdi`; `kinny-georgeff-commitment`;
complementary learning systems). Taken up as card 097.2.

Previous focus (cards 091–092, set 2026-10-08): recall's prior through a
learned router and recall queried by the effect a goal needs (Pritzel et
al. 2017, neural episodic control; Wu et al. 2022, memorizing
transformers; STRIPS; hindsight experience replay; Hammond 1989), until
card 092's decision (revise, 2026-10-09).

Previous focus (cards 087–089, set 2026-10-08): properties as learned
action effects (Gibson; Montesano et al.; deep ensembles; Achille and
Soatto; Fisher's discriminant; invariant causal prediction; Webb et
al.; Gatys et al.; Tenenbaum and Freeman), until card 089's decision.

Previous focus (card 082, set 2026-10-06):

- **Card / rung:** [card 082](experiments/082-reasoning-over-recalled-tries/card.md),
  a network that reasons over recalled tries. Rung 6 (P3), with P10 and
  P21. Set with the user on 2026-10-06, after cards 076–081: recall
  compares the query with each stored try separately and votes, so it
  cannot induce "in every opening the door and key were alike in
  colour" from its memory; hand-built inputs (roles, "on the way", card
  070's projection) each passed one test only. CHARTER now judges recall
  and networks by held-out prediction of the agent's own experience.
- **Why these:** networks that read a set of stored examples and predict
  a new one (in-context), relations as comparisons that keep identity
  out, retrieval trained with the reasoning stage, and what training
  data makes a network reason over its context rather than memorise.
- **Until:** card 082 has a decision.

| Paper (papi name) | What to take from it |
| ----------------- | -------------------- |
| `2106_02584` (Kossen et al. 2021, Non-Parametric Transformers) | The whole dataset is the input; attention alternates between rows (datapoints) and within a row (attributes). Trained on one dataset by masking targets and predicting them from other rows: our setting. Lookup task: correct on 99.9% where k-NN and deep kernels are at chance (RMSE 5.2–6.4 against NPT's 0.24–0.75); predictions follow labels changed at test time. Shuffling the other rows at test time shows whether a dataset's predictions use them. Never tested on a held-out attribute value. O(n²) in rows |
| `1901_05761` (Kim et al. 2019, Attentive Neural Processes); `1807_01622` (Garnelo et al. 2018, Neural Processes) | Reading a context set of (input, outcome) pairs: averaging the set into one vector underfits; self-attention among context pairs and cross-attention from the query fix it. Dot-product attention can collapse to nearest neighbour. Both meta-train on many separate functions, which we do not have |
| `2112_10510` (Müller et al. 2021, Prior-Data Fitted Networks) | A transformer trained on millions of synthetic datasets from a hand-written prior does Bayesian inference in context (matches the exact GP posterior; transfers to 20 real tabular sets). Its quality is the prior's; a hand-written prior is the engineering CHARTER rules out |
| `2304_00195` (Altabaa et al. 2023, Abstractor) | Relational cross-attention: attention scores from inner products of the objects' projections, values from symbols (positions or roles), so object features never reach the output. Several heads, separate query and key projections for asymmetric relations; all pairs. The diagonal leaks single-object information |
| `2012_14601` (Webb et al. 2021, ESBN) | A memory that passes only similarities and a confidence to its controller, never the vectors: 95–100% on four relational tasks with 95 of 100 images withheld (Transformer 67–80%). Needs normalisation over the context (50% without it). Tested on sameness of whole images only |
| `2206_05056` (Kerg et al. 2022, CoRelNet); `1706_01427` (Santoro et al. 2017, Relation Networks) | All-pairs similarity matrices as the only input carry to unseen relations (83–100%), but fail (≈58%) when training lacks the base patterns, drop to ≈55% when raw features are added beside the relations, and are fooled by a spurious shared feature without regularisation. A relation network on raw object pairs learns shortcuts and does not carry (26.5% on held-out images) |
| `2202_08417` (Goyal et al. 2022, retrieval-augmented RL) | A learned retrieval (keys and queries from the agent's own encoder, top-k) followed by attention over the retrieved items, trained end to end with the agent; retrieved items re-encoded each update, so no stale keys. Querying with the plain state was no better than no retrieval; a learned query was (+11% Atari; BabyAI 45% to 74%). The router-then-reasoner design |
| `2203_08913` (Wu et al. 2022, Memorizing Transformers) | k-nearest-neighbour retrieval from a large memory into one attention layer, top 32; approximate retrieval is tolerated, normalised keys handle staleness; gains come from rare items |
| `2205_05055` (Chan et al. 2022) | When a network reasons in context and when it memorises in its weights: few classes, no variation within a class and fixed meanings give memorisation (100 classes: in-context at chance); many rare classes, variation and meanings that change between contexts give in-context learning. One world with one fixed relation is the memorising regime unless the outcome cannot be read from the query alone |

Previous focus (cards 049–050, set 2026-10-03):

- **Card / rung:** [card 049](experiments/049-recall-through-conditions/card.md),
  recall through the conditions that matter, and
  [card 050](experiments/050-own-tries-first/card.md), own tries first
  (kept as version 9, 2026-10-04). Before rung 1. Set with the user on
  2026-10-03, after tracing card 048's failures. In both, a token in view
  that does not matter to the action (a second door; a key missing from
  the floor) dropped every matching stored try's weight to zero. Recall
  compares the whole view, and its similar-thing level is off (β ≈ 0 in
  all five seeds). The aim is memory queried through the conditions that
  matter for each action, generative and preventive, so that irrelevant
  tokens cannot veto a memory and new but similar situations can query
  old ones.
- **Why these:** which cues enter a query (contrast, default exclusion,
  cue competition); conditions matched one way, from the stored try to
  the present; references relative to the target and the hand; relations
  as distances within one part; conditions from few contrasting tries.
- **Until:** the user sets the focus for card 051 (the planner).

| Paper (papi name) | What to take from it |
| ----------------- | -------------------- |
| Cheng 1997, causal power (not in papi); Cheng and Novick 1991 (not in papi) | A cue's generative and preventive power come from the contrast in outcomes with and without it. A cue that never varied has no measurable power: it is background (an enabling condition), not a mismatch |
| Griffiths and Tenenbaum 2005, causal support (not in papi) | Whether a cue matters at all is a model comparison, "matters" against "does not", decided from few tries. Exclusion is the default |
| `1110_2211` (Pasula, Zettlemoyer and Kaelbling 2007) | Rules name only the objects they need, by their relation to the action's target, with a penalty on rule size and a noise outcome for rare exceptions |
| `2406_03234` (Hwang et al. 2024) | Which inputs matter changes with the situation: a learned codebook over (state, action), each code with its own sparse dependencies |
| `2007_02863` (Pitis et al. 2020, CoDA) | Interactions are local, so tokens outside an action's local dependencies should not affect a match. Its dependencies come from query–key attention, which LESSONS warns against |
| `2106_03443` (Seitzer et al. 2021); `elden` (Hu et al. 2023) | Whether a token mattered in one try, scored by how much the model's prediction changes with it |
| `2103_00589` (Silver et al. 2021); `2105_14074` (NSRTs, Chitnis et al. 2022); `schema-networks-zero-shot-transfer-with-a-generative-causal` (Schema Networks); DOORMAX (Diuk et al. 2008) | Group tries by effect, keep what every success shares, add a condition only when a failure forces it; stored conditions are matched one way, so extra objects cost nothing. Their attributes are given, ours are learned |
| Rescorla and Wagner 1972; Kamin 1969; Kruschke 2001, EXIT (none in papi) | Cue competition: a cue earns weight only by predicting what the others leave unexplained (blocking). Attention fast and specific to the situation |
| Kruschke 1992, ALCOVE; Nosofsky 1986; Medin and Schaffer 1978 (none in papi) | Recall's present base. Similarity multiplied over dimensions lets one mismatch veto a memory, unless attention on irrelevant dimensions is near zero |
| `the-relational-bottleneck-as-an-inductive-bias-for-efficient` (Webb et al.); Gentner et al. 1993; Hummel and Holyoak 1997, LISA | Relations as similarities between two objects in one part, never their attributes: what carries "same colour" to a new colour |
| Gopnik and Sobel 2000; Mitchell, Keller and Kedar-Cabelli 1986; Love et al. 2004, SUSTAIN (none in papi) | One success and one failure differing in one token can settle a condition; a surprising outcome stores a new case or re-indexes old ones |
| `1501_01332` (Peters, Bühlmann and Meinshausen 2016, invariant causal prediction) | A true condition predicts the outcome alike in every room or layout; a distractor does not |
| `neural-production-systems` | A rule binds one primary slot and one context slot. Its selection is query–key attention, the failure in LESSONS |

Previous focus (cards 036–037, set 2026-09-29):

- **Card / rung:** [card 036](experiments/036-fresh-codes/card.md), fresh
  codes for new things, then [card 037](experiments/037-recall-in-the-planner/card.md),
  recall in the planner. Before rung 1. Set with the user on 2026-09-29,
  after card 035 stopped. Recall on the encoder's vectors replaces the
  counted lookup behind the planner's entries, and novelty is judged per
  action by the vote's weight. Card 036 first gives each new tile a name
  of its own.
- **Why these:** exemplar recall as the source of predictions, with a
  learned metric per action and its weight as doubt.
- **Until:** card 037 has a decision.

| Paper (papi name) | What to take from it |
| ----------------- | -------------------- |
| `1703_01988` (Pritzel et al. 2017, Neural Episodic Control) | Predictions from a kernel-weighted sum over stored experiences in a slowly changing embedding; it latches onto a success as soon as it is seen. Card 037's entries are this over the planner's stored tries |
| Nosofsky 1986, the generalized context model (not in papi) | A new item is judged by a similarity-weighted vote of stored items, with learned attention per dimension: recall's k and λ |
| Kruschke 1992, ALCOVE (not in papi) | Attention per dimension learned from error; it shifts to the dimensions that tell outcomes apart. Card 037 learns λ per world and action |
| `1604_02354` (Bayesian NCA; restates Goldberger et al. 2004) | Leave-one-out prediction as the objective for the metric: card 037's λ fit |
| `1703_05175` (Snell et al. 2017, prototypical networks) | Distance to a prototype as doubt; card 035's radius and card 036's fresh codes |
| MacKay and Peto 1995, "A hierarchical Dirichlet language model" (not in papi) | A context's own counts are smoothed toward a shared back-off distribution through a Dirichlet prior whose strength α is fitted by leave-one-out likelihood. Card 038's recall (revision agreed with the user on 2026-09-30): a key's own tries with the neighbours' vote as the prior, α fitted per world and action |

Previous focus (card 035, stopped 2026-09-29):

- **Card / rung:** [card 035](experiments/035-novelty-from-own-tiles/card.md),
  novelty from a code's own tiles: card 034's recall-trained codes, read
  with a new rule. A tile is "new" in a codebook only when it lies
  farther from its nearest code than α times that code's own spread, with
  α set by leave-one-out on familiar tiles. Before rung 1. Approved by the
  user on 2026-09-29 ("get the novelty signal working"), under CHARTER.md's
  new point "One latent space". It was first drafted as a sparsity
  penalty on recall's weights; that probe placed the failure in the
  novelty rule.
- **Why these:** a code as a prototype with a radius, and choosing that
  radius without the test; the sparsity papers were read for the first
  draft.
- **Until:** card 035 has a decision (stopped; see its section 8).

| Paper (papi name) | What to take from it |
| ----------------- | -------------------- |
| `2107_10098` (Lachapelle et al. 2022, mechanism sparsity) | Latent factors are recovered, up to permutation, when each mechanism depends on few of them, the learned dependency graph is penalised for its edges, and the data vary enough (a graph connectivity condition). Binary masks with a penalty on their expected count; synthetic data only. Card 035's first draft used the continuous analogue on recall's weights, one group per part |
| Yuan and Lin 2006, "Model selection and estimation in regression with grouped variables" (not in papi) | The group lasso: a Euclidean norm per group of weights, summed, sets whole groups to zero. Card 035's first draft grouped recall's weights by the part a codebook quantises |
| `1811_12359` (Locatello et al. 2019) | Without declared biases nothing identifies factors, and seeds matter as much as methods: fixed settings, many seeds, no selection by the test |
| `1703_05175` (Snell et al. 2017, prototypical networks) | A class is a prototype, and a query's distance to it is the doubt. Card 035's code is a prototype whose radius comes from its own training tiles |
| `1604_02354` (Bayesian NCA; restates Goldberger et al. 2004) | Leave-one-out prediction on the training set as the selection signal: card 035 chooses α this way, on familiar tiles only |
| Kruschke 1992, ALCOVE (not in papi) | An exemplar model that learns attention per dimension by error-driven learning; attention shifts to the dimensions that tell the categories apart. Recall's per-action weights are this attention |

Previous focus (card 034, stopped 2026-09-29):

- **Card / rung:** [card 034](experiments/034-recall-teaches-the-encoder/card.md),
  recall teaches the encoder: card 033's recall term trains card 031's
  encoder, with room in the codebooks, so that things that act alike share
  codes and a new appearance falls in with them. Before
  rung 1. Set with the user on 2026-09-29, after card 033 stopped (its
  section 8, and CHARTER.md's "Current direction"). Kinds are distributed
  across the codebooks, never merged. The user's fallback is to build
  outline and appearance as separate parts of the encoder.
- **Why these:** the direction's second layer needs encoder features in
  which a kind's members agree. Card 031's encoder has none (card 033,
  section 8).
- **Until:** card 034 has a decision (stopped at its gate; see its section 8).

| Paper (papi name) | What to take from it |
| ----------------- | -------------------- |
| `1802_02745` (Feinman & Lake 2018) | A small convolutional network trained to name objects by shape, with colour and texture random, learns to generalise by shape. Doing so for a new colour of a trained category (first order) takes less data than for a new category (second order). Second order reaches 0.7 at 6 examples of 8 categories, 3 of 32, or 12 of 4. We have 4 kinds with several members, 3 members each: the edge of that range. L2 regularisation and small random shifts mattered |
| `1706_08606` (Ritter et al. 2017) | ImageNet-trained one-shot models prefer shape to colour, but the strength varies greatly across seeds of equal accuracy: report many seeds |
| `1811_12231` (Geirhos et al. 2019) | ImageNet networks judge by texture, not shape. Training on style-transferred images makes them judge by shape, so the bias comes from the data |
| `2004_11362` (Khosla et al. 2020, supervised contrastive) | The kind term of the drafting probes: each item is pulled toward the others with its label and away from the rest (temperature 0.1). Hard labels; card 034 uses recall's graded, per-action similarity instead |
| `1703_05175` (Snell et al. 2017, prototypical networks) | A class is the mean of its members; a new item is classed by its distance to the means. This is the second layer's "which kind is this new appearance", with distance as doubt |
| Smith et al. 2002, "Object name learning provides on-the-job training for attention" (not in papi) | Toddlers taught four categories organised by shape start to extend new names by shape: the shape bias is learned |

Previous focus (card 033, stopped 2026-09-29):

- **Card / rung:** [card 033](experiments/033-relation-codes/card.md),
  relation codes: recall of door tries, compared in card 031's encoder,
  predicts a door of a never-seen colour; "fits" becomes a code when it
  explains the tries more cheaply than one rule per colour. Before rung 1.
  Drafted with the user, 2026-09-29, after card 032 was stopped.
- **Why these:** the user's direction, counting → recall → weights. Recall
  is fast and nonparametric; replay trains the encoder slowly. The
  comparison must be computed, never looked up, to reach a new colour.
  Card 030's reading on relations, Plotkin and description length (below)
  still applies.
- **Until:** card 033 has a decision (stopped; see its section 8).

| Paper (papi name) | What to take from it |
| ----------------- | -------------------- |
| `1703_01988` (Pritzel et al. 2017, Neural Episodic Control) | A slowly changing encoder writes every experience to a memory; a lookup is a kernel-weighted sum over the p nearest keys (p = 50). It learns significantly faster than DQN and A3C, and latches onto a success as soon as it is experienced. Card 033's recall is this with one neighbour and our comparison as the key |
| `1606_04460` (Blundell et al. 2016, Model-Free Episodic Control) | Nearest-neighbour memory over a VAE's 32 numbers learns one-shot and faster than parametric learners early on, and may be overtaken later: the case for recall first, weights later |
| Shepard 1987, "Toward a universal law of generalization" (not in papi) | Generalisation falls off exponentially with distance in a psychological space: card 033's vote weight exp(−distance) |
| Nosofsky 1986, the generalized context model (not in papi) | Exemplar model: a new item is classed by a similarity-weighted vote of stored items, with learned attention weights per dimension; card 033's λ |
| `1604_02354` (Bayesian NCA; restates Goldberger et al. 2004) | Neighbourhood components analysis learns a metric by leave-one-out nearest-neighbour prediction: each point is predicted from the others. Card 033's objective, with unnormalised weights and a prior so that distance shows as doubt |
| `2012_14601` (Webb et al. 2021, ESBN) | A memory binds entity embeddings to abstract variables, retrieved by comparison; rules carry to withheld entities (at least 95% with 95 of 100 images withheld). Labelled tasks |
| `the-relational-bottleneck-as-an-inductive-bias-for-efficient`, `2304_00195` | Processing that sees only comparisons between objects transfers to new objects |

Previous focus (card 032, stopped 2026-09-29):

- **Card / rung:** [card 032](experiments/032-fewest-codes/card.md),
  fewest codes: card 031's encoder must describe the tiles with as few
  codes as it can. [Card 031](experiments/031-codes-from-pixels/card.md)
  (codes from pixels; revise) found that learned codes lose nothing but
  name tiles arbitrarily. Before rung 1. Changed with the user,
  2026-09-29, when card 030 was stopped.
- **Why these:** the user's direction (card 031, appendix B): tile, later
  a segment → vector → several codebooks → rules and goals over codes. The
  codebooks must not repeat each other, and what each code means must not
  be engineered; principled mathematics should make them useful. The
  papers cover how discrete codes are learned, and what does and does not
  identify separate factors without labels.
- **Until:** card 032 has a decision.

**What the reading found** (2026-09-29, read in full; papi's automatic
summaries failed for lack of a Gemini key):

1. **Nothing identifies factors from single images** (`1811_12359`).
   Infinitely many entangled codes fit the data equally well. Across
   more than 12,000 models, the seed and the regularisation strength
   mattered more than the method, and choosing a model by a label-based
   score is supervised selection. So: declare the biases, fix the
   settings in advance, report many seeds, and never tune on the test.
2. **Changes identify factors**, when each change touches few of them and
   which ones changes varies:
   - Pairs sharing all but a few factors (`2002_02886`): median DCI
     score (a disentanglement measure) on Shapes3D 94.6% against 70.9%
     for the best unsupervised method. A partial failure on SmallNORB.
   - A Laplace prior on changes between frames (`2007_10930`): dSprites
     MCC 58.8 against 46.0 and 41.6 for two earlier methods (Ada-GVAE,
     PCL). MCC is the mean correlation between matched true and learned
     factors. No gain on Natural Sprites, where shape never changes
     within a pair.
   - Sparse masks on what each latent and action depends on
     (`2107_10098`): MCC about 0.97 against about 0.58 unregularised, on
     synthetic 20-number data only. No gain with linear transitions.
3. **In our world these separate only what changes.**
   - Toggling changes only a door's state; stepping into a doorway
     changes only the agent drawn there.
   - Colour never changes apart from the kind of thing (pick up and drop
     change both), so no paper's condition holds for colour.
   - An attribute that never changes is separated only by capacity
     (codebooks too small to name every tile) and by the encoder's
     structure.
   - No guarantee covers discrete codes, or inputs outside the training
     data.
4. **Several codebooks** (`1803_03382`): cutting the encoder's output into
   slices, each quantised against its own codebook, avoids index collapse
   (only a few codes in use). Two slices were best for translation. The
   paper says nothing about factors.
5. **Pitfalls.**
   - Pairs in which the thing leaves its place (a key picked up leaves
     floor) share nothing, and pair methods then force false sharing.
   - A codebook large enough to name every tile leaves the others free to
     go unused.
   - Counting dependencies alone can favour one merged kind × colour
     code.

| Paper (papi name) | What to take from it |
| ----------------- | -------------------- |
| `1811_12359` (Locatello et al. 2019) | No factors without declared biases; seeds matter more than methods; fixed settings, many seeds, no selection by the test. Names interaction, grouping and time as sources of bias |
| `2002_02886` (Locatello et al. 2020) | Pairs that share all but k factors identify them (continuous, invertible, varied sharing). Ada-GVAE estimates which factors changed with a threshold; paired reconstruction is a label-free selection score. Our door toggles and doorways fit (k = 1); pick up and the vase breaking share nothing |
| `2007_10930` (Klindt et al. 2021, SlowVAE) | An absolute-value (Laplace) prior on changes identifies factors up to permutation; no help for factors that never change within a pair |
| `2107_10098` (Lachapelle et al. 2022) | Sparse masks on what each latent and action depends on. In counts: each entry reads as few codebooks as possible. Synthetic vectors only |
| `1803_03382` (Kaiser et al. 2018) | Sliced vector quantisation: several codebooks for one vector, avoiding index collapse |
| `1711_00937` (VQ-VAE) | Nearest-code quantisation, straight-through gradients, codebook and commitment terms |
| `1705_00154` (Asai and Fukunaga 2018, LatPlan) | Discrete latent propositions learned from pixels, so that a classical planner can test them exactly. Card 042: codes give recall the identity a planner needs; a weighting of continuous vectors could not |
| `1802_04942` (Chen et al. 2018, β-TCVAE) | The total correlation (the sum of each latent's entropy minus their joint entropy) is the term that measures latents sharing information; penalising it favours factorial codes. Card 032: with tiles kept distinct the joint entropy is fixed, so penalising the sum of codebook entropies penalises it exactly (Barlow's minimum entropy codes, 1989) |
| `1802_05983` (Kim and Mnih 2018, FactorVAE) | The same penalty estimated with a discriminator; better trade-off of rebuilding against disentanglement than β-VAE. We compute it directly: codes are discrete and there are 20 tiles |

Previous focus (card 030, stopped 2026-09-29; still the reading for the
card on rules with a shared variable):

- **Card / rung:** the transfer card, to be drafted from
  [card 029's appendix B](experiments/029-subgoals-from-the-model/card.md):
  attributes found as recurring differences, rules over relations, a
  withheld colour as the test. Before rung 1. Changed with the user,
  2026-09-28.
- **Why these:** the user's framing. An attribute is any respect in which
  things can be alike or differ, found by general pattern matching, and
  shape is as much an attribute as colour. The evidence should choose
  which attribute a rule uses (the goal "get a key" is about shape; the
  door's condition is a colour match), and compression should favour
  rules that apply to new objects. The papers cover four things:
  - what an attribute is, and how to test one;
  - finding attributes and relations by comparison;
  - rules over relations, and compression;
  - how far to generalise, and how to test it.
- **Until:** the transfer card has a decision.

**What the review found** (21 papers added to papi on 2026-09-28 and read
in full; papi's automatic summaries failed for lack of a Gemini key):

1. **No paper finds the attributes themselves from raw pixels without
   labels.** DORA and SME are given features and dimensions. ARC solvers,
   PrAE, the Abstractor and the neuro-symbolic concept learner are given
   attribute slots, values or words. Methods that learn everything from
   data collapse on values held out: PGM's held-out shape–colour split
   fell to 12.5%, which is chance. The discovery step is ours to design.
2. **They agree on the rest:**
   - **An attribute is a transformation that recurs.** It commutes with
     the others (Higgins: factors are subgroups of the world's
     symmetries), and it can be checked against the model (Ravindran: a
     symmetry is a relabelling that leaves the dynamics unchanged;
     nothing has to perform it). Swapping red and blue across keys and
     doors together leaves every learned rule unchanged. Swapping only
     the keys breaks the door rule, so the model forces colour to line
     up across shapes.
   - **Once an attribute exists, "same along it" is a generic relation**
     (DORA 2022: same, more or less on any dimension). A match needs a
     shared variable (Plotkin: the same pair of differing terms becomes
     the same variable); otherwise the result is "any key opens any
     door".
   - **Compression favours the relational rule only when it pays.** In
     DreamCoder's terms, one colour never pays and three can. So score
     the attributes and the rules together.
   - **The agent's own play cannot tell "same colour, for all colours"
     from "one rule per colour seen".** Both fit, and only the prior
     (description length) separates them (Tenenbaum & Griffiths, weak
     sampling). The withheld-colour test measures that prior.
   - **Which attribute a rule uses is identified only by contrasts**, for
     example a wrong-colour key held at a door (Hill et al.; DORA).
3. **Pitfalls.**
   - A relational rule carries to an unseen colour only if attribute
     values are computed from pixels, not looked up. Relational-model
     clusters, the Apperception Engine's invented facts and learned
     symbols all give an unseen colour no value.
   - Building the answer in: a hand-split palette, or a special colour
     atom, makes a pass measure us, not the agent.
4. **Our tiles** (checked 2026-09-28):
   - One substitution of pixel values (red values to blue values) turns
     every red key and door into the blue one, and each object's outline
     is identical across colours. Colour is exactly a recurring
     difference.
   - But a key and its closed door share no pixel value: the key uses
     reds 255, 226, 198, 170 and 85; the door uses 192, 161 and 114. So
     "same colour" cannot be pixel equality. It has to be the
     correspondence the substitutions define. The open door shares
     shades with the key, and toggling links the open door to the closed
     one.
5. **Test design the papers suggest:**
   - PGM's graded splits: a withheld pair (blue key and blue door seen,
     never used together), a withheld colour (blue never seen), and a new
     shape.
   - A near miss: a wrong key at the withheld door stays shut.
   - A permuted world (each key opens a door of another colour), where
     the match rule must not win.
   - Training on 1, 2 and 3 colours, to show the break-even point.
   - A record of what we built in (Chollet).

| Paper (papi name) | What to take from it |
| ----------------- | -------------------- |
| `1812_02230` (Higgins et al. 2018) | Factors as subgroups of symmetry transformations. A factoring test: two families of transformations commute, each changing only its own part (red key → blue key → blue door equals red key → red door → blue door). Assumes the group is given |
| `an-algebraic-approach-to-abstraction-in-reinforcement-learni` (Ravindran 2004) | A symmetry is a relabelling that leaves the dynamics unchanged. It confirms a candidate attribute against our model and aligns colour across shapes. The same-colour orbit of (held, door) pairs (every pair a colour swap can reach) is the relational atom. Bayesian choice among candidate transformations needs a floor on exact models |
| `1904_00243` (Caselles-Dupré et al. 2019) | Still images cannot fix a transformation; transitions can. With exact pixels a substitution is explicit, and checking it against the effects model supplies the interaction. Its factoring was built by hand |
| `2002_11963` (van der Pol et al. 2020) | "Transform, then step" equals "step, then transform", as a working test. Negatives prevent collapse (our evidence penalty). Equivariance under actions gives structure within an object, not the key–door correspondence |
| `equivalence-notions-and-model-minimization-in-markov-decisio` (Givan et al.) | Our kinds are bisimulation groups: they group things, never pair members of one group with members of another. Colour is that pairing |
| `1412_2309` (Chalupka et al.) | An attribute is a difference that makes a difference to a given outcome: colour matters for opening, not for picking up. "X in view" atoms can be spurious correlates |
| `a-theory-of-the-discovery-and-predication-of-relational-conc` (DORA 2008) | Predicates are what survives intersection across compared pairs. Our kinds and rule outcomes can supply the comparison schedule DORA leaves open. Given features and dimensions |
| `1910_05065` (Doumas et al. 2022) | Same, more or less, read off any dimension as a generic relation; zero-shot transfer from Breakout to Pong by analogy. Its pixel front end codes x, y, size and RGB by hand |
| `gentner-structure-mapping` (1983), `gentner-markman-analogy-similarity` (1997), `structure-mapping-engine` (1989) | Alignment: an attribute is an alignable difference between things otherwise in correspondence. Candidate inferences are the transfer step (a 25-month-old with coloured keys and doors asks "Where the white door?"). All on hand-coded predicates |
| `2102_10717` (Mitchell 2021) | Survey of analogy in AI: every method is given its vocabulary. Shortcut warnings (a ResNet scores about 90% on RAVEN from the answer panels alone) |
| `1902_00120` (Hill et al. 2019) | Which relation a rule uses is identified only by contrasts that differ in that relation alone |
| `the-relational-bottleneck-as-an-inductive-bias-for-efficient`, `2304_00195`, `2012_14601` | Processing that sees only comparisons between objects transfers to new objects: ESBN is at least 95% correct with 95 of 100 images withheld. One relation per attribute. All trained on labelled tasks |
| `plotkin-inductive-generalization` (Plotkin 1970) | Least general generalisation: the same pair of differing terms becomes the same variable, which is how a match arises: opens(door(X)) ← held(key(X)). On atomic appearance names it gives "any key opens any door". It also runs on pixel arrays, where pixels differing only by a (red, blue) pair become one colour variable |
| `2008_07912` (ILP at 30), `2005_02259` (Popper) | Predicate invention scored by compression; a bias that bans constants. Popper is an offline check that size alone favours the relational rule once attribute facts exist. Background knowledge supplied and noise-free |
| `1910_02227` (Apperception Engine) | The cheapest unified theory, with rules that contain no constants. Groups of mutually exclusive invented predicates are formal attributes. Invented values are facts about each object, so an unseen colour gets none. The follow-up on raw pixels (Sokoban, 2021) is not in papi |
| `schema-networks-zero-shot-transfer-with-a-generative-causal`, `an-object-oriented-representation-for-efficient-reinforcemen` | Our current learner's relatives. DOORMAX's merge is a propositional generalisation: given a "held colour = front colour" term, it learns the match after three colours. Discovering the term is the problem |
| `kemp-structural-form`, `learning-systems-of-concepts-with-an-infinite-relational-mod` | A grid is a product of two forms, scored against a flat partition (shape × colour). The clique form says "the relation holds within clusters" for the whole form. The infinite relational model allows one partition per type and cannot state "diagonal" |
| `tenenbaum-griffiths-generalization` (2001) | Generalise widely along dimensions where examples vary, narrowly where they do not. Under weak sampling (the agent's own play), only the prior separates the relational rule from per-colour rules |
| `lake-bpl` (Lake et al. 2015) | How much a kind may vary is learned from other kinds. A new outline gets a modest prior of being a key, settled by one pick up |
| `dreamcoder` | Library abstraction by compression. The relational rule pays only with enough colours; score the factoring and the rules together |
| `2103_14230` (PrAE) | A rule's probability sums over the value assignments it allows, so rules can teach attributes. No attribute labels, but the attribute slots are given |
| `1807_04225` (PGM), `bongard-logo`, `1911_01547` (Chollet), `2210_09880` (ARGA) | Test designs: graded held-out splits (held-out shape–colour fell from 59.1% to 12.5%, chance 12.5%), near-miss negatives, a record of what was built in. ARGA's dynamic parameter binding is our match rule, over given attributes |

Earlier focus (cards 001–002, until card 002 has a decision):

- **Cards / rung:** 001 (architecture selection, before rung 1) and 002
  (rare-event sampling, rung 1).
- **Why these:** each is the source of a mechanism one of card 001's five
  candidates or card 002's samplers uses; the cards say which. Both cards
  serve the first part of GOAL.md's main insight: how close a goal is.
- **Until:** card 002 has a decision.

| Paper (papi name) | What to take from it |
| ----------------- | -------------------- |
| `2304_01203` (QRL) | Hub H's structure: encoder, latent transition, quasimetric d, actions scored by d(T(z, a), g); goals as sets of states; its transition loss in the learned quasimetric (round 2). Random-policy MountainCar: 85.5 ± 3.6 vs Q-learning 22.1, 5 seeds |
| `2208_08133` (MRN) | H's quasimetric head, by its Proposition 1; distance to a goal set is the minimum over members. Its RL results are weak evidence |
| `2211_15120` (IQE) | Better-evidenced quasimetric head, used by QRL; the alternative to MRN |
| `2402_15567` (HILP) | H's objective: action-free lower-expectile regression toward 1 + d on hindsight goals, EMA target |
| `ogbench` | Why expectile over QRL: GCIVL holds up on noisy and pixel data where QRL falls to near 0; E's Q-function expectile method is strong on noisy manipulation |
| `1707_01495` (HER) | Hindsight goals from later in the same trajectory ("future"); its goal space was supplied, which we do not do |
| `2110_09514` (LEXA) | Frames of other trajectories as far goals; learned temporal distance beats latent similarity when goals involve objects. Its on-policy distance does not transfer to random-action data |
| `leworldmodel` | H's transition loss: latent prediction with SIGReg, AdaLN action input; weak on low-diversity data |
| `dreamerv3` | Candidate A: pixel reconstruction carries the representation; recurrent state-space model for rung 2 |
| `latent-actions` | Candidate D: per-location discrete change codes with a decoder that sees the current state |
| `schema-networks-zero-shot-transfer-with-a-generative-causal` | Candidate D: persistence by default, sparse causes of change; not its supplied entities |
| `1711_00937` (VQ-VAE) | Candidate D: discrete codebook for event codes |
| `disco-rl` | Candidate D: a goal as a distribution fitted to examples; per-factor precision as agreement weights (it used 30–50 supplied examples) |
| `hiql` | Candidate C: a high-level head that outputs a latent subgoal (the encoding of the state k steps later), trained by advantage-weighted regression on one action-free goal value; k = 3 on pixel Procgen Maze, 25–50 elsewhere. Also candidate E's value. Its action-free value is unbiased only under deterministic dynamics |
| `universal-value-function-approximators` (UVFA) | Candidate E: one goal-conditioned value over states and goals |
| `1511_05952` (prioritised replay) | Card 002: sampling transitions by priority. Arms B and C are our priorities, not its TD error; we drop its importance correction |

## High Level Papers

These are papers to guide the architecture to adhere to the goal. It is useful to read these when thinking about candidate components or direction changes.

| Paper (papi name) | Component or test it shaped | What we borrowed | What we changed or did not take |
| ----------------- | --------------------------- | ---------------- | ------------------------------- |
| `dream-rsi` | Not yet; long-term self-improvement | Discovery history reused as a replay simulator to improve exploration cheaply | An LLM coding-agent setting; relevance to a pixel world model is still to be worked out |
| `temporal-distance-jepa` | The reachability head in card 001; later goal and planning rungs | Directed temporal cost as a planning signal; rollout-consistency loss (Push-T 85.3 → 60.0 without it, 3 seeds) | Needs expert demonstrations for its step-count labels; small gains over LeWM on most tasks |
| `2606_19594` (Saulus et al. 2026, *Unsupervised Causal Abstractions Discovery*) | Not yet; condition and object discovery | A high-level variable is a narrow point through which a group of low-level variables causes another group; "anchors" (low-level variables tied to one high-level variable only) make it identifiable | Needs supplied low-level variables, the number of high-level variables, and intervention regimes; tested on synthetic models and a small network, not pixels or agents |
| Gentner, *Structure-Mapping* (1983) | Not yet; relation transfer (rung 6) | | Not in papi; needs the PDF |
| McGovern & Barto, *Automatic Discovery of Subgoals in RL using Diverse Density* (2001) | Not yet; inferring a goal's conditions (rung 2) | Subgoals are the states common to successful trajectories and absent from failed ones: the second part of the main insight | Not in papi; needs the PDF |
| `rudder` (RUDDER) | Not yet; inferring a goal's conditions (rung 2) | Credit a delayed success to the steps that caused it, by return decomposition | Built on rewards; we would decompose reaching a goal condition instead |
| `align-rudder` | Not yet; conditions from demonstrations (rung 2, P16) | Align a few demonstrations to find the shared steps that lead to success (it mined a diamond in Minecraft, though rarely) | Its alignment runs on clustered states; ours would need learned states |
| `1602_02867` (VIN, Tamar et al. 2016) | Card 067's walking (version 15) | Value iteration as local updates on a map: each place's value from its neighbours, iterated to a fixed point | Not learned end to end: run on the agent's own believed map, walkability from recall and neighbours from the learned moves; only placements next to a changed one are updated |
| `1906_05253` (SoRB) | Not yet; deliberation (rung 4) | Plan over waypoints from the replay buffer, using a learned distance as edge weights: reasoning over a few states, not many simulated steps | Its waypoints are states, not conditions |
| Quinlan, *Learning Logical Definitions from Relations* (FOIL, 1990) | Card 005's condition finder | FOIL's gain for choosing the next condition of a rule: kept successes × gain in log precision, tolerant of a few wrong labels | Our atoms are variable values, one rule per way of succeeding; not in papi |
| `s-parse-autoencoders-f-ind-h-ighly-i-nter-pretable-f-eatures` (Cunningham et al. 2023) | Card 006: splitting a learned state into parts | A sparse autoencoder turns a vector into a few active directions from a large dictionary | Built for language models; papi has the title only, the text needs adding (arXiv 2309.08600) |
| `1511_01644` (Letham, Rudin, McCormick & Madigan, Bayesian rule lists, 2015) | Card 010's condition finder | A decision list whose rules are admitted by Bayesian evidence, each rule with a Beta-Bernoulli success rate | Our rules are over conditions of an attempted action; greedy search with look-ahead instead of their sampler |
| `2203_09634` (predicate invention) | Not yet; deliberation over conditions (rung 4) | Learned predicates and operators with preconditions and effects, planned over abstractly (bilevel planning): the "if A, then B, then goal" of P21 | Predicates come from a grammar over supplied object features, and goals are supplied predicates; both conflict with C1 |
| `strips` (Fikes & Nilsson 1971), `cs_9401101` (Nilsson, teleo-reactive programs) | Card 029's subgoals | Means-ends analysis: an operator's unmet preconditions become subgoals; re-evaluate from the top at every step | Operators counted from the agent's own pixels (card 028), not written by hand |
| `1106_0243` (Koehler and Hoffmann 2000, reasonable goal orderings) | Card 073's orders between needs | B is ordered before A when, once A holds, B cannot be reached without at least temporarily destroying A (Definition 8); a sufficient test on conditions alone: every action that achieves B has a precondition that cannot hold while A holds (Definition 10) | Their goals are STRIPS atoms and "cannot hold together" comes from planning-graph analysis; here the needs are recall's conditions, judged in the situation A's chain produces, at every depth of the chain |
