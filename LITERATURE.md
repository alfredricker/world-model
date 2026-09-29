# Literature

Full text and notes live in paperpipe; use the `papi` name to look them up.

## Current focus

The papers agents should work from until the user changes or requests a change to this section. Read these (via `papi`) before proposing designs for the listed cards.

- **Card / rung:** theory sessions after card 026, then card 027 (kinds
  from what actions do in front; before rung 1): how the agent discovers
  conditions and the objects they are about, from three signals: contrast,
  relations and causes. Changed with the user, 2026-09-28.
- **Why these:** card 026 left conditions and their targets supplied by
  exact computation. These papers define abstractions by what they look
  like, what they relate to, or what they make a difference to.
- **Until:** card 027 has a decision.

| Paper (papi name) | What to take from it |
| ----------------- | -------------------- |
| `1412_2309`, `1512_07942` (Chalupka et al.) | A macro-variable is a cell of the causal partition: situations with the same effect of an intervention. Prediction from partial views gives finer classes than the causal ones, never coarser; the smallest change that flips the outcome locates the cause in the image |
| `1812_03789`, `1707_00819` (causal abstraction) | Validity test: every way of making a high-level variable true must have the same effect (total cholesterol fails; "holding a key" against "holding the matching key" is ours) |
| `2606_19594` (UCAD) | High-level variables as narrow points in the causal graph, identified by anchors |
| `cs_9401101` (teleo-reactive programs) | Our tree almost exactly: each node the weakest condition from which its action achieves the parent; parameters bound at run time; Nilsson proposes growing the tree where no node holds |
| `from-skills-to-symbols-learning-symbolic-representations-for` | The symbols needed and sufficient for planning are set by the skills: their initiation sets and effects. Our conditions are initiation sets |
| `deepsym`, `2309_00889` | Object kinds and relations as whatever discrete codes predict action effects; objects supplied by perception |
| `equivalence-notions-and-model-minimization-in-markov-decisio` (Givan, Dean & Greig 2003) | Card 027's kinds: the coarsest grouping in which every action has the same effect and leads into the same groups (stochastic bisimulation), found by splitting |
| `2205_08515` (EISEN) | Objects as what moves together, from pairwise affinities without slots; the agent's own motion explained away first |
| `learning-systems-of-concepts-with-an-infinite-relational-mod` (IRM), `a-theory-of-the-discovery-and-predication-of-relational-conc` (DORA), `the-relational-bottleneck-as-an-inductive-bias-for-efficient` | Kinds as sets of things that relate alike; properties before relations; relations as comparisons between learned codes, which transfer to new members. All start from given units |

Previous focus (cards 001–002, until card 002 has a decision):

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
| `1906_05253` (SoRB) | Not yet; deliberation (rung 4) | Plan over waypoints from the replay buffer, using a learned distance as edge weights: reasoning over a few states, not many simulated steps | Its waypoints are states, not conditions |
| Quinlan, *Learning Logical Definitions from Relations* (FOIL, 1990) | Card 005's condition finder | FOIL's gain for choosing the next condition of a rule: kept successes × gain in log precision, tolerant of a few wrong labels | Our atoms are variable values, one rule per way of succeeding; not in papi |
| `s-parse-autoencoders-f-ind-h-ighly-i-nter-pretable-f-eatures` (Cunningham et al. 2023) | Card 006: splitting a learned state into parts | A sparse autoencoder turns a vector into a few active directions from a large dictionary | Built for language models; papi has the title only, the text needs adding (arXiv 2309.08600) |
| Letham, Rudin, McCormick & Madigan, *Interpretable Classifiers Using Rules and Bayesian Analysis* (Bayesian rule lists, 2015) | Card 010's condition finder | A decision list whose rules are admitted by Bayesian evidence, each rule with a Beta-Bernoulli success rate | Our rules are over conditions of an attempted action; greedy search with look-ahead instead of their sampler; not in papi |
| `2203_09634` (predicate invention) | Not yet; deliberation over conditions (rung 4) | Learned predicates and operators with preconditions and effects, planned over abstractly (bilevel planning): the "if A, then B, then goal" of P21 | Predicates come from a grammar over supplied object features, and goals are supplied predicates; both conflict with C1 |
