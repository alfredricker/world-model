# Literature

Full text and notes live in paperpipe; use the `papi` name to look them up.

## Current focus

The papers agents should work from until the user changes or requests a change to this section. Read these (via `papi`) before proposing designs for the listed cards.

- **Card / rung:** card 028 (one learned model of what every action does,
  moves included; every condition of the tree computed on the agent's own
  predictions; before rung 1). Changed with the user, 2026-09-28.
- **Why these:** they learn action models (when an action works and what
  it changes) over objects from experience, and plan with them. Their
  objects, attributes or predicates are supplied; ours are appearances and
  kinds counted from pixels (card 027).
- **Until:** card 028 has a decision.

| Paper (papi name) | What to take from it |
| ----------------- | -------------------- |
| `cs_9401101` (Nilsson, teleo-reactive programs) | Our tree: each node the weakest condition from which its action achieves the parent; the thing acted on is bound at run time |
| `from-skills-to-symbols-learning-symbolic-representations-for` (Konidaris et al.) | The symbols needed for planning are set by the skills: their initiation sets (our conditions) and effects (our learned effects) |
| `1905_12006`, `2205_02092` (James, Rosman & Konidaris) | Learn those symbols in the agent's own egocentric space, so rules carry across tasks, then bind them to a task's objects; ego- and object-centric observations. Our facts: things by kind, and where they are relative to the agent |
| `1110_2211` (Pasula, Zettlemoyer & Kaelbling) | Effects that happen only sometimes: a rule is an action, a context and outcomes with probabilities (plus a noise outcome), found by greedy search scored by likelihood minus complexity; rules name objects by their relation to the one acted on |
| `an-object-oriented-representation-for-efficient-reinforcemen` (Diuk, Cohen & Littman, OO-MDPs) | Deterministic effects on object attributes, conditioned on conjunctions of simple relations, learned from few examples; objects and relations supplied |
| `schema-networks-zero-shot-transfer-with-a-generative-causal` | Rules from entity attributes and the action to each attribute's next value, used for planning by inference from the goal; entities supplied |
| `1511_01644` (Letham et al., Bayesian rule lists) | Card 010's rule finder, reused for when an effect happens: rules admitted by Bayesian evidence |
| `2203_09634`, `2603_08599` | Learned predicates and operators planned over abstractly: predicates from a grammar over supplied features (2203_09634); probabilistic rules from effect predictors, checked by a continuous model (2603_08599) |

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
| `1511_01644` (Letham, Rudin, McCormick & Madigan, Bayesian rule lists, 2015) | Card 010's condition finder | A decision list whose rules are admitted by Bayesian evidence, each rule with a Beta-Bernoulli success rate | Our rules are over conditions of an attempted action; greedy search with look-ahead instead of their sampler |
| `2203_09634` (predicate invention) | Not yet; deliberation over conditions (rung 4) | Learned predicates and operators with preconditions and effects, planned over abstractly (bilevel planning): the "if A, then B, then goal" of P21 | Predicates come from a grammar over supplied object features, and goals are supplied predicates; both conflict with C1 |
