# Literature

Full text and notes live in paperpipe; use the `papi` name to look them up.

## Current focus

The papers agents should work from until the user changes or requests a change to this section. Read these (via `papi`) before proposing designs for the listed cards.

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
| Saulus, *Unsupervised Causal Abstraction Discovery* | Not yet | | Not in papi; needs an arXiv ID or PDF |
| Gentner, *Structure-Mapping* (1983) | Not yet; relation transfer (rung 6) | | Not in papi; needs the PDF |
| McGovern & Barto, *Automatic Discovery of Subgoals in RL using Diverse Density* (2001) | Not yet; inferring a goal's conditions (rung 2) | Subgoals are the states common to successful trajectories and absent from failed ones: the second part of the main insight | Not in papi; needs the PDF |
| `rudder` (RUDDER) | Not yet; inferring a goal's conditions (rung 2) | Credit a delayed success to the steps that caused it, by return decomposition | Built on rewards; we would decompose reaching a goal condition instead |
| `align-rudder` | Not yet; conditions from demonstrations (rung 2, P16) | Align a few demonstrations to find the shared steps that lead to success (it mined a diamond in Minecraft, though rarely) | Its alignment runs on clustered states; ours would need learned states |
| `1906_05253` (SoRB) | Not yet; deliberation (rung 4) | Plan over waypoints from the replay buffer, using a learned distance as edge weights: reasoning over a few states, not many simulated steps | Its waypoints are states, not conditions |
| `2203_09634` (predicate invention) | Not yet; deliberation over conditions (rung 4) | Learned predicates and operators with preconditions and effects, planned over abstractly (bilevel planning): the "if A, then B, then goal" of P21 | Predicates come from a grammar over supplied object features, and goals are supplied predicates; both conflict with C1 |
