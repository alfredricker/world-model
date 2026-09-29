# Literature

Full text and notes live in paperpipe; use the `papi` name to look them up.

## Current focus

The papers agents should work from until the user changes or requests a change to this section. Read these (via `papi`) before proposing designs for the listed cards.

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
| `strips` (Fikes & Nilsson 1971), `cs_9401101` (Nilsson, teleo-reactive programs) | Card 029's subgoals | Means-ends analysis: an operator's unmet preconditions become subgoals; re-evaluate from the top at every step | Operators counted from the agent's own pixels (card 028), not written by hand |
