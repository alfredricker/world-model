# Energy modules: cards 146–160

**Set aside by the user on 2026-11-23. It may be picked up later.** This
note says why the series was started, what it showed, what it did not, and
where to begin if it is resumed. The series ran from 2026-11-19 to
2026-11-21 and has 18 cards.

## Motivation

The user's idea (2026-11-19): build the sample-efficient memory into the
model itself, with no separate memory component. The model is layers of
**energy modules**. A module is a small unit with a scalar energy over its
state; it takes a patch of input and settles to a low-energy state, called
an attractor. Attractors hold features at the bottom, objects and concepts
in the middle, and at the top the computation that picks an action. Each
level segments its own events. Later conditions from the user: no
hand-built geometric invariance, and strip away what the model is fed
(2026-11-20).

It followed two weeks in which learned representations kept failing their
bars on OGBench (cards
[141](../141-representation-test-bed/card.md)–[145](../145-a-state-that-changes-only-when-the-scene-does/card.md)):
reconstruction kept the background and lost the cube.

The benefits the literature claims for this form, with the GOAL property
each would serve:

| Claimed benefit | Source | Property |
|---|---|---|
| A pattern is written to an attractor memory in one shot, so rare events need few examples | modern Hopfield networks | P10, P19 |
| Energies add, so terms learned apart combine for new layouts and combinations | Hinton 2002; Du, Li and Mordatch 2020 | P3, P8 |
| Sparse pairwise couplings can be read as rules | CRAM (arXiv 2611.17309): an L1 penalty leaves exactly Sudoku's 810 constraint pairs | P12 |
| Settling gives feedback, filling-in of what is hidden, and more computation for harder cases | Energy-Based Reasoners (arXiv 2611.08977) | P2, P5, P21 |
| One mechanism for binding, memory and reasoning | AKOrN (Miyato et al. 2025) | C2 |

Every published win is on variables that are given, discrete and fully
visible (Sudoku, mazes). No paper shows the form on variables a network
had to find in images. The papers are in
[LITERATURE.md](../../LITERATURE.md), section "Energy modules and
binding"; the design notes are sections 6–7 of
[the write-up](../../theory/thousand_brains_energy_formulation.md).

## The cards

Test worlds: Tetrominoes (a published object-discovery benchmark), our own
sprite world (a hand that can push, take and release coloured shapes, seen
as pixels), a world built from parts, a stream of words, and MiniGrid.

| Card | Question | Verdict | Main number |
|---|---|---|---|
| [146](146-grouping-as-settling-on-a-known-benchmark/card.md), [146.1](146.1-grouping-as-settling-colder/card.md) | Can settling group patches into objects on a published benchmark? | uninterpretable, then fail | 65.2 against 89.5 for plain clustering as declared; 87.8 against 86.5 after tuning the threshold and temperature |
| [147](147-binding-from-interaction-in-a-sprite-world/card.md) | Can points be bound into objects from how they move as the hand acts? | pass | 98.4% against 39.9% without evidence kept over time |
| [148](148-the-action-as-an-output-token/card.md) | Can the action be an output token that settles to what gives a wanted effect? | pass | 95.6% after 100 steps of experience |
| [149](149-a-sequence-of-actions-by-recurrence/card.md), [149.1](149.1-testing-a-no-by-trying/card.md) | Does a sequence of actions come out by using the energy step by step? | fail; 149.1 abandoned | in order 96.9%, but 58.8% where the records hold no such combination |
| [150](150-three-feed-forward-layers-of-energy-modules/card.md) | Do three stacked layers run on pixels? | pass | 99.5%, however the layers are clocked |
| [151](151-layers-of-attractor-modules-in-a-world-of-parts/card.md) | What should a layer pass upward? | fail as worded | every pattern's energy 92.0% against 35.3% for the winner alone |
| [152](152-an-object-module-that-knows-what-an-object-is-and-does/card.md) | Can one object module hold where, what and what it does? | pass | a never-met object handled from its looks, 91.2% |
| [153](153-kinds-found-by-one-energy-in-two-domains/card.md), [153.1](153.1-dynamics-classes-from-appearance-and-transitions/card.md) | Can one energy find kinds of behaviour in sprites and in words? | fail, then pass | word classes from 100 sentences 38.2 → 84.3% once objects that look alike share proposals |
| [154](154-feedback-from-object-modules-to-binding/card.md) | Does feedback from a recognised object help binding? | fail (2 of 3), kept | a still, known object bound 0 → 96.6% |
| [155](155-object-modules-fed-by-binding/card.md) | Do object modules work without the simulator's masks? | fail (2 of 3), kept | a never-met object is taken in 39% of tries against 100% with masks |
| [156](156-appearance-couplings-taught-by-the-agents-own-action/card.md) | Can motion teach appearance couplings that bind before anything moves? | fail (1 of 3), revise | never-met objects bound 0 → 96.9% |
| [157](157-transitions-read-from-the-binding-layers-own-states/card.md) | Can transitions be learned from the model's own tokens? | fail (2 of 3), kept | 86.4% against 85.0% with the simulator's tokens |
| [158](158-events-as-changes-of-an-objects-latent-state/card.md) | Are events changes of an object's latent state, and can their conditions be read off? | fail (1 of 3), kept | event F1 71.3 against 53.5 read from what is seen; conditions 30 of 40 |
| [159](159-contact-as-a-learned-relation-of-places/card.md) | Can contact be a learned relation between places? | fail (0 of 3), stop | an object bound by appearance has its true extent in about half of the steps |
| [160](160-minigrid-goals-against-ordinary-baselines/card.md) | Does the learned energy reach goals in MiniGrid, against ordinary baselines? | fail (1 of 3), revise | 44.5% of layouts against 72.5% for a perceptron in the same agent |

## What worked

- **Binding from the agent's own interaction** (147, 154, 156). Points
  that move together as the hand acts become one object, provided evidence
  is kept over time. Feedback from a stored pattern binds a known object
  that is still or part hidden. Couplings predicted from appearance, taught
  only by motion, bind a never-met object before it moves. This was the
  user's condition for going on, and it was met in the sprite world.
- **A stack of modules runs on pixels** (150, 151), and what to pass
  upward has a clear answer: the energy of every stored pattern, not the
  winning pattern.
- **Attractors written in one shot** are recognised at a new place and
  half hidden (152: 100%).
- **"Held" as a latent state** (157, 158). No frame shows that an object
  is held; inferring it as a hidden state placed events better than
  reading them from motion.
- **Acting by chaining learned conditions backward from a goal** (160).
  From 700,000 reward-free random steps the agent solved DoorKey-8x8 and
  16x16 in 99% of new layouts. PPO, trained with the reward for the same
  number of steps, solved 6% and 0.3%. Told MiniGrid's true rules, the
  same agent solved all five test tasks. **This result belongs to the
  chaining agent, not to the energy:** a perceptron in the energy's place
  did the same.

## What did not work

- **Against an ordinary network the energy lost.** Card 160 is the only
  test on equal terms. In the same agent, on the same experience, a
  perceptron solved 72.5% of layouts and the sparse energy 44.5% (57.9%
  after a repair to the planner made after the test).
- **Each module learns alone.** "Hands must be empty" was learned per
  object and was missing for 5 to 8 of 24 objects in card 160. Card 153.1
  had already needed objects that look alike to share what they learn.
- **A sparse model splits credit when one fact enters twice.** A door's
  opening entered as its new look and as a memory that the closed door had
  gone; the planner imagined only the first and found no plan (160).
- **No sign of sample efficiency.** With half the experience DoorKey fell
  from 99% to 7%. Whole records in a Hopfield memory needed about ten
  times the experience of pairwise couplings (148).
- **Settling did not beat simpler read-outs.** On Tetrominoes it matched
  plain clustering only after tuning (146.1). Where the records held no
  such combination, the energy answered arbitrarily (149).
- **Perception without masks stayed partial.** Objects bound by appearance
  covered about 70% of their true extent, and every relation between
  places failed downstream (159).
- **Most cards failed their own bars.** Cards 153 to 160 each failed at
  least one criterion, except 153.1.

## What was never tested

Cards 146 to 159 were compared with trivial baselines (no memory, one kind
for all objects), never with an ordinary network given the same inputs.
They show that each part is needed inside this design. They do not show
that the design beats another. Card 160 tested only half of the idea: its
"energy" is a sparse softmax regression per object, with no settling and
no attractors. Composition, sample efficiency and settling under a partial
view were never measured against a baseline.

The earlier MiniGrid agent, recall over stored tries with a network prior
(architecture version 27, cards 001–112 in [the archive](../archive/)),
solved DoorKey-8x8, BlockedUnlockPickup and ObstructedMaze-Full at 100%
through a 7 × 7 occluded view. It had more hand-built structure and about
3.2 million steps of experience in each test world, so the two are not
comparable as they stand.

## If this is picked up again

1. **Run the comparison on equal terms first.** MiniGrid, the agent and
   tokens of card 160, three models of effects underneath: the sparse
   energy, a perceptron, and recall with a network prior. Score success
   against the amount of experience (P19), a held-out colour or
   combination (P3, P8), and conditions named right (P12). Declare in
   advance that the energy is dropped if it wins none by a set margin.
2. **Repair the two faults card 160 found before that run:** a token that
   changes in place must not also count as gone from view, and codes that
   behave alike should share conditions (card 153.1's classes).
3. **Test settling where it could matter:** MiniGrid's 7 × 7 occluded
   view, where belief has something to fill in. Under a full view there is
   nothing to settle.
4. **Unexplained:** KeyCorridor at 8% after the planner repair; DoorKey
   falling at 175,000 steps of experience; the perceptron's 41 to 46% on
   the two pickup tasks.

## Where things are

- Code: `tools/card146/` to `tools/card160/`. The chaining agent, the
  rules agent, the perceptron and PPO are in `tools/card160/` and do not
  depend on the energy.
- Run outputs: `runs/146/` to `runs/160/` (not in git).
- Results as graph nodes: `graph/result/146-*` to `160-*`, and the
  principles they tested under `graph/principle/`.
- Lessons: [LESSONS.md](../../LESSONS.md), "Energy forms (cards 146–159)"
  and "Give a sparse model each fact once".
