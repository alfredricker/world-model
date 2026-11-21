# Thousand Brains Energy Formulation for Conditional Planning

## Architectural Overview

This document formalizes a continuous, energy-based generalization of a non-generative conditional planner inspired by Jeff Hawkins' Thousand Brains Theory and Energy-Based Reasoners. Rather than rolling out dense pixel or latent trajectories forward in time, the architecture divides inference across two coupled energy timescales:

1. **Layer 1 (Cortical Binding & Event Segmentation):** A population of autonomous energy modules binds sensory patches to object identity prototypes and reference-frame coordinates. Routine navigation updates reference frames smoothly via learned affine transformations, while sparse $\ell_1$ switching penalties carve continuous sensorimotor experience into discrete, causally meaningful event boundaries.
2. **Layer $L$ (Energy-Based Backward Chaining):** A recurrent reasoning layer operates over a depth-$D$ stack of latent subgoal conditions. Clamping the top of the stack to a goal condition and the bottom to the current belief state causes backward chaining, precondition protection, cycle avoidance, and epistemic exploration triggers to emerge through energy minimization.

---

## 1. Layer 1: Cortical Binding and Reference-Frame Modules

Let the belief state $s_t$ consist of $N$ cortical modules. Each module $i \in \{1, \dots, N\}$ maintains a structured state tuple $(f_i, w_i)$:

* **What ($f_i \in \mathbb{R}^{d_f}$):** A continuous identity embedding representing the bound concept or object prototype (with a designated null attractor $f_\emptyset$ for unobserved space).
* **Where ($w_i \in \mathbb{R}^{d_w}$):** A continuous coordinate in an egocentric or object-centric reference frame, augmented with a dedicated hand/effector frame coordinate $w_{\mathcal{h}}$.

### Local Binding Energy

At timestep $t$, given an observation $o_t$ and previous action $a_{t-1}$, each module minimizes a local binding energy $E_{\text{bind}, i}$ that balances sensory reconstruction at the predicted coordinate $w_i$, path integration from the previous step, and associative memory attraction:

$$E_{\text{bind}, i}(f_i, w_i; o_t, a_{t-1}) = \frac{m_i(w_i)}{2} \left\| E_\theta(o_t(w_i)) - f_i \right\|_2^2 + \frac{\lambda_w}{2} \left\| w_i - \left(M_{a_{t-1}} w_{i, t-1} + b_{a_{t-1}}\right) \right\|_2^2 + \Phi(f_i)$$

* $E_\theta(o_t(w_i)) \in \mathbb{R}^{d_f}$ is the encoder output sampled at spatial offset $w_i$.
* $m_i(w_i) \in \{0, 1\}$ is a visibility mask indicating whether coordinate $w_i$ falls inside the agent's active sensor field (e.g., the $7 \times 7$ view window) and is unoccluded. When $m_i(w_i) = 0$, the sensory term vanishes, allowing module $i$ to retain $f_i$ in working memory while continuing to path-integrate $w_i \leftarrow M_a w_i + b_a$.
* $(M_a, b_a)$ is the learned affine transformation associated with action $a$.
* $\Phi(f_i)$ is a continuous Hopfield associative energy over a learned bank of $K$ concept prototypes $\{\xi_k\}_{k=1}^K$:

$$\Phi(f_i) = -\frac{1}{\beta_{\text{mem}}} \log \sum_{k=1}^{K} \exp\left(\beta_{\text{mem}} \langle f_i, \xi_k \rangle\right) + \frac{1}{2} \| f_i \|_2^2$$

### Lateral Consensus Voting

To resolve local sensory ambiguity and maintain structural coherence across modules (such as ensuring an object held in the hand frame $w_i = w_{\mathcal{h}}$ does not simultaneously occupy a floor tile), modules interact via pairwise compatibility matrices $W_{ij}$:

$$E_{\text{vote}}^{(1)}(s_t) = -\frac{1}{2} \sum_{i \neq j} \left\langle \psi(f_i, w_i), \, W_{ij} \psi(f_j, w_j) \right\rangle$$

where $\psi(f_i, w_i)$ projects the bound identity and reference-frame coordinate into a shared voting subspace. The total Layer 1 energy is:

$$E^{(1)}(s_t; o_t, a_{t-1}) = \sum_{i=1}^{N} E_{\text{bind}, i}(f_i, w_i; o_t, a_{t-1}) + E_{\text{vote}}^{(1)}(s_t)$$

Fast inner-loop gradient descent on $E^{(1)}$ settles the layer into a bound belief state $s_t^* = \{(f_i^*, w_i^*)\}_{i=1}^N$.

---

## 2. Temporal Abstraction via Event Segmentation

To prevent upper planning layers from reasoning over dense clock ticks, Layer 1 separates smooth spatial movement from discrete causal state transitions. Define pairwise relational vectors $r_{ij, t} \in \mathbb{R}^{d_r}$ between modules (encoding relationships such as *holding*, *blocking*, *adjacent*, or *inside*) computed via a relational readout $r_{ij, t} = \rho_\phi(f_{i,t}^*, w_{i,t}^*, f_{j,t}^*, w_{j,t}^*)$.

We govern temporal transitions between $s_{t-1}$ and $s_t$ under action $a_t$ with an event-segmentation energy:

$$E_{\text{event}}(s_t, s_{t-1}, a_t) = \mathcal{L}_{\text{dyn}}(s_t \mid s_{t-1}, a_t) + \gamma \sum_{i=1}^{N} \| f_{i,t} - f_{i,t-1} \|_1 + \beta \sum_{i < j} \| r_{ij,t} - r_{ij,t-1} \|_1$$

* **Smooth Navigation Regime:** When the agent moves through empty space, updating $w_{i,t} = M_{a_t} w_{i,t-1} + b_{a_t}$ keeps the transition residual $\mathcal{L}_{\text{dyn}}$ near zero. Because changing $f_i$ or $r_{ij}$ incurs $\ell_1$ penalties $\gamma$ and $\beta$, non-spatial states remain clamped ($f_{i,t} = f_{i,t-1}$, $r_{ij,t} = r_{ij,t-1}$), and no event boundary is emitted.
* **Event Boundary Regime:** When an action induces a causal interaction (e.g., picking up a key, unlocking a door, opening a box), keeping $f_{i,t}$ and $r_{ij,t}$ constant causes $\mathcal{L}_{\text{dyn}}$ to spike above the thresholds $(\gamma, \beta)$. Minimizing $E_{\text{event}}$ forces a sparse step jump $\Delta f_i \neq 0$ or $\Delta r_{ij} \neq 0$.

Each sparse jump defines an extracted condition transition $e_\tau = (c_{\text{pre}}, a_\tau, c_{\text{post}})$ indexed by discrete event time $\tau$, populated directly into the agent's episodic event memory.

---

## 3. Layer $L$: Energy-Based Backward Chaining

Let a condition state $u \in \mathbb{R}^{d_u}$ represent a target relational or object configuration over the bound belief state. To replace explicit recursive tree search (`Solve(c, s, d, C, P)`) with continuous energy relaxation, Layer $L$ maintains a fixed-depth stack of $D$ latent subgoal conditions $u^{(1)}, \dots, u^{(D)}$ and associated candidate transitions $a^{(1)}, \dots, a^{(D)}$, with the top level clamped to the active goal condition $u^{(0)} = g$.

The planner minimizes the total planning energy:

$$E_{\text{plan}}(u^{(1:D)}, a^{(1:D)}; s_t^*, g) = \sum_{d=0}^{D-1} E_{\text{chain}}(u^{(d)}, a^{(d)}, u^{(d+1)}) + E_{\text{protect}}(u^{(0:D)}, a^{(1:D)}) + E_{\text{cycle}}(u^{(0:D)}) + E_{\text{walk}}(u^{(D)}, s_t^*)$$

### Energy Components

1. **Backward Chaining Energy ($E_{\text{chain}}$):** Parameterized by the learned effect model $F_\omega$, $E_{\text{chain}}$ assigns low energy if and only if executing transition $a^{(d)}$ from precondition state $u^{(d+1)}$ produces $u^{(d)}$:

$$E_{\text{chain}}(u^{(d)}, a^{(d)}, u^{(d+1)}) = \left\| u^{(d)} - F_\omega(u^{(d+1)}, a^{(d)}) \right\|_2^2 + \Omega_{\text{effect}}(u^{(d+1)}, a^{(d)}, u^{(d)})$$

where $\Omega_{\text{effect}}$ incorporates kernel-weighted retrieval over episodic event memory so transitions observed in the current episode exert strong attraction. Clamping $u^{(0)} = g$ propagates gradients downward from $d = 0$ to $d = D$, pulling each deeper level $u^{(d+1)}$ into a valid precondition attractor for $u^{(d)}$.

2. **Protected Condition Energy ($E_{\text{protect}}$):** Analogous to the protected set $P$ in symbolic backward chaining, $E_{\text{protect}}$ penalizes any lower-level transition $(u^{(d+1)}, a^{(d)})$ that clobbers a condition already established at a shallower level $d' < d$:

$$E_{\text{protect}}(u^{(0:D)}, a^{(1:D)}) = \lambda_{\text{prot}} \sum_{d=0}^{D-1} \sum_{d'=0}^{d-1} \max\left(0, \, \left\langle u^{(d')}, \, u^{(d+1)} - F_\omega(u^{(d+1)}, a^{(d)}) \right\rangle \right)$$

3. **Cycle Avoidance Energy ($E_{\text{cycle}}$):** Analogous to ancestor tracking $C \cup \{c\}$, a repulsive Gaussian kernel prevents subgoal chains from looping back onto identical states across different stack depths:

$$E_{\text{cycle}}(u^{(0:D)}) = \sum_{0 \le d < d' \le D} \kappa \exp\left( -\frac{\| u^{(d)} - u^{(d')} \|_2^2}{\sigma^2} \right)$$

4. **Spatial Grounding / Walking Energy ($E_{\text{walk}}$):** Couples the deepest subgoal $u^{(D)}$ to the current belief state $s_t^*$ by measuring the minimum path-integration distance required to bring the required target objects into valid interaction coordinates (such as facing the target tile or holding the required item):

$$E_{\text{walk}}(u^{(D)}, s_t^*) = \lambda_{\text{walk}} \, \mathcal{D}_{\text{place}}\!\left(s_t^*, \, u^{(D)}\right)$$

where $\mathcal{D}_{\text{place}}$ is the shortest-path metric over the affine placement graph $w \to M_a w + b_a$.

### Execution, Intention Persistence, and Epistemic Halting

At each environment step $t$, Layer $L$ runs $K_{\text{plan}}$ steps of gradient descent on $E_{\text{plan}}$:

$$u^{(d)} \leftarrow u^{(d)} - \eta_u \nabla_{u^{(d)}} E_{\text{plan}}, \qquad d \in \{1, \dots, D\}$$

* **Intention Persistence:** Initializing $u^{(1:D)}_t$ from the converged subgoal stack $u^{*(1:D)}_{t-1}$ of the previous timestep provides warm-start hysteresis, maintaining a consistent multi-step intention unless unexpected observations raise $E_{\text{plan}}$.
* **Automatic Exploration Fallback (`No plan`):** When a required object is occluded or inside an unopened container (so $u^{(D)}$ must bind to an unknown token $f_\emptyset$), no valid chain of known effects can connect $s_t^*$ to $g$. Consequently, relaxation terminates with high residual energy:

$$E_{\text{plan}}(u^{*(1:D)}, a^{*(1:D)}; s_t^*, g) > \epsilon_{\text{halt}}$$

This residual energy threshold acts as an intrinsic halting signal to suspend goal chaining and dispatch the agent toward the nearest uninspected frontier token ($m_i(w_i) = 0$) or unknown container.

---

## 4. Local Training Objectives

To train the system from random play and online episode trials without backpropagating gradients through unrolled planning loops:

1. **Affine Placement Operators $(M_a, b_a)$:** Trained via self-supervised spatial consistency across consecutive visible frames where $f_{i,t}^* \approx f_{i,t-1}^*$:

$$\mathcal{L}_{\text{place}} = \sum_{i : m_i(w_{i,t}) = 1} \left\| w_{i,t}^* - \left(M_{a_{t-1}} w_{i,t-1}^* + b_{a_{t-1}}\right) \right\|_2^2$$

2. **Prototype & Relational Encoders $(\theta, \xi, \phi)$:** Trained using contrastive predictive coding across event boundaries, pulling together pre/post embeddings of identical object instances while separating distinct object classes and relational states.
3. **Transition & Chaining Model ($F_\omega$):** Trained directly on segmented event tuples $(c_{\text{pre}}, a_\tau, c_{\text{post}})$ extracted by $E_{\text{event}}$ using a contrastive energy margin loss against negative actions $a^-$ and non-achieving states $c^-$:

$$\mathcal{L}_{\text{chain}}(\omega) = E_{\text{chain}}(c_{\text{post}}, a_\tau, c_{\text{pre}}) + \max\left(0, \, \delta_{\text{margin}} - E_{\text{chain}}(c_{\text{post}}, a^-, c_{\text{pre}})\right)$$

---

## 5. Experimental Verification Checklist

When implementing and benchmarking this formulation on MiniGrid and continuous extensions:

1. **Event Sparsity Calibration:** Sweep $\gamma$ and $\beta$ during random play in `DoorKey-8x8` and `BlockedUnlockPickup`. Verify that the event trigger rate stays near zero during pure translation/rotation and spikes selectively on `pickup`, `drop`, `toggle`, and `unlock` transitions.
2. **Residual Energy Calibration:** Measure the distribution of converged $E_{\text{plan}}^*$ in `ObstructedMaze-Full` before and after discovering the target key inside a closed box. Confirm a clean separation around $\epsilon_{\text{halt}}$ so the agent transitions reliably between frontier exploration and goal chaining.
3. **Non-Clobbering Ablation:** Compare episode success rates on `BlockedUnlockPickup` with $\lambda_{\text{prot}} > 0$ versus $\lambda_{\text{prot}} = 0$ to verify that $E_{\text{protect}}$ prevents the planner from dropping a blocking ball back in front of an unlocked doorway.
4. **Warm-Start Convergence Speed:** Measure the number of inner-loop gradient steps $K_{\text{plan}}$ required to reach $E_{\text{plan}} < \epsilon_{\text{halt}}$ with warm-started $u^{(1:D)}_{t-1}$ initialization versus cold initialization at each step.
---

## 6. Review notes (2026-11-19, thinking session; no code, no runs)

Written by Claude with the user, after reading sections 1–5 against two new
papers (Energy-Based Reasoners, arXiv 2611.08977; CRAM, arXiv 2611.17309)
and this project's results. The papers, and those found in the search that
followed, are in [LITERATURE.md](../LITERATURE.md), section "Energy modules
and binding".

### The idea, in the user's words

Build the sample-efficient memory into the model, with no separate memory
component: layers of energy modules that map sensory patches to attractor
states, and higher layers that think recurrently over the lower layers'
states. Attractor states should hold features at low levels, concepts at
intermediate levels, and at the highest level the computation that produces
the right action. **The user's condition (2026-11-19):** if such states
cannot be produced, the module formulation has no way forward, so binding
is examined first.

### Three claims, and what supports each

| Claim | Evidence so far |
|---|---|
| Memory can be attractors inside a module instead of a separate store | Retrieval from a modern Hopfield energy is attention over stored patterns, and a new pattern is written in one shot. One store per module matches our result that recall keyed per object reads 90.3% against 57.9% for one flat key (cards 132–134.1) |
| A higher level can reason by settling over the lower level's states | CRAM: a pairwise energy learned by a local rule solves Sudoku from 30 puzzles and its weights are the rules. Only shown where the states are given, discrete and fully visible |
| Modules can bind patches to those states | No support in either paper. Our own attempts failed their bars (cards 116–116.2, 119, 124, 141–145): reconstruction keeps the background and loses the cube |

Two different problems are called binding. One is deciding which patches
make one object. The other is tying an identity to a place or a role in
memory. Hopfield networks, the Tolman-Eichenbaum Machine and Monty address
the second and assume the first is solved or supply it by hand.

### Event segmentation at every level (the user's amendment)

Sections 1–3 segment events only in the bottom layer and make the top
layer a different machine, a stack of subgoals of fixed depth. The user's
point: higher levels understand more abstract transitions, so each level
should segment its own events. The form this suggests is one module
repeated at every level:

- a state that changes only at a cost;
- its input is the level below, read only at that level's boundaries;
- its own attractor memory, and a transition model over its own events.

A higher level's state is then a regime in which the lower level's
transitions are predictable ("unlocked" is the stretch in which a pull opens
the drawer), which is what this project calls a condition. Chains stay
short because each level's events are coarse, so no deep stack is needed.
Biological counterpart: the cortical hierarchy of event time scales, short
events in sensory areas and long ones in posterior medial cortex, with
nested boundaries (Baldassano et al. 2017; Zacks's fine and coarse
boundaries). Caveat: `visual-scene` play data strings behaviours together
almost at random, so a second level may find little there beyond the lock
rules.

### What to drop or change in sections 1–5

- **Per-action affine maps, the 7 × 7 view and the MiniGrid checklist
  (sections 1, 4, 5).** The maps come from the Tolman-Eichenbaum Machine,
  where actions are discrete. Here the arm's pose is given by
  proprioception, actions are continuous, and MiniGrid is retired.
- **The walking term (section 3).** It hides a shortest-path search inside
  the energy and measures closeness in steps, which the user ruled out
  (2026-10-31).
- **The protection term (section 3).** An inner product between two
  vectors does not measure "this step undoes that condition".
- **Descent through a deep chained transition model (section 3).**
  Energy-Based Reasoners needed noise, restarts and supervision along the
  descent to make it reach right answers, with solution labels. CRAM's
  cheap local training holds for pairwise energies over discrete variables.
- **One threshold for "no plan" (section 3)** is reasonable: CRAM fits its
  stopping threshold from training solutions and never stopped on a wrong
  board.

### Link to card 145

Card 145 is section 2's switching penalty in amortised form: an encoder
with switches that cost to open. After 120 minutes (one seed) the state
held the drawer, the window and both locks, each in its own dimensions; the
cube was absent; 38% of switch openings were tiny corrections (the state
moved by 0.03–0.08 there, against 1.6–3.7 at true changes). A prediction,
untested: if the state is inferred by minimising an energy with an L1 cost
on its change, changes worth less than the cost come out exactly zero
(a soft threshold; Bullo 2026 gives the mathematics for predictive coding).
That would not bring in the cube.

### Order of work

1. **Binding first** (the user, 2026-11-19). The literature review is in
   LITERATURE.md. No design is chosen.
2. Deferred, cheap, and independent of binding: the upper half alone on
   the simulator's five variables. A pairwise energy over event records
   (state before, action, state after) learned by CRAM's local rule, scored
   on how many events it needs to learn that a pull opens the drawer only
   when unlocked, against the kept recall on the memory world (cards
   136–136.1), and on whether clamping a goal and settling yields "press
   the button first".
3. A second level over the first level's events, for the amendment above.

Adopting any of this changes CHARTER's "Current direction"; that needs the
user's explicit agreement and has not been made.

### Ways to try the Tolman-Eichenbaum Machine for binding (2026-11-19, ideas only)

The user (2026-11-19): the precise layers are not needed, and modules need
not be tried right away. What the machine binds: a location code, moved by
the action taken, joined to what is sensed there in one conjunctive code,
stored in a fast Hebbian memory and recalled from either half. Slow weights
learn the structure shared by many worlds; the fast memory holds what is
where in this one. Its transformer form (Whittington, Warren and Behrens
2022) is attention over stored pairs of location and content. It senses one
place at a time, so an object's weight is how often it is visited, not how
many pixels it covers.

1. **What is where, with the arm's pose as the location code.** The pose is
   given, so nothing has to learn to integrate movement. Content is the
   patch features around the gripper. The memory answers "what is expected
   at this pose" and "where was this content last". Tests binding of the
   cube to a place from a local reading, against card 143's rule (cube in
   the open drawer 27.7 → 1.70 cm). Risk: the gripper's own appearance
   fills the local patches.
2. **Which object plays which role, on the memory world (cards 136–136.1).**
   Trained over many drawn worlds: slow weights learn that a child needs its
   parents, the fast memory binds which switch is which child's parent from
   a few touches. Scored on cold touches against the kept recall and the
   exact observer. The setting the machine was built for; no pixels.
3. **An object as content that shares one frame.** Store features at grid
   places; an object is the set of stored pairs that one shift re-anchors
   when the scene moves. Nearest to Monty; most speculative (card 124: a
   carried cube merges with the arm; the camera is fixed).

### Ways to try AKOrN for binding (2026-11-19, ideas only)

What it is (Miyato et al. 2025, read today): each unit is a unit vector
that rotates; units pull each other into or out of alignment through
learned couplings, and the data enters as a fixed pull on each unit. With
symmetric couplings each update lowers a Hopfield-like energy. Units that
end aligned are bound. Trained with an image-level contrastive objective,
not reconstruction; objects are read by clustering the final features with
a given number of clusters. CLEVRTex FG-ARI 88.5 (slot attention 62.4, best
slot model 92.9); Pascal VOC 52.0 against DINO's 47.2. Images only; not
run on frozen pretrained features, where earlier synchrony models gained
little. It decides which patches make one object, the half the
Tolman-Eichenbaum Machine leaves out. Binding is pairwise, so an object of
two patches is not outvoted by the background as it is under reconstruction.

1. **Gate, no training of a binder.** Read "same object?" for pairs of our
   stored DINOv2 patches with a probe (Li et al. 2025), scored per object
   against the simulator's segmentation render: cube, buttons, handle. If
   the cube's pairs cannot be read, no pairwise binder on these features
   will find it.
2. **Settle with those couplings.** One oscillator per patch, couplings
   from the probe among nearby patches (the signal is local, Singal 2026),
   Kuramoto steps, cluster. Against card 116.1's slots.
3. **Carry the oscillators through time.** Start each frame from the last
   frame's state so an object keeps its phase. A carried cube locking to
   the gripper and unlocking at release would be binding and an event in
   one signal. Untested anywhere that the search found.

The two fit together: synchrony groups the patches; the fast memory stores
what each group is and where.

### Where to test binding first (2026-11-19, the user's question; not decided)

The user: "maybe ogbench isn't the right place to test this initially?"
Against OGBench `visual-scene` as the first place: it is one scene with
the same five objects every episode, so binding cannot be told from
memorising a layout; the cube is a few pixels across in 64 × 64 frames and
the arm covers it; no factor (size, number, occlusion, being carried) can
be varied alone; a run takes hours (card 145: 120 minutes a seed). For it:
it is where cards are judged, and it has actions, the arm's pose, locks and
a segmentation render.

Options, simplest first:

1. **A published small benchmark with known numbers.** AKOrN reports
   Tetrominoes (32 pixels) and dSprites (64 pixels) with one block and 50
   epochs. Reproducing a number shows our implementation works before a
   failure of ours is read as a failure of the idea. No actions, no time.
2. **Our own small sprite world, as a test bed for one component** (the
   precedent is the memory world, card 136). A hand that carries objects,
   one lock rule, exact masks, new objects every episode, and each factor
   swept alone, object size down to one patch included. All three levels
   have a ground truth: masks, object states, the order of actions. Learned
   from raw pixels; frozen DINOv2 does not suit flat sprites.
3. **OGBench's own renderer with factors swept** (the cube 1, 2 and 4 times
   its size, the arm hidden, a higher resolution): the bridge back. Whole
   cards stay judged on `visual-scene`.

Risk of 1 and 2: a mechanism tuned to a toy (the MiniGrid history; CRAM's
caution that a hand-set energy solves the puzzle benchmarks). Guard: no
world-specific part in the mechanism, and each sweep runs up to OGBench's
values.

---

## 7. Candidate energy: synchrony for grouping, a conjunctive memory for what is where (2026-11-19, a sketch; nothing built)

Asked for by the user after section 6. It replaces sections 1–2 as the
candidate for binding; it says nothing yet about choosing actions. Two
terms are published forms; the terms that join them are Claude's and
untested.

### Variables

- **Given each frame:** patches $i$ with content $c_i \in \mathbb{R}^d$ (an
  encoder's output) and place $r_i \in \mathbb{R}^2$ on the grid.
- **Settled each frame:** one phase per patch, a unit vector
  $x_i \in \mathbb{R}^N$, $\|x_i\| = 1$. Patches with the same phase are
  one object.
- **Kept across frames:** $K$ object phases $u_k$ (unit vectors; an object
  keeps its phase, so it keeps its identity), and a memory
  $M = \{(g_n, f_n)\}$ of stored pairs, a place code and a content.
- **Read from the phases** (not free variables):

$$m_{ik} = \frac{e^{\beta \langle x_i, u_k\rangle}}{\sum_{k'} e^{\beta \langle x_i, u_{k'}\rangle}}, \qquad f_k = \frac{\sum_i m_{ik} c_i}{\sum_i m_{ik}}, \qquad w_k = \frac{\sum_i m_{ik} r_i}{\sum_i m_{ik}}, \qquad g_k = \gamma(w_k)$$

  $m_{ik}$ is how much patch $i$ belongs to object $k$; $f_k$ is the
  object's content, $w_k$ its place, and $\gamma$ a place code at several
  scales (grid-like).

### Energy

$$E = E_{\text{group}} + E_{\text{assign}} + E_{\text{mem}} + E_{\text{change}}$$

$$E_{\text{group}} = -\tfrac{1}{2} \sum_{i \neq j} J_{ij}\, \langle x_i, x_j \rangle, \qquad J_{ij} = J_\theta(c_i, c_j, r_i - r_j), \quad J_{ij} = 0 \text{ beyond a radius}$$

Patches that look like parts of one object attract ($J_{ij} > 0$) and
neighbours that do not repel ($J_{ij} < 0$). This is AKOrN's energy
(its equation 3) with the data moved from a pull on each unit into the
couplings, which is where the pairwise "same object" signal of frozen
features lives (Li et al. 2025). It is a continuous relaxation of
correlation clustering.

$$E_{\text{assign}} = -\frac{1}{\beta} \sum_i \log \sum_k e^{\beta \langle x_i, u_k \rangle}$$

A modern Hopfield energy over phases: each patch falls into one object's
phase. It does inside the energy what AKOrN does afterwards by clustering
with a given number of clusters.

$$E_{\text{mem}} = -\frac{1}{\beta_m} \sum_k \rho_k \log \sum_n e^{\beta_m \left( \langle g_k, g_n \rangle + \langle f_k, f_n \rangle \right)}, \qquad \rho_k = \min\Big(1, \sum_i m_{ik}\Big)$$

The Tolman-Eichenbaum memory in its attention form: each occupied object
is pulled toward a stored pair that matches it in place and in content at
once (the sum inside the exponent is a product of the two matches, so it
is a conjunction). Through $m_{ik}$ it also acts on the patches: a patch
is pulled into the phase of the object whose remembered content and place
it fits. That is how a two-patch cube can be bound when the bottom-up
couplings are weak, once it has been stored.

$$E_{\text{change}} = \lambda \sum_k \big\| s_k^{t} - s_k^{t-1} \big\|_1, \qquad s_k = (f_k,\; o_k), \quad o_k = w_k - w_{\pi(k)}$$

Section 2's cost on change, on the object's state: its content and its
place relative to a parent $\pi(k)$, which is the world or another object.
Minimised by inference, changes worth less than $\lambda$ come out exactly
zero. In the first version every parent is the world. With a parent that
can switch, a carried cube has a constant place in the hand's frame, and
picking up and letting go are the two moments its parent changes.

### Settling, writing, reading

- **Settle:** $\dot{x}_i = \mathrm{Proj}_{x_i}\!\big( -\partial E / \partial x_i \big)$
  on the sphere, with noise, from the last frame's phases.
- **Write:** after settling, an occupied object whose best match in $M$ is
  poor, or whose state $s_k$ changed, stores $(g_k, f_k)$ in one shot,
  replacing the pair it came from. An object with no patches keeps its
  pair: that is belief out of view.
- **Read:** clamp a content and settle the place ("where is the cube"), or
  clamp a place and settle the content ("what is at the hand").

### What is learned slowly

$J_\theta$, the patch encoder if the input is raw pixels, and $\gamma$.
Candidate signals: the Tolman-Eichenbaum objective, predicting held-out
content at a place through the memory, sampled evenly over objects, not
patches, so a small object counts as much as a large one; and a local
contrast between real and corrupted groupings, as CRAM trains without
differentiating through the settling. Which of these trains is open.

### Biological counterparts

Grouping by synchrony in visual cortex (von der Malsburg; Singer and Gray;
disputed). The conjunctive memory: hippocampal CA3 joining a place code
from medial entorhinal cortex to content from lateral entorhinal cortex.
Writing at changes: the hippocampal response at event boundaries.

### Weak points

1. $E_{\text{group}}$ has a trivial minimum, every phase equal, unless
   enough couplings are negative; it is not convex, so noise and restarts
   will be needed (as Equilibrium Reasoners found).
2. $K$ is a ceiling on objects in view, and which phase an object first
   takes is arbitrary.
3. The cube still needs bottom-up couplings that separate it from the
   gripper at least once; the memory helps only afterwards.
4. The mean of a group's features is a crude content.
5. No training signal is shown to work.
6. Nothing here chooses an action.

### On the test ladder (section 6)

Tetrominoes and dSprites test $E_{\text{group}} + E_{\text{assign}}$ alone
against AKOrN's published numbers. The sprite world adds $E_{\text{mem}}$
and $E_{\text{change}}$.

### First check of terms 1 and 2 (2026-11-19, card 146's gate)

On Tetrominoes with perfect couplings (+1 same object, −1 different), the
unit-vector state above does not recover the objects (FG-ARI 68 to 85
against the grid's ceiling of 96.4): a repelling pair of unit vectors is
best satisfied by pointing opposite ways, and only two groups can be
opposite. Writing a patch's state as a belief over the $K$ objects,
$p_i = \mathrm{softmax}(\beta v_i)$, with
$E = -\tfrac12 \sum_{i \ne j} J_{ij} \langle p_i, p_j \rangle + \beta^{-1} \sum_i \langle p_i, \log p_i \rangle$,
reaches the ceiling with $K$ at the count and at twice it. That is CRAM's
energy with patches as cells and object identities as values, so the
grouping level and the reasoning level would share one form. Couplings
between near patches alone leave two objects that do not touch free to
merge; a weak repulsion between far pairs repairs it. Details:
[card 146](../experiments/146-grouping-as-settling-on-a-known-benchmark/card.md).

### Terms 1 and 2 on real features (2026-11-19, cards 146 and 146.1)

On a trained AKOrN's features (Tetrominoes; our three runs score FG-ARI
89.3 against the published 88.6), with couplings a cosine minus a threshold
of 0.8 between all pairs: settling with a ceiling of 8 objects scores 87.8,
against 89.5 for clustering told the number of objects and 88.9 for
clustering with a distance threshold. A ceiling of 16 scores the same;
clustering told 8 scores 82.7. The settled state has lower energy than the
true grouping in 86 to 99% of images, so what error remains is in the
couplings. Two cautions from the first attempt (65.2): the threshold must
sit between same-object and different-object similarity, and the descent
must end cold enough to commit; neither showed with perfect couplings.
Open: the couplings are a fixed function with a threshold placed by
labelled images; learning them is next.

### Where the couplings should come from (the user, 2026-11-19, after cards 146 and 146.1)

The user: "I'm not sure if this coupling mechanism is well aligned with my
goal for the model. I think the learning signal should be by agent
interaction with objects: their effect and event segmentations and the
tracking of their motion." Cards 146 and 146.1 took the couplings from the
similarity of static features, with a threshold placed by labelled images;
that source is set aside. What those cards showed about the form (a
belief over objects per patch, settled with only a ceiling on the number)
does not depend on the source. Not decided: whether pairwise couplings
stay as the place the evidence is stored, with interaction supplying it
(two patches that move or change together under the agent's action
attract; one moving while the other stays repels, and outweighs many
co-movements), or whether each object carries its own motion and a patch
joins the object whose motion predicts its change, with no pairwise term.
Tetrominoes has no agent and cannot test either; the sprite world, the
ladder's second step, can.

**Result ([card 147](../experiments/147-binding-from-interaction-in-a-sprite-world/card.md), 2026-11-19).**
Both forms were built in a sprite world and both pass: pairwise couplings
accumulated from co-motion and read by settling find 97.9% of moved things,
object files that own the points whose motion they have predicted best find
98.4% (no memory: 39.9%). In energy terms the first is the pairwise term of
section 7 with $J_{ij}$ learned as log-odds from motion; the second replaces
it with a term per point and file, $E = \sum_i \sum_k p_{ik} S_{ik}$, where
$S_{ik}$ counts the steps in which file $k$'s displacement failed to predict
point $i$, and settling is a plain minimum. What decided the result in both
was accumulation over time with moving apart outweighing moving together;
the per-step version of either fails. The files are far cheaper and are the
form carried forward; they have no way to close a file opened by a misread
motion, which the pairwise form tolerates. Not yet used: effects other than
motion, event boundaries as such, the action.

**Output tokens ([card 148](../experiments/148-the-action-as-an-output-token/card.md), 2026-11-19).** The
user asked whether a token of the energy can stand for an action or a word,
and chose the couplings form over object files ("more true to the energy
formulation and the CRAM paper"). One step is a set of categorical tokens:
the hand's effect, the action $(a_y, a_x, a_g)$, and per object contact,
held and effect. With $x_i$ the one-hot state of token $i$,
$E = -\sum_{i<j} x_i^\top K_{ij} x_j - \sum_i b_i^\top x_i$, the object–action
couplings shared by every object and summed over the objects of the scene.
$K$ is learned by pseudolikelihood. Clamping everything but the action and
settling gives the action that produces the wanted effect: 95.6% after 100
steps of experience, 100% after 1,000, against 77.5% and 99.3% for whole
records retrieved from a modern Hopfield memory. The couplings factorise
(move and grip are separate facts) and the records do not. A word token
would be the same variable with another codebook; that, sequences, and
effects needing a hidden token are untested.

**Sequences (the user, 2026-11-19; thinking, nothing built).** The user:
"I was thinking the natural design was a recurrence ... whatever output you
choose updates the internal state accordingly allowing for a sequence of
output states." Two forms exist in the literature (LITERATURE, "Sequences
from energy models"). (1) Recurrence: settle one step, commit the output,
update the state, settle again; the energy lives inside a step and the
update between steps is a learned transition that is not a descent
(asymmetric couplings in sequence Hopfield networks, the action moving the
location code in TEM, the appended token in Energy-Based Transformers).
(2) One energy over the whole sequence, every step settled together (CRAM,
EBR, Diffuser), which can revise early steps and needs a fixed length. For
the agent the recurrence closes through the world: state and goal clamped,
the action settled and executed, the new observation re-clamps the state.
What must be carried inside between steps is what the observation does not
give: what is held, and which wanted effect is pending. Card 148's energy
already holds "taking needs contact" as a coupling, so an unmet condition
can be settled as a token and become the next wanted effect, which is the
project's backward chaining over conditions (no imagined rollouts). Open:
reaching contact depends on relative position, a dependence on three tokens.

**Modules, and sequences tried ([card 149](../experiments/149-a-sequence-of-actions-by-recurrence/card.md), 2026-11-19).**
The user asked whether what we have is a modular energy model, with each
patch settling and deeper layers reading the settled states up to an
output token. Partly: card 148's energy is a sum of one term per object
sharing the action token (a product of experts); card 147's binding is a
second energy over points whose settled state is collective, not per
patch; the two are not yet connected, and the tokens between them
(touched, held) were designed by hand, which is the open problem in the
literature too. Feed-forward stacking first, feedback only where a lower
level is shown ambiguous. Card 149 then ran the recurrence: each step the
action is proposed (state and wanted effect clamped), verified (state and
action clamped, effect settled), and a belief of what is held is settled
forward from what was seen. Carrying an object to a place works (96.9%);
holding fails on two of five streams, because the energy proposes a take
without a move, which no record holds, and its forward answer for that is
arbitrary. A pairwise energy trained at the data is defined only near the
data; the agent has to try what it proposes and record it.

**Layers of modules in parallel (the user, 2026-11-20; thinking, nothing built).**
The user, after stopping card 149.1: "maybe we should try out experiments
with modularizing energy into a layered structure, each module performing
parallel computation ... each module can receive updates and internally
update its energy state, and where the modules from deeper layers receive
the energy state of previous layers asynchronously. Meaning the deeper
layers update continuously in time based on the abstractions of
intermediate layers", and asks whether a module should receive another
module's scalar energy or its attractor state. Answer given: the state is
what carries content and is what the literature passes (hierarchical
predictive coding, hierarchical associative memory, equilibrium
propagation; multiple-timescale recurrent networks for the slow upper
layers); the natural message is the code of which attractor the lower
module is in (its belief over stored patterns), which is what cards 147
and 148 already use. The scalar energy is worth passing beside it as how
well the module's input fits what it knows: a confidence for the layer
above, a boundary signal for event segmentation when it jumps, and the
"have I seen this" signal that card 149 lacked (true of a Hopfield energy
over stored patterns, not of couplings trained by pseudolikelihood).
Mechanics: one state tensor per layer, every module of a layer updated in
one batched operation, each layer with its own time constant
(tau dx/dt = -dE/dx); asynchronous updating is a random subset per tick
and is the case for which Hopfield's energy descent is proved. Open
choices: feed-forward only (a pipeline of settling modules, no total
energy) or feedback as well (one total energy, joint settling); and how
the variables between layers are learned, which nothing in the literature
supplies.

**Feed-forward layers, tested (cards [150](../experiments/150-three-feed-forward-layers-of-energy-modules/card.md) and [151](../experiments/151-layers-of-attractor-modules-in-a-world-of-parts/card.md), 2026-11-20).**
On sprite pixels three layers (motion per pixel, binding per point, one
module per object) give each object's motion in 99.5% of moving steps and
do not care how they are clocked; the action layer was not put on top
because nothing below produces "touched" and "held". In a world of
patches, objects and scenes, a module is a memory of stored patterns with
energy $E_k(x)$ for pattern $k$; the message upward is the normalised
vector $m_k = -E_k(x) - \log\sum_j e^{-E_j(x)}$, and a module above has
$E_K = -\sum_{c} m^{(c)}_{K_c}$ over its children $c$, where $K_c$ is the
child pattern that $K$ contains. With this the stack equals a flat memory
of whole scenes at every noise level; passing only $\arg\min_k E_k$ drops
92.0% to 35.3%, and passing only $\min_k E_k$ gives chance. Each layer
learns by storing its input when $\min_k E_k$ is over a threshold, so the
scalar energy's job is "is this new", not content. On a stream one
layer's low-pass memory is enough and several add their lags.

**The object module: where, what, does (card [152](../experiments/152-an-object-module-that-knows-what-an-object-is-and-does/card.md), 2026-11-20).**
A slot becomes a module with three parts. *Where* is the place $s$ of the
object's own frame. *What* is a set of memories in that frame; for the
individual's stored pattern $k$ (a mask $M_k$ and a texture $T_k$) the
energy of the pair $(k, s)$ against the seen points $V$ with colours $x$ is
$E(k,s) = \frac{1}{|V|}\big[\sum_{p \in V \cap M_k^s} \lVert x_p - T^s_{k,p}\rVert^2 + c\,|V \setminus M_k^s| + c\,|M_k^s \cap B|\big]$,
where $B$ is the background that is seen: a pattern pixel hidden by
another object costs nothing. What and where are the joint minimum, which
is the Tolman-Eichenbaum form (a location code bound to what is sensed
there) with a Hopfield memory. *Does* is a belief $w$ over kinds; kind
$\kappa$ is a set of couplings $J^\kappa = J^0 + \Delta^\kappa$ over the
step's tokens with $\Delta^\kappa$ sparse, and an object's evidence for
$\kappa$ is the number of its records that $J^\kappa$ mispredicts. Looks
enter as a pairwise energy between shape code, colour code and kind;
the answer under a belief is $\sum_\kappa w_\kappa\,[\text{answer under } J^\kappa]$,
a mixture, not the answer under $\sum_\kappa w_\kappa J^\kappa$. A single
pairwise energy with a kind token cannot hold "this grip takes one kind
and releases another". Results: recognition at a new place and half
hidden 100%; a never-met object handled from its looks as if its
behaviour were told in nine worlds of ten. Open: the rule that finds the
kinds (mine is a pile of thresholds and failed in one world of five), and
binding without true masks.

**Kinds by one energy (card [153](../experiments/153-kinds-found-by-one-energy-in-two-domains/card.md), 2026-11-20).**
The thresholds of card 152's kind learner are replaced by one energy over
the grouping $z$ (item $i$ has kind $z_i$):
$L(z, J) = \sum_i E_i(z_i) - \log P_{\text{CRP}}(z)$, with
$E_i(\kappa) = -\sum_{r \in \text{records}(i)} \log p_{J^\kappa}(\text{effects}_r \mid \text{rest}_r)$,
the effect tokens of a record taken together (scored one given the
others they explain each other and no kind pays), and
$\log P_{\text{CRP}}(z) = K \log\alpha + \sum_\kappa \log\Gamma(n_\kappa)$,
$\alpha = 1$, the prior of latent-cause models. Every item fits its own
sparse difference $\Delta_i$ from its kind's couplings; $J^{z_i} + \Delta_i$
is a candidate kind, stored if it lowers $L$ counting only other items'
records; kinds are refitted, merged if that lowers $L$, and given up when
left with one member. The belief over kinds for a known item is
$\propto \exp(f_{\text{looks}}(\kappa) - E_i(\kappa))$: looks as prior,
records as likelihood, both in nats, in training as at test. Results:
three kinds in 20 of 20 sprite worlds; the same code finds four word
classes in a stream of words (99.2% of the gap to the best reader of
forms). Two findings beside the rule. Card 152's failure was cured by
teaching the looks couplings with every item (weighted by the entropy of
its records' belief), not by the rule: two kinds that predict the same
are harmless in a mixture. And the rule fails where no single item has
records enough to propose a kind (30 episodes in one world; 100
sentences): the proposal must come from items that share a looks code,
and be judged after both kinds are refitted.

**Classes from appearance and transitions together (card [153.1](../experiments/153.1-dynamics-classes-from-appearance-and-transitions/card.md), 2026-11-20).**
Object $i$ has appearance features $a_i$ (attractor indices of its
appearance memories), transitions $D_i$ and a dynamics class $z_i$. The
posterior $p(z_i \mid a_i, D_i) \propto p(z_i \mid a_i)\,p(D_i \mid z_i)$
is used during learning as at test. A class is proposed from one
object's residual couplings or from the pooled transitions of all
objects sharing an appearance attractor; it is fitted and kept only if
it lowers
$S = -\sum_i \log \sum_k p(z_i = k \mid a_i, D_i^{\text{fit}})\,p(D_i^{\text{held}} \mid k)$,
fitted on one half of the transitions and scored on the other, both
ways. No prior on the number of classes. Results: every sprite run at 30
episodes (card 153: 19 of 20); the word stream at 100 sentences 84.3% of
the gap (38.2), at 200 sentences 98.0% (84.9); nothing lost with more
data. Costs: extra small classes are admitted (up to 8 for 4), a
rule-breaking word with one or two transitions is outweighed by its
appearance, and runs take 4 to 34 times longer.

**Feedback as stored couplings (card [154](../experiments/154-feedback-from-object-modules-to-binding/card.md), 2026-11-20).**
The first downward path. The binding layer's energy is
$E(b) = -\tfrac12 \sum_{pq} J_{pq}\, b_p \cdot b_q$ with $J$ accumulated
from co-motion. An object module stores, for a group that motion bound,
its pattern $m$ (mask and appearance in a 16 by 16 frame) and the mean
couplings $J^{in}_m$, $J^{out}_m$ the group had. At first sight a pattern
recognised at place $s$ adds $J^{in}_m$ between the points it explains
and $J^{out}_m$ from them to the rest, and sets its module's belief for
them; the layer then settles as before. Recognition:
$\text{score}(m, s) = \sum_{p \in m,\ \text{free}} [\,-\log q_{mp}$ if
matched, $\log \tfrac12$ if not$]$, accepted above
$\log(\#\text{patterns} \times \#\text{places})$, greedily, explained
pixels no longer free. Results, sprite pixels, no masks: known things
that have not moved are bound in 98.7 and 99.8% of cases (0 without
feedback; 100 with true-mask patterns); a quarter or more hidden, 96.3
and 99.7%. Open: objects that never move (no pattern), patterns of
kinds instead of individuals, a memory that grows without bound.

**The masks removed (card [155](../experiments/155-object-modules-fed-by-binding/card.md), 2026-11-20).**
Cards 152 to 153.1 rerun with an object defined as a stored pattern of
card 154's memory, its appearance features read from the pattern and its
transitions counted from its first move. An object the binding has found
is handled as with masks (take action right 100%, effects at the bound;
both classes of moving things found in every world). The cost is one
thing: an object exists for the model only after it has moved. A
never-met object is not bound at first sight (take 39.1% against 99.8),
a known object that never moved in training is invisible (all known
objects 94.5 against 98.8), and no class is learned for fixed objects.
The user: couplings as a function of appearance need a training
incentive from the agent's interaction, not a designed loss or prior.

**Appearance couplings taught by the agent's own action (card [156](../experiments/156-appearance-couplings-taught-by-the-agents-own-action/card.md), 2026-11-20).**
A network reads the frame around two points and their offset and gives
three states for the pair, together, apart, no evidence, taught by the
binding layer's own $J$ at the end of each episode ($J \ge 3a$,
$J \le -3b$, otherwise). At first sight
$A_{pq} = J^{in} P(\text{together}) + J^{out} P(\text{apart})$ is added
to the binding energy, the states' true frequencies put back and the
pairs whose likeliest state is "no evidence" centred. Taking each pair's
likeliest state instead leaves an object unbound when it looks like
fixed ones; the plain expectation fragments the ground, because a 0.2%
chance of "apart" times $J^{out}$, summed over all pairs, outweighs the
layer's default pull of 0.05. Results: never-met objects bound before
they move in 96 to 97% of cases; the take action right for 93.9% with a
script's experience (39.1 without the couplings). An agent that goes to
the groups its own binding sees teaches the couplings as well, but its
takes always include a move and the test asks for one without (75.0).

**Transitions from the binding layer's own states (card [157](../experiments/157-transitions-read-from-the-binding-layers-own-states/card.md), 2026-11-21).**
An object's transition at a step is read from its points: contact (the
box around them beside the hand's box), the displacement most of them
made, and whether that is the hand's own. No simulator token and no mask
in learning; "held" is dropped because no frame shows it, and the test
asks for the action that brings the object along. On the same experience
of the agent, 86.4% right for never-met objects against 85.0 with the
simulator's tokens (100.0 with the true classes given); class purity
88.9 against 97.8. The ground joins the fixed class, and patterns on the
hand form a class of what always comes along.
