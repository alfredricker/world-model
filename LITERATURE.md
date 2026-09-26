# Literature

Full text and notes live in paperpipe; use the `papi` name to look them up.

## Current focus

The papers agents should work from until the user changes or requests a change to this section. Read these (via `papi`) before proposing designs for the listed cards.

- **Cards / rung:** 001 (architecture selection, before rung 1).
- **Why these:** each is the source of a mechanism one of the five
  candidates uses; the card's appendix says which.
- **Until:** card 001 has a decision.

| Paper (papi name) | What to take from it |
| ----------------- | -------------------- |
| `dreamerv3` | Candidate A: pixel reconstruction, not reward, carries the representation; its recurrent state-space model is the reference for rung 2 |
| `1911_12247` (C-SWM) | Candidate B: contrastive latent transition objective in grid worlds; not its object slots |
| `1807_03748` (CPC) | Candidate B: the InfoNCE loss |
| `leworldmodel` | Hub H and D: end-to-end latent prediction with SIGReg, no EMA; AdaLN action input; weak on low-diversity data |
| `latent-actions` | Candidate D: per-location discrete change codes with a decoder that sees the current state |
| `schema-networks-zero-shot-transfer-with-a-generative-causal` | Candidate D: persistence by default, sparse causes of change; not its supplied entities |
| `1711_00937` (VQ-VAE) | Candidate D: discrete codebook for event codes |
| `2402_15567` (HILP) | All candidates: steps-to-goal learned action-free by expectile regression on hindsight goals, reward-free |
| `2304_01203` (QRL) | All candidates: quasimetric distances for one-way reachability |
| `2208_08133` (MRN) | All candidates: metric-residual quasimetric head for d |
| `universal-value-function-approximators` (UVFA) | Candidate E: one value function over states and goals |
| `hiql` | Candidate E: action-free goal-conditioned value from offline data; its representation of goals |

## High Level Papers

These are papers to guide the architecture to adhere to the goal. It is useful to read these when thinking about candidate components or direction changes.

| Paper (papi name) | Component or test it shaped | What we borrowed | What we changed or did not take |
| ----------------- | --------------------------- | ---------------- | ------------------------------- |
| `dream-rsi` | Not yet; long-term self-improvement | Discovery history reused as a replay simulator to improve exploration cheaply | An LLM coding-agent setting; relevance to a pixel world model is still to be worked out |
| `temporal-distance-jepa` | The reachability head in card 001; later goal and planning rungs | Directed temporal cost as a planning signal; rollout-consistency loss (Push-T 85.3 → 60.0 without it, 3 seeds) | Needs expert demonstrations for its step-count labels; small gains over LeWM on most tasks |
| Saulus, *Unsupervised Causal Abstraction Discovery* | Not yet | | Not in papi; needs an arXiv ID or PDF |
| Gentner, *Structure-Mapping* (1983) | Not yet; relation transfer (rung 3) | | Not in papi; needs the PDF |
