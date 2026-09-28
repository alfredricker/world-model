# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-09-28. **Rung:** before rung 1. **Architecture:** version 3,
  experimental shared map/context and shared spatial feature computation.
- **Direction:** retain condition recursion. Conditions, readiness and walking
  use one learned state; no separate frozen condition model at inference.
- **Latest card 020 (done, pass, keep):** conditions read the same recurrent
  spatial features as readiness. Both score 99.61% on 512 evaluator-labelled
  test frames from 128 new layouts, against the 99% bar. Flat condition
  readout: 98.05%; its readiness: 99.80%. One seed; controlled component only.
- **Card 017 (revise):** exact audit found 31 door-readiness input groups
  with opposite true labels, involving 82 of 302 positives in 200,500 queried
  poses. Local-only readiness is insufficient; a wider head fits training
  but scores 17.97% on changed geometry. Full-tree coverage remains incomplete.
- **Card 018 (revise):** changing only the remote goal in training pairs
  lifts wide-head test readiness to 96.68% versus 50% local-only. All 17
  errors occur at a goal offset absent from positive training examples.
- **Card 019 (revise):** shared local updates lift readiness to 99.80%;
  the flat condition head stays at 97.85%, motivating card 020.
- **Next decision:** a full-tree component gate using architecture 3, with
  adequate coverage and learned-label comparison. The original small sample
  has only 9 goal-ready and 6 key-ready test poses at the agent versus 20
  required, and some deeper ways have none. Do not confuse this with the
  balanced evaluator pairs of cards 018–020; no walking run is justified yet.
- **Implementation:** `src/worldmodel/spatial_state.py`. Full-tree screen:
  `tools/card017/bench_shared.py`; controlled comparisons: `bench_context.py`.
  Readiness/condition heads share both patch encoding and spatial processing.
- **Checks:** all 56 repository tests pass, including the exact counterexample,
  orientation handling and both heads' gradients through the shared processor.
- **Retained evidence:** card 012's learned key-world conditions give 95%
  acting with exact walking, 30.2% all learned, 0.4% random. Card 016's
  learned egocentric walking gives 48.4% at both 30k and 90k updates.
- **Scope:** cards 018–020 use evaluator labels and constructed goal pairs;
  they do not establish full-tree retention, discovery or autonomous walking.
  No capability-ladder rung has passed. All new runs were under one minute.
- **Pinned:** duplicate detector merging needs literature; demonstrations
  once random play is too thin. LESSONS.md still needs a separate merge pass.
