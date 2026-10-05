# Overnight 2026-10-05

Temporary record (AGENTS.md, "Overnight sessions"). Objective: an encoder
that is well principled and scalable, keeps the goal/condition
hierarchy, and with which the agent still discovers conditions. Then
goals from example frames, then a view smaller than the map.

## Card 054: identity up to noise, and perception that keeps what is visible

Approved before the user slept. Codes become "the same appearance up to
noise" (no learned codebook); the encoder keeps every visible
difference apart by a margin.

- **Step A (codes only, on card 053's encoder):** the planner still
  reaches 100% in the familiar worlds and still discovers the right
  conditions (toggle reads the key, or the switch), with no codebook.
  Recall: matching key 45% → 100%, but other key 85% → 60%: card 053's
  encoder leaves no clean gap between noise and real differences (boxes
  of every colour within noise; doors 7× noisier than keys), so noisy
  copies split. As declared, on to step B.
- **Step B (the encoder keeps visibly different tiles at least m apart;
  three arms):** m = 0.5 is strongest: recall's colour cases 100% /
  100% (codebook: 45% / 85%), noise changes 1.8% of predictions (bar
  1%), the planner 100% in the familiar worlds, chained rooms 100%,
  cluttered 100% (version 10: 100%, 98%, 98–100%), and the right
  conditions admitted. m = 1.0 and m = 1.0 with noisy copies pulled
  together were weaker on recall (other key 52–58%). Picking up while
  holding fails under every encoder (rare in random play; a limit of
  recall's data). Next: seeds 400–402.
