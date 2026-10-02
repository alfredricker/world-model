# Status

Overwritten each session. At most 40 lines.

- **Date:** 2026-10-01. **Rung:** before rung 1. **Architecture:** version 8
  (cards 038–047, kept with card 047): planning on the encoder's vectors,
  recall in two levels (mixed), the state as tokens, walking as a learned
  approach plus its conditions, and situations from both levels. Seeds
  400–404: all four familiar worlds 100%, card 029's steps; unseen rooms
  100% at 1.01–1.11 times the shortest route.
- **Direction:** CHARTER.md's "Current direction", with rule 8 and one
  latent space. User, 2026-09-30: work backward over conditions, not
  imagined action sequences (GOAL.md P21). User, 2026-10-01: codes for
  identity, vectors for similarity; fixes must be general principles;
  tokenize everything (tiles, the held thing), optimize recall later.
- **Next decision:** the next card, with the user. Discussed 2026-10-01
  toward Crafter: partial view and a map first (chained rooms), then an
  inventory with counts and goals as conditions on the state. Open
  within MiniGrid: new colours fail where the encoder puts them far from
  known doors (similarity 0.005–0.021; transfer, paused by the user);
  chance outcomes (an action that works sometimes reads as never).
- **Recent cards:**
  - [047](experiments/047-situations-from-both-levels/card.md): pass,
    keep (version 8);
  - [046](experiments/046-openable-by-identity/card.md): pass, revise
    (the user): one failed try made a new-colour door never openable;
  - [045](experiments/045-movement-through-conditions/card.md) (walking
    through conditions, no imagined step): pass, revise (the user): its
    filter worked around recall's similarity-only "openable";
  - [044](experiments/044-state-as-tokens/card.md): pass, keep;
  - [043](experiments/043-consistent-hypotheticals/card.md): pass, keep.
- **Open in version 8:** the same-thing level reads all four codebooks;
  conjunctions are still checked in spliced situations; no relations;
  routes for a small fixed world; 0.42–0.72 s per layout (version 5:
  0.015–0.024).
- **Demo:** `runs/045_demo_unseen.mp4` (`tools/card045/demo.py`). **papi:**
  GLM-5.3 via OpenRouter; needs credit to regenerate summaries.
- **Pinned:** card 029's arm 3 in the both world; demonstrations once
  random play is too thin; GOAL.md hypotheses draft.
