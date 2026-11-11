# Status

Overwritten each session. At most 40 lines.
- **Date:** 2026-11-11, `master`, uncommitted: cards 119–127 written (121, 123, 117.1, 126 gate only), LESSONS.
  **Rung:** 3 (skills and plans), open. **Architecture:** version 27 (MiniGrid).
  **World:** OGBench `visual-scene`; cards 120 and 122 use the state dataset's episodes, rendered.
- **Direction:** published learned methods below ([card 113](experiments/113-post-mortem-learned-stack/card.md));
  conditions are event segmentation: stable stretches between boundaries (the user,
  2026-11-01). Planner bar: GAS 54.4% on `visual-scene-play`. Portability tests paused.
- **Rung 1 passed** ([114](experiments/114-lock-effects-from-pixels/card.md),
  [114.1](experiments/114.1-counterfactual-lock/card.md)): a predictor on frozen DINOv2 patches
  follows the lock (96.9–100%; actions alone 0–5%). 115–116.2 (surprise, slots) failed.
- **[119](experiments/119-variables-from-lasting-change/card.md): fail, revise.** Lasting change finds
  boundaries on DINOv2 block 2 (F1 0.63; block 12, 0.30); units fail (drawer 0.17).
- **[120](experiments/120-touch-brackets-events/card.md): pass, keep.** OGBench's `gripper_contact`
  (state dataset only): 82% of touches hold exactly one transition. Before against after a
  touch: cube top patch exact 36% (119: 13%); a lock flip shows on the handle, not the button.
- **[122](experiments/122-sharper-frames/card.md): fail, revise.** 256-px renders, 32 × 32 patches,
  strict measures. Cube exact top patch 61% (64 px: 36%), drawer 80% (27%). Button found
  27% (bar 60%): the handle outranks it; handle aside, 95%. Small extents still flood.
- **[123](experiments/123-regions-of-change/card.md), [121](experiments/121-extent-from-a-seed/card.md) gate only:** regions of change give the button; the cube merges into the arm.
- **[124](experiments/124-tokens-from-consequences/card.md): fail, revise** (30 min per arm, one seed).
  Slots at 256 px. Present only: finds the cube (0.55), mean 0.28 (bar 0.40). Next frame given
  action and touch: loses the cube, nearly follows the drawer's lock (0.83, bar 0.9). A cost
  on token change merges the cube into the arm. No button token in any arm.
- **[125](experiments/125-conditions-by-contrast-over-touches/card.md): fail (2 of 3), revise.** The
  event as the unit (the user): touches at one place, split by changed or not; the patches whose
  before-features differ are the condition. Finds each lock's button; predicts as the true lock.
- **Setup:** ogbench, mujoco, dm_control, scikit-learn by pip, not in `pyproject.toml`.
  Rendering: `MUJOCO_GL=egl`, a 64-bit libglvnd on `LD_LIBRARY_PATH`,
  `__EGL_VENDOR_LIBRARY_DIRS=/run/opengl-driver/share/glvnd/egl_vendor.d`. Data `~/.ogbench/data`. Cards 001–112: `experiments/archive/`.
- **[117.1](experiments/117.1-self-from-proprioception/card.md) gate not passed** (the self from pose).
- **[125.1](experiments/125.1-conditions-read-where-steady/card.md): fail (2 of 3), revise.** A patch counts only where steady: the arm drops out, a lock's state is read exactly.
- **[125.2](experiments/125.2-chain-by-change-of-state/card.md): pass, keep.** The chain by change of
  state between steady readings: all found, none false; the place to touch is the true button's.
- **[126](experiments/126-dense-layer-from-touch-consequences/card.md) gate only:** no room for a
  trained layer in this scene; kept for a harder one (the user).
- **[127](experiments/127-plans-from-remembered-touches/card.md): pass, keep.** Plans backward over
  conditions, each step a remembered touch redone, no training: task 1 92%, task 2 94% (bars 80, 60);
  no condition check 0%. No cube: five-task mean 37% < 54.4%. **For the user:** cube next? adopt as v28?
