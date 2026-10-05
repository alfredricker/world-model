#!/usr/bin/env bash
# card 065: card 056's goals from frames, through version 14's small view (seed 399)
cd "$(dirname "$0")/../.."
bin/prun python tools/card065/goals_small.py runs/054/b_m0.5_399.pt --occlude --frontier --keep-look --out runs/065/main_399.json --layouts 30 > runs/065/main_399.out 2>&1
echo done
