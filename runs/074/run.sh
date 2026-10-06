#!/bin/sh
# Card 074: version 17 with conflicts read from plans (WM_CONFLICTS=1 in place of card 073's WM_ORDER=1).
# 1) weighings on training layouts (seeds from 500,000; the tests use 1,000,000): the memory of orders;
# 2) the criteria on card 073's test episodes.
cd "$(dirname "$0")/../.."
export WM_REL_ENCODER=runs/070/encoder.pt WM_TRYING=1 WM_CONFLICTS=1
WM_SEED_TEST=500000 timeout 900 bin/prun python tools/card069/run.py tiers --tier 1 --online 0 --n 60 --memory 066 --out runs/074/train_tier1.json > runs/074/train_tier1.log 2>&1
WM_SEED_TEST=500000 timeout 1200 bin/prun python tools/card069/run.py tiers --tier 2 --online 0 --n 40 --memory 066 --out runs/074/train_tier2.json > runs/074/train_tier2.log 2>&1
WM_SEED_TEST=500000 timeout 900 bin/prun python tools/card073/known.py purple runs/074/train_decoy.json > runs/074/train_decoy.log 2>&1
for hue in red green blue purple yellow grey; do
  timeout 900 bin/prun python tools/card073/known.py $hue runs/074/known_$hue.json > runs/074/known_$hue.log 2>&1
  timeout 900 bin/prun python tools/card069/run.py decoy --fold $hue --hold opening --online 0 --n 100 \
    --memory 066 --out runs/074/decoy_$hue.json > runs/074/decoy_$hue.log 2>&1
done
timeout 900 bin/prun python tools/card069/run.py tiers --tier 1 --online 0 --n 200 --memory 066 --out runs/074/tier1.json > runs/074/tier1.log 2>&1
timeout 1500 bin/prun python tools/card069/run.py tiers --tier 2 --online 0 --n 100 --memory 066 --out runs/074/tier2.json > runs/074/tier2.log 2>&1
