#!/usr/bin/env bash
# card 053 step 2a baseline: version 10's pixel recipe (rebuild, codebook, pair and effect terms) on the same stream, weights saved
cd "$(dirname "$0")/../.."
bin/prun python tools/card052/predflip.py --seed 399 --checkpoints 40 --updates 2000 --mu 0.1 --restart 100000000 \
    --starts 0.5 --strat 1 --colour 1 --tint 0 --out runs/053/s2_pixel_399.json > runs/053/s2_pixel_399.out 2>&1
echo done
