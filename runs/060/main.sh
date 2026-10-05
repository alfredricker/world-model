#!/usr/bin/env bash
# card 060: occlusion; arm A (version 11's exploration) and arm B (frontier, and opening a door to see beyond)
cd "$(dirname "$0")/../.."
W=runs/054/b_m0.5_399.pt; P="bin/prun python tools/card057/partial.py $W --arm B --occlude"
$P --out runs/060/fam_A_399.json --layouts 30 > runs/060/fam_A_399.out 2>&1 &
$P --frontier --out runs/060/fam_B_399.json --layouts 30 > runs/060/fam_B_399.out 2>&1 &
$P --out runs/060/chain_A_399.json @@ --chain one_door --seed 399 --n 100 > runs/060/chain_A_399.out 2>&1 &
$P --frontier --out runs/060/chain_B_399.json @@ --chain one_door --seed 399 --n 100 > runs/060/chain_B_399.out 2>&1 &
wait
echo done
