#!/usr/bin/env bash
# card 060's declared revision: the door to open and look past is kept between steps
cd "$(dirname "$0")/../.."
W=runs/054/b_m0.5_399.pt; P="bin/prun python tools/card057/partial.py $W --arm B --occlude --frontier --keep-look"
$P --out runs/060/fam_Bk_399.json --layouts 30 > runs/060/fam_Bk_399.out 2>&1 &
$P --out runs/060/chain_Bk_399.json @@ --chain one_door --seed 399 --n 100 > runs/060/chain_Bk_399.out 2>&1 &
wait
echo done
