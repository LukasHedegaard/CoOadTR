#!/usr/bin/env bash

mkdir -p runs

for FEATURES in "anet" "kin"
do
    DIM_FEATURE=$([ "$FEATURES" = "anet" ] && echo "3072" || echo "4096")

    for LAYERS in 3
    do
        
        for SEED in 1 2 3 4 5
        do
        
            python main.py \
                --feature $FEATURES \
                --dim_feature $DIM_FEATURE \
                --num_layers $LAYERS \
                --decoder_layers 5
                --enc_layers 64 \
                --seed $SEED \
            &> "runs/oadtr-b${LAYERS}_${DATASET}_seed${SEED}_${FEATURES}.txt"

        done
    done
done

