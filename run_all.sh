#!/usr/bin/env bash

mkdir -p runs

for FEATURES in "kin" #"anet" 
do
    DIM_FEATURE=$([ "$FEATURES" = "anet" ] && echo "4096" || echo "4096")

    for LAYERS in 3
    do
        
        for SEED in 1 2 3 4 5
        do
        
            python main.py \
                --feature "tvseries_${FEATURES}_features.pickle" \
                --dim_feature $DIM_FEATURE \
                --num_layers $LAYERS \
                --decoder_layers 5 \
                --enc_layers 64 \
                --dataset tvseries \
                --seed $SEED \
            &> "runs/oadtr_b${LAYERS}_seed${SEED}_${FEATURES}.txt"

        done
    done
done
