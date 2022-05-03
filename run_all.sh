#!/usr/bin/env bash

mkdir -p runs

for FEATURES in "kin" #"anet" #"kin"
do
    DIM_FEATURE=$([ "$FEATURES" = "anet" ] && echo "4096" || echo "4096")

    for LAYERS in 1 2
    do
        if (( $(($CLS_POS + 1)) < $(($LAYERS * 2)) ))
        then
            for SEED in 1 2 3 4 5
            do
            
                python main.py \
                    --feature "tvseries_${FEATURES}_features.pickle" \
                    --dim_feature $DIM_FEATURE \
                    --num_layers $LAYERS \
                    --enc_layers 64 \
                    --seed $SEED \
                    --dataset tvseries \
                &> "runs/oadtr_b${LAYERS}_seed${SEED}_${FEATURES}.txt"

            done
        fi
    done
done
