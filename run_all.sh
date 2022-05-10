#!/usr/bin/env bash

mkdir -p runs_thumos

for FEATURES in "anet" "kin"
do
    DIM_FEATURE=$([ "$FEATURES" = "anet" ] && echo "3072" || echo "4096")

    for LAYERS in 1 2
    do
        for SEED in 1 2 3 4 5
        do
        
            python main.py \
                --feature $FEATURES \
                --dim_feature $DIM_FEATURE \
                --num_layers $LAYERS \
                --decoder_layers 5 \
                --enc_layers 64 \
                --seed $SEED \
            &> "runs_thumos/oadtr_b${LAYERS}_seed${SEED}_${FEATURES}.txt"

        done
    done
done


mkdir -p runs_thumos_audio

for FEATURES in "anet" "kin"
do
    DIM_FEATURE=$([ "$FEATURES" = "anet" ] && echo "7168" || echo "8192")

    for LAYERS in 1 2
    do
        for SEED in 1 2 3 4 5
        do
        
            python main.py \
                --feature "${FEATURES}_audio" \
                --dim_feature $DIM_FEATURE \
                --num_layers $LAYERS \
                --enc_layers 64 \
                --seed $SEED \
            &> "runs_thumos_audio/oadtr_b${LAYERS}_seed${SEED}_${FEATURES}.txt"

        done
    done
done