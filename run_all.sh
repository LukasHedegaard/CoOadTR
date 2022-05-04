#!/usr/bin/env bash

mkdir -p runs

for FEATURES in "kin" #"anet" "kin"
do
    DIM_FEATURE=$([ "$FEATURES" = "anet" ] && echo "3072" || echo "4096")

    for LAYERS in 1 2
    do
        for POS_ENC in "recycling_fixed" #"recycling_fixed" "recycling_learned" 
        do
            for NUM_EMB in 127 #64 127
            do
                for SEED in 1 2 3 4 5
                do
                
                    python main.py \
                        --feature "${FEATURES}" \
                        --dim_feature $DIM_FEATURE \
                        --num_layers $LAYERS \
                        --positional_encoding_type $POS_ENC \
                        --num_embeddings $NUM_EMB \
                        --enc_layers 64 \
                        --seed $SEED \
                    &> "runs_thumos/oadtr_b${LAYERS}_${POS_ENC}_${NUM_EMB}_seed${SEED}_${FEATURES}.txt"

                done
            done
        done
    done
done

for FEATURES in "anet" "kin"
do
    DIM_FEATURE=$([ "$FEATURES" = "anet" ] && echo "4096" || echo "4096")

    for LAYERS in 1 2
    do
        for POS_ENC in "recycling_fixed"
        do
            for NUM_EMB in 127 #64 127
            do
                for SEED in 1 2 3 4 5
                do
                
                    python main.py \
                        --feature "tvseries_${FEATURES}_features.pickle" \
                        --dim_feature $DIM_FEATURE \
                        --num_layers $LAYERS \
                        --positional_encoding_type $POS_ENC \
                        --num_embeddings $NUM_EMB \
                        --enc_layers 64 \
                        --seed $SEED \
                        --dataset tvseries \
                    &> "runs_tvseries/oadtr_b${LAYERS}_${POS_ENC}_${NUM_EMB}_seed${SEED}_${FEATURES}.txt"

                done
            done
        done
    done
done

for FEATURES in "anet" "kin"
do
    DIM_FEATURE=$([ "$FEATURES" = "anet" ] && echo "7168" || echo "8192")

    for LAYERS in 1 2
    do
        for POS_ENC in "recycling_fixed"
        do
            for NUM_EMB in 127 #64 127
            do
                for SEED in 1 2 3 4 5
                do
                
                    python main.py \
                        --feature "${FEATURES}_audio" \
                        --dim_feature $DIM_FEATURE \
                        --num_layers $LAYERS \
                        --positional_encoding_type $POS_ENC \
                        --num_embeddings $NUM_EMB \
                        --enc_layers 64 \
                        --seed $SEED \
                        --dataset thumos \
                    &> "runs_thumos_audio/oadtr_b${LAYERS}_${POS_ENC}_${NUM_EMB}_seed${SEED}_${FEATURES}.txt"

                done
            done
        done
    done
done