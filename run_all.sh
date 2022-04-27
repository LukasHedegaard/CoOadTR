#!/usr/bin/env bash

mkdir -p runs

for DATASET in "anet" "kin"
do
    DIM_FEATURE=$([ "$DATASET" = "anet" ] && echo "3072" || echo "4096")

    for LAYERS in 1 2 3
    do
        for CLS_POS in -1 0 1 2 3 4
        do
            if (( $(($CLS_POS + 1)) < $(($LAYERS * 2)) ))
            then
                for SEED in 1 2 3 4 5
                do
                
                    echo python main.py \
                        --feature $DATASET \
                        --dim_feature $DIM_FEATURE \
                        --num_layers $LAYERS \
                        --enc_layers 64 \
                        --seed $SEED \
                        --cls_token_layer_idx $CLS_POS \
                    &> "runs/oadtr-b${LAYERS}_clspos${CLS_POS}_seed${SEED}_${DATASET}.txt"

                done
            fi
        done
    done
done
