#!/bin/bash
# recommend hyperparameters grid from original paper: 
# Adam lr: {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}, Adam wd: {1e-5, 1e-4, 1e-3, 1e-2}
# Muon lr: {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2}, Muon wd: {1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3}
set -e

# export available ascend devices
DEVICE="${DEVICE:-0}"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"

# last name for run output directory
RUN_DIR="${RUN_DIR:-RUN0}"

echo "BEFORE: OPTIMIZER='$OPTIMIZER'"
# [Options] "adam" or "muon"
OPTIMIZER="${OPTIMIZER:-adam}"
echo "AFTER: OPTIMIZER='$OPTIMIZER'"

# [Options] "Industrial_and_Scientific" or "Office_Products"
CATEGORY="${CATEGORY:-Industrial_and_Scientific}"

# Training hyperparameters (small T5 -> large batch fits on one GPU).
BATCH_SIZE="${BATCH_SIZE:-256}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
SEED="${SEED:-42}"
SAMPLE="${SAMPLE:--1}"

DATA_DIR="minionerec_engine/data/Amazon"
OUTPUT_DIR="outputs/generative/sft_tiger_${OPTIMIZER}/${CATEGORY}/${RUN_DIR}"

train_file=$(ls -f ${DATA_DIR}/train/${CATEGORY}*11.csv | head -n 1)
eval_file=$(ls -f ${DATA_DIR}/valid/${CATEGORY}*11.csv | head -n 1)
index_path="${DATA_DIR}/index/${CATEGORY}.index.json"

echo "=========================================="
echo "        MuonRec TIGER SFT Pipeline        "
echo "=========================================="
echo "Category   : ${CATEGORY}"
echo "Optimizer  : ${OPTIMIZER}"
echo "Train Data : ${train_file}"
echo "Index      : ${index_path}"
echo "Output Dir : ${OUTPUT_DIR}"
echo "=========================================="

COMMON_ARGS="--train_file ${train_file} \
    --eval_file ${eval_file} \
    --index_path ${index_path} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --sample ${SAMPLE} \
    --seed ${SEED}"

if [ "$OPTIMIZER" = "muon" ]; then
    python experiments/generative/sft_tiger_muon.py ${COMMON_ARGS} \
        --muon_lr "${MUON_LR:-1e-4}" \
        --adam_lr "${ADAM_LR:-1e-4}" \
        --muon_wd "${MUON_WD:-1e-5}" \
        --adam_wd "${ADAM_WD:-0}"
elif [ "$OPTIMIZER" = "adam" ]; then
    python experiments/generative/sft_tiger_adam.py ${COMMON_ARGS} \
        --learning_rate "${ADAM_LR:-1e-4}" \
        --weight_decay "${ADAM_WD:-0}" \
        --use_adam "${USE_ADAM:-True}"
else
    echo "Error: Unknown OPTIMIZER '${OPTIMIZER}'. Choose 'adam' or 'muon'."
    exit 1
fi
