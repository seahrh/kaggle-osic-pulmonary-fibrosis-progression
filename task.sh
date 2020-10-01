#!/usr/bin/env bash
set -x #echo on

DATE=$(date '+%Y%m%d_%H%M%S')
REGION="asia-southeast1"
PROJECT_NAME="kagglemuggle"
IMAGE_URI="asia.gcr.io/${PROJECT_NAME}/osic-pulmonary-fibrosis:1.0"
BUCKET_NAME="osic-pulmonary-fibrosis-asia-southeast1"
JOB_NAME="${MODEL_NAME}_${DATE}"
JOB_DIR="gs://${BUCKET_NAME}/models/${JOB_NAME}"
DATA_DIR="data"

gcloud ai-platform jobs submit training "${JOB_NAME}" \
        --master-image-uri "${IMAGE_URI}" \
        --job-dir "${JOB_DIR}" \
        --region "${REGION}" \
        --scale-tier custom \
        --master-machine-type n1-standard-4 \
        --master-accelerator count=1,type=NVIDIA_TESLA_T4 \
        -- \
        --data_dir="${DATA_DIR}" \
        --epochs="${EPOCHS}"
