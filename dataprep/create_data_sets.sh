#!/usr/bin/env bash

PROJECT=$(gcloud config get-value project)

# FORMAT=NEWLINE_DELIMITED_JSON
FORMAT=CSV

echo Extracting training set ...
bq extract \
    --destination_format=${FORMAT} \
    --noprint_header \
    tmp.jokes_train \
    gs://${PROJECT}/lachebeque/data/train.data
echo "gs://${PROJECT}/lachebeque/data/train.data"
gsutil cat gs://${PROJECT}/lachebeque/data/train.data | head -n 3

echo

echo Extracting eval set ...
bq extract \
    --destination_format=${FORMAT} \
    --noprint_header \
    tmp.jokes_eval \
    gs://${PROJECT}/lachebeque/data/eval.data
echo "gs://${PROJECT}/lachebeque/data/eval.data"
gsutil cat gs://${PROJECT}/lachebeque/data/eval.data | head -n 3