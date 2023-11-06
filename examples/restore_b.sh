#!/usr/bin/env bash
# Copyright 2023  Bofeng Huang

# Normalize text in NeMo's manifest files.
# Run analysis on each sub dataset.

export TRANSFORMERS_CACHE="/projects/bhuang/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/projects/bhuang/.cache/huggingface/datasets"
export DATASETS_VERBOSITY="error"

export CUDA_VISIBLE_DEVICES="0"

# model_name=bofenghuang/whisper-medium-cv11-french
model_name=bofenghuang/whisper-large-v2-cv11-french
# model_name=openai/whisper-large-v2

# input_file_path=/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/train/train_facebook_multilingual_librispeech_manifest_normalized.json
# input_file_path=/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/validation/validation_facebook_multilingual_librispeech_manifest_normalized.json
# input_file_path=/projects/bhuang/corpus/speech/nemo_manifests/facebook/multilingual_librispeech/french/test/test_facebook_multilingual_librispeech_manifest_normalized.json

# input_file_path=/projects/bhuang/corpus/speech/nemo_manifests/media_speech/FR/media_speech_manifest_normalized.json

# input_file_path=/projects/bhuang/corpus/speech/nemo_manifests/gigant/african_accented_french/train/train_gigant_african_accented_french_manifest_normalized.json
# input_file_path=/projects/bhuang/corpus/speech/nemo_manifests/gigant/african_accented_french/test/test_gigant_african_accented_french_manifest_normalized.json

# input_file_path=/projects/bhuang/corpus/speech/nemo_manifests/att_hack/att_hack_manifest_normalized_min1_dedup256.json

# input_file_path=/projects/bhuang/corpus/speech/nemo_manifests/PolyAI/minds14/fr-FR/train/train_PolyAI_minds14_manifest_normalized.json

input_file_path=/projects/bhuang/corpus/speech/nemo_manifests/lingualibre/FR/lingualibre_manifest_normalized_min05_dedup4.json


tmp_path="${input_file_path%.*}"
output_file_path=${tmp_path}_pnc.json
processed_file_path=${tmp_path}_pnc_cleaned.json

# python examples/restore_b.py \
#     --model_name_or_path $model_name \
#     --input_file_path $input_file_path \
#     --output_file_path $output_file_path

python examples/postprocess_restore_b.py $output_file_path $processed_file_path

# split into 5
# nchunks=5
# a=(`wc -l $input_file_path`)
# lines=`echo $(($a/$nchunks)) | bc -l`
# split --numeric-suffixes=1 --additional-suffix=.json -l $lines $input_file_path tmp_data/train_facebook_multilingual_librispeech_manifest_normalized_
