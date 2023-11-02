import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_CACHE"] = "/projects/bhuang/.cache/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/projects/bhuang/.cache/huggingface/datasets"


from restore_b import main

model_name = "bofenghuang/whisper-large-v2-cv11-french"
# model_name = "openai/whisper-large-v2"

# input_file_path = "/projects/bhuang/corpus/speech/nemo_manifests/facebook/voxpopuli/fr/train/train_facebook_voxpopuli_manifest.json"
input_file_path = "tmp.json"


output_file_path = "tmp_new.json"

main(
    model_name_or_path=model_name,
    input_file_path=input_file_path,
    output_file_path=output_file_path,
)
