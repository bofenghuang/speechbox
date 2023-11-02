#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang


import json
import re
import string
import os
import time

import fire
import torch
import tqdm
from datasets import Audio, load_dataset

# from datasets.utils.logging import disable_progress_bar

from speechbox import PunctuationRestorer

SAMPLING_RATE = 16000

# disable_progress_bar()


def main(
    model_name_or_path: str,
    input_file_path: str,
    output_file_path: str,
    language: str = "fr",
    audio_column_name: str = "audio_filepath",
    text_column_name: str = "text",
    num_beams: int = 1,
    fp16: bool = True,
):

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # punctuations = string.punctuation
    # punctuations = re.sub(r"-'", "", string.punctuation)
    punctuations = "!,.?"

    # load the restoring class
    model_kwargs = {}
    if fp16:
        model_kwargs["torch_dtype"] = torch.float16

    restorer = PunctuationRestorer.from_pretrained(model_name_or_path, punctuations=punctuations, **model_kwargs)
    restorer.model.eval()
    restorer.to("cuda")
    print("Model has been loaded")

    # set lang token id
    restorer.set_language(language)

    dataset = load_dataset("json", data_files=input_file_path, split="train")
    print(dataset)

    dataset = dataset.map(
        lambda x: {f"tmp_{audio_column_name}": x[audio_column_name]},
        num_proc=16,
    )

    dataset = dataset.cast_column(audio_column_name, Audio(sampling_rate=SAMPLING_RATE, mono=True))

    # Uncomment for testing
    # dataset = dataset.select(range(10))

    def restore(example):
        audio = example[audio_column_name]["array"]
        sampling_rate = example[audio_column_name]["sampling_rate"]

        text = re.sub(rf"[{re.escape(punctuations)}]", "", example[text_column_name]).lower()

        # try:
        restored_text, probs = restorer(audio, text, sampling_rate=sampling_rate, num_beams=num_beams)
        # except RuntimeError:
        #     restored_text, probs = "", None

        # example.pop(audio_column_name)
        # example[audio_column_name] = example.pop(f"tmp_{audio_column_name}")
        # example["restored_pnc_text"] = restored_text
        # example["restored_pnc_probs"] = probs

        # result = {
        #     "restored_pnc_text": restored_text,
        #     "restored_pnc_probs": probs,
        # }

        # with open(output_file_path, "a") as manifest_f:
        #     manifest_f.write(f"{json.dumps(result, ensure_ascii=False)}\n")

        return {"restored_pnc_text": restored_text, "restored_pnc_probs": probs}

    start_time = time.perf_counter()

    dataset = dataset.map(
        restore,
        # remove_columns=dataset.column_names,
        # keep_in_memory=True,
        desc="Restore punctuation and case",
    )
    # print(dataset)

    dataset = dataset.remove_columns([audio_column_name])
    dataset = dataset.map(
        lambda x: {audio_column_name: x[f"tmp_{audio_column_name}"]},
        remove_columns=[f"tmp_{audio_column_name}"],
        num_proc=16,
    )

    # dataset.to_json(output_file_path, orient="records", lines=True, force_ascii=False)
    # better handle backslash in path
    with open(output_file_path, "w") as manifest_f:
        for _, sample in enumerate(tqdm.tqdm(dataset, desc="Saving", total=dataset.num_rows, unit=" samples")):
            manifest_f.write(f"{json.dumps(sample, ensure_ascii=False)}\n")

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}. The generated"
        f" data is saved in {output_file_path}"
    )


if __name__ == "__main__":
    fire.Fire(main)
