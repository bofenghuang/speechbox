#!/usr/bin/env python
# coding=utf-8
# Copyright 2023  Bofeng Huang

import sys

sys.path.append("/home/bhuang/myscripts")

import json
import os
import re
import string
import time

import fire
import tqdm
from datasets import load_dataset

# from datasets.utils.logging import disable_progress_bar
from text_normalization.normalize_french import FrenchTextNormalizer

# disable_progress_bar()


def main(
    input_file_path: str,
    output_file_path: str,
):

    dataset = load_dataset("json", data_files=input_file_path, split="train")
    print(dataset)

    normalizer = FrenchTextNormalizer()

    def process_function(example):
        s = example["restored_pnc_text"]
        s = normalizer(
            s,
            do_lowercase=False,
            do_ignore_words=False,
            symbols_to_keep="'-,.?!:;$%@&#~()",
            do_num2text=False,
            do_text2num=True,
        )
        s = re.sub(rf"(?<=[\.\?!]\s)([{normalizer.kept_chars}])", lambda x: x.group().upper(), s)  # Uppercase 1st non-space character after .?!
        s = re.sub(rf"^([{normalizer.kept_chars}])", lambda x: x.group().upper(), s)  # Uppercase 1st character
        s = re.sub(r"[,]+$", ".", s)  # replace comma at the end
        s= re.sub(rf"([{normalizer.kept_chars}])$", r"\1.", s)  # add period if no punctuation at the end
        example["text"] = s
        return example

    start_time = time.perf_counter()

    dataset = dataset.map(
        process_function,
        remove_columns=["restored_pnc_text", "restored_pnc_probs"],
        # keep_in_memory=True,
        num_proc=8,
        desc="process",
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
