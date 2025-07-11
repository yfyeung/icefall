#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file takes as input the filename of LM training data
generated by ./local/prepare_lm_training_data.py and sorts
it by sentence length.

Sentence length equals to the number of BPE tokens in a sentence.
"""

import argparse
import logging
from pathlib import Path

import k2
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-lm-data",
        type=str,
        help="Input LM training data, e.g., data/bpe_500/lm_data.pt",
    )

    parser.add_argument(
        "--out-lm-data",
        type=str,
        help="Input LM training data, e.g., data/bpe_500/sorted_lm_data.pt",
    )

    parser.add_argument(
        "--out-statistics",
        type=str,
        help="Statistics about LM training data., data/bpe_500/statistics.txt",
    )

    return parser.parse_args()


def main():
    args = get_args()
    in_lm_data = Path(args.in_lm_data)
    out_lm_data = Path(args.out_lm_data)
    assert in_lm_data.is_file(), f"{in_lm_data}"
    if out_lm_data.is_file():
        logging.warning(f"{out_lm_data} exists - skipping")
        return
    data = torch.load(in_lm_data, weights_only=False)
    words2bpe = data["words"]
    sentences = data["sentences"]
    sentence_lengths = data["sentence_lengths"]

    num_sentences = sentences.dim0
    assert num_sentences == sentence_lengths.numel(), (
        num_sentences,
        sentence_lengths.numel(),
    )

    indices = torch.argsort(sentence_lengths, descending=True)

    sorted_sentences = sentences[indices.to(torch.int32)]
    sorted_sentence_lengths = sentence_lengths[indices]

    # Check that sentences are ordered by length
    assert num_sentences == sorted_sentences.dim0, (
        num_sentences,
        sorted_sentences.dim0,
    )

    cur = None
    for i in range(num_sentences):
        word_ids = sorted_sentences[i]
        token_ids = words2bpe[word_ids]
        if isinstance(token_ids, k2.RaggedTensor):
            token_ids = token_ids.values
        if cur is not None:
            assert cur >= token_ids.numel(), (cur, token_ids.numel())

        cur = token_ids.numel()
        assert cur == sorted_sentence_lengths[i]

    data["sentences"] = sorted_sentences
    data["sentence_lengths"] = sorted_sentence_lengths
    torch.save(data, args.out_lm_data)
    logging.info(f"Saved to {args.out_lm_data}")

    statistics = Path(args.out_statistics)

    # Write statistics
    num_words = sorted_sentences.numel()
    num_tokens = sentence_lengths.sum().item()
    max_sentence_length = sentence_lengths[indices[0]]
    min_sentence_length = sentence_lengths[indices[-1]]

    step = 10
    hist, bins = np.histogram(
        sentence_lengths.numpy(),
        bins=np.arange(1, max_sentence_length + step, step),
    )

    histogram = np.stack((bins[:-1], hist)).transpose()

    with open(statistics, "w") as f:
        f.write(f"num_sentences: {num_sentences}\n")
        f.write(f"num_words: {num_words}\n")
        f.write(f"num_tokens: {num_tokens}\n")
        f.write(f"max_sentence_length: {max_sentence_length}\n")
        f.write(f"min_sentence_length: {min_sentence_length}\n")
        f.write("histogram:\n")
        f.write("  bin  count  percent\n")
        for row in histogram:
            f.write(
                f"{int(row[0]):>5} {int(row[1]):>5}   "
                f"{100.*row[1]/num_sentences:.3f}%\n"
            )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
