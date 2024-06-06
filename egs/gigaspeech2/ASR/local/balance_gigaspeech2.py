#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.             (Yifan Yang)
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

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

import jsonlines
from lhotse import CutSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm import tqdm

from icefall.utils import str2bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuts-file",
        type=str,
        action="append",
        required=True,
    )

    return parser.parse_args()


def process_jsonl(args):
    frequency_dict = defaultdict(int)
    text_dict = {}
    for cuts_path in args.cuts_file:
        with jsonlines.open(cuts_path) as reader:
            for line in tqdm(reader):
                text = line["supervisions"][0]["text"]
                key = hash(text)
                frequency_dict[key] += 1
                text_dict[key] = text

    frequency_dict = dict(
        sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)
    )
    drop_rate_dict = defaultdict(float)
    print(list(frequency_dict.values())[:100])
    for key, value in frequency_dict.items():
        print(value, text_dict[key])
        message = (
            "Type r to remove all samples\n"
            + "Type number to keep custom number of samples\n"
            + "Type s to skip.\n"
            + "Type q to quit.\n"
        )
        cmd = input(message)
        if cmd == "r":
            drop_rate_dict[key] = 100
        elif cmd == "s":
            drop_rate_dict[key] = 0
        elif cmd == "q":
            break
        else:
            drop_rate_dict[key] = 1 - min(int(cmd) / value, 1)
    print(drop_rate_dict)

    for cuts_path in args.cuts_file:
        with jsonlines.open(cuts_path) as reader, jsonlines.open(
            cuts_path.replace(".jsonl", f"_balanced.jsonl"), "w"
        ) as writer:
            for line in tqdm(reader):
                text = line["supervisions"][0]["text"]
                key = hash(text)
                if random.random() >= drop_rate_dict[key]:
                    writer.write(line)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    process_jsonl(args)


if __name__ == "__main__":
    main()
