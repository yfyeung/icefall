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
from pathlib import Path

from jiwer import cer
from lhotse import CutSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import str2bool


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recogs-file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
    )

    return parser.parse_args()


def filter_gigaspeech2(args):
    in_out_dir = Path("data/fbank")

    with open(args.recogs_file) as f:
        lines = f.read().splitlines()
        lines = iter(lines)

    tot_cnt = 0
    val_cnt = 0
    while True:
        try:
            ref_id, ref = next(lines).split("\t")
            hyp_id, hyp = next(lines).split("\t")
            assert ref_id == hyp_id, f"{ref_id}, {hyp_id}"

            ref = ref.replace("ref=[", "").replace("]", "").replace("'", "").replace(",", "").replace(" ", "")
            hyp = hyp.replace("hyp=[", "").replace("]", "").replace("'", "").replace(",", "").replace(" ", "")
            score = cer(ref, hyp)

            tot_cnt += 1
            if score <= args.threshold:
                val_cnt += 1
            if tot_cnt > 20:
                break
            print(score)
            print(ref)
            print(hyp)
        except StopIteration:
            break

    logging.info(
        f"total cuts: {tot_cnt}, filtered cuts: {tot_cnt - val_cnt}, filtered rate: {1 - val_cnt / tot_cnt}"
    )


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    args = get_args()
    filter_gigaspeech2(args)


if __name__ == "__main__":
    main()
