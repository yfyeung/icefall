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

import logging
import re
from pathlib import Path

import editdistance
import torch
from funasr import AutoModel
from num2words import num2words
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from lhotse import CutSet
from tqdm import tqdm


def normalize_text(text: str) -> str:
    # Convert to upper case
    text = text.upper()

    # Remove brackets with content
    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
    text = re.sub(r"\([^\)]*\)", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    def _num2words(match):
        number = int(match.group())
        return num2words(number, lang="ko")

    # Replace numbers with Korean words
    text = re.sub(r"\d+", _num2words, text)

    # Remove blank symbols
    text = re.sub(r"\s+", "", text).strip()

    return text

def annotate_with_sensevoice():
    in_out_dir = Path("data/fbank")

    dataset_parts = (
        # "test",
        "ytnnews24-0",
        # "ytnnews24-1",
    )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    asr_model = AutoModel(
        model="FunAudioLLM/SenseVoiceSmall", device="cuda:0", hub="hf"
    )

    asr_error = 0
    asr_total = 0

    for partition in dataset_parts:
        cuts_path = in_out_dir / f"gigaspeech3_cuts_{partition}_annotated.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue

        raw_cuts_path = in_out_dir / f"gigaspeech3_cuts_{partition}_raw.jsonl.gz"

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = CutSet.from_file(raw_cuts_path)

        logging.info("Annotating audios")
        for cut in tqdm(cut_set):
            res = asr_model.generate(
                input=cut.recording.sources[0].source,
                cache={},
                language="ko",  # "zn", "en", "yue", "ja", "ko", "nospeech"
                use_itn=False,
                batch_size=64,
                hub="hf",
            )
            asr_hyp = normalize_text(rich_transcription_postprocess(res[0]["text"]))
            asr_ref = normalize_text(cut.supervisions[0].text)
            asr_error = editdistance.eval(list(asr_hyp), list(asr_ref))
            asr_total = len(list(asr_ref))

            cer = asr_error * 1.0 / asr_total

def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    annotate_with_sensevoice()


if __name__ == "__main__":
    main()
