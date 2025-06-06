#!/usr/bin/env python3
import logging
import re
from pathlib import Path

from lhotse import CutSet, SupervisionSegment
from lhotse.recipes.utils import read_manifests_if_cached


def normalize_text(
    utt: str,
    whitespace_pattern=re.compile(r"\s\s+"),
) -> str:
    return whitespace_pattern.sub(" ", punct_pattern.sub("", utt))


def preprocess_dataoceanai_alg():
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    output_dir.mkdir(exist_ok=True)

    dataset_parts = ("train", "test")

    logging.info("Loading manifest")
    prefix = "dataoceanai-alg"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    for partition, m in manifests.items():
        logging.info(f"Processing {partition}")
        raw_cuts_path = output_dir / f"{prefix}_cuts_{partition}_raw.{suffix}"
        if raw_cuts_path.is_file():
            logging.info(f"{partition} already exists - skipping")
            continue

        # Note this step makes the recipe different than LibriSpeech:
        # We must filter out some utterances and remove punctuation
        # to be consistent with Kaldi.
        logging.info("Filtering OOV utterances from supervisions")
        # m["supervisions"] = m["supervisions"].filter(has_no_oov)
        logging.info(f"Normalizing text in {partition}")
        # for sup in m["supervisions"]:
        #   sup.text = normalize_text(sup.text)

        # Create long-recording cut manifests.
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )
        # Run data augmentation that needs to be done in the
        # time domain.
        #  if partition not in ["DEV", "TEST"]:
        #      logging.info(
        #          f"Speed perturb for {partition} with factors 0.9 and 1.1 "
        #          "(Perturbing may take 8 minutes and saving may"
        #          " take 20 minutes)"
        #      )
        #      cut_set = (
        #          cut_set
        #          + cut_set.perturb_speed(0.9)
        #          + cut_set.perturb_speed(1.1)
        #      )
        #
        # Note: No need to perturb the training subset as not all of the
        # data is going to be used in the training.
        logging.info(f"Saving to {raw_cuts_path}")
        cut_set.to_file(raw_cuts_path)


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    preprocess_dataoceanai_alg()


if __name__ == "__main__":
    main()
