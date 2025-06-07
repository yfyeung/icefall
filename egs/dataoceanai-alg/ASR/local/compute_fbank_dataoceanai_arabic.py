#!/usr/bin/env python3

import logging
from pathlib import Path

import torch
from lhotse import CutSet, KaldifeatFbank, KaldifeatFbankConfig

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_dataoceanai_arabic():
    in_out_dir = Path("data/fbank")
    # number of workers in dataloader
    num_workers = 20

    # number of seconds in a batch
    batch_duration = 1000

    subsets = ("train", "test")
    prefix = "dataoceanai-alg"

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))

    logging.info(f"device: {device}")

    for partition in subsets:
        cuts_path = in_out_dir / f"{prefix}_cuts_{partition}.jsonl.gz"
        if cuts_path.is_file():
            logging.info(f"{cuts_path} exists - skipping")
            continue

        raw_cuts_path = in_out_dir / f"{prefix}_cuts_{partition}_raw.jsonl.gz"

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = CutSet.from_file(raw_cuts_path)

        logging.info("Computing features")

        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{in_out_dir}/{prefix}_feats_{partition}",
            num_workers=num_workers,
            batch_duration=batch_duration,
            overwrite=True,
        )
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)
        logging.info(f"Saved to {cuts_path}")


def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank_dataoceanai_arabic()


if __name__ == "__main__":
    main()
