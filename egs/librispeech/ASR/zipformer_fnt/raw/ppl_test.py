#!/usr/bin/env python3
#
# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang)
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
Usage:
(1) greedy search
./transducer/decode.py \
        --epoch 14 \
        --avg 7 \
        --exp-dir ./transducer/exp \
        --max-duration 100 \
        --decoding-method greedy_search

(2) beam search
./transducer/decode.py \
        --epoch 14 \
        --avg 7 \
        --exp-dir ./transducer/exp \
        --max-duration 100 \
        --decoding-method beam_search \
        --beam-size 8
"""


import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from gigaspeech_asr_datamodule import GigaSpeechAsrDataModule
from scaling import ScheduledFloat
from zipformer import Zipformer2
from decoder import Decoder
from joiner import Joiner
from model import AsrModel
from subsampling import Conv2dSubsampling
from train import add_model_arguments, get_model

from icefall.checkpoint import average_checkpoints, load_checkpoint
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    write_error_stats,
    add_sos,
)
import k2
import sys


class AdaptationModel(nn.Module):
    def __init__(
        self,
        vocab_decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.vocab_decoder = vocab_decoder
        self.joiner = joiner
        self.blank_id = vocab_decoder.blank_id

    def forward(self, y):
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]
        lm_target = torch.clone(y.values)
        # since we don't use a pad_id, we just concat the labels, which can be done by using its values
        lm_target = lm_target.to(torch.int64)
        lm_target[lm_target > self.blank_id] -= 1

        sos_y = add_sos(y, sos_id=self.blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=self.blank_id)
        vocab_decoder_out, _ = self.vocab_decoder(sos_y_padded)
        lm_probs = self.joiner.vocab_decoder_proj(vocab_decoder_out)
        # lm_probs = [item[:y_len] for item, y_len in zip(lm_probs, y_lens)]

        result = dict()
        # result['y'] = y
        result["label"] = []
        result["predict"] = []
        result["ppl"] = []
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        acc_len = 0
        ppl = 0
        for i in range(lm_probs.shape[0]):
            result["ppl"].append(
                torch.exp(
                    loss_fn(
                        lm_probs[i, : y_lens[i]],
                        lm_target[acc_len : acc_len + y_lens[i]],
                    )
                )
            )
            ppl += result["ppl"][-1]
            result["predict"].append(
                torch.argmax(lm_probs[i, : y_lens[i]], dim=-1)
                .cpu()
                .detach()
                .numpy()
                .tolist()
            )
            result["label"].append(
                lm_target[acc_len : acc_len + y_lens[i]].cpu().detach().numpy().tolist()
            )
            acc_len += y_lens[i]

        # note that the last output of each sentence is not used here
        return ppl, result


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-source",
        type=str,
        default="transducer-checkpoint",
        help="type of the source of decoder state dict, can be transducer-checkpoint, adaptation-checkpoint, transducer-dict, adaptation-dict",
    )

    parser.add_argument(
        "--init-state-dict",
        type=str,
        default=None,
        help="type of the source of decoder state dict, can be transducer-checkpoint, adaptation-checkpoint, transducer-dict, adaptation-dict",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=34,
        help="It specifies the checkpoint to use for decoding."
        "Note: Epoch counts from 0.",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=11,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch'. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="transducer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Used only when --decoding-method is beam_search",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="transducer",
        help="the model to compute ppl, can be transducer or adaptation",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech_dataset",
        help="which dataset to test",
    )

    add_model_arguments(parser)
    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            # parameters for conformer
            "feature_dim": 80,
            "subsampling_factor": 4,  # not passed in, this is fixed.
            "warm_step": 2000,
            # decoder params
            "num_decoder_layers": 2,
            "env_info": get_env_info(),
        }
    )
    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder


def get_decoder_model(params: AttributeDict):
    decoder = Decoder(
        vocab_size=params.vocab_size,
        embedding_dim=params.decoder_embedding_dim,
        blank_id=params.blank_id,
        num_layers=params.num_decoder_layers,
        hidden_dim=params.decoder_dim,
        output_dim=-1,  # we move the output project layer to the joiner
    )
    return decoder


def ppl_one_batch(
    model: nn.Module, sp: spm.SentencePieceProcessor, batch: dict, batch_idx: int
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = model.device

    # Now for the decoder, i.e., the prediction network
    texts = batch
    texts = [item.replace("-", " ") for item in texts]
    y = sp.encode(texts, out_type=int)
    y_old = y
    y = k2.RaggedTensor(y).to(device)
    ppl, result = model(y)
    result["batch_idx"] = [batch_idx] * len(batch)
    result["y"] = y_old
    result["text"] = texts
    return ppl, len(batch), result


def compute_ppl_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    total_size = 0
    total_ppl = 0
    result = {
        "batch_idx": [],
        "y": [],
        "label": [],
        "predict": [],
        "ppl": [],
        "text": [],
    }
    for batch_idx, batch in enumerate(dl):
        ppl, batch_size, batch_result_dict = ppl_one_batch(
            model=model, sp=sp, batch=batch, batch_idx=batch_idx
        )
        total_size += batch_size
        total_ppl += ppl
        for name, item in batch_result_dict.items():
            result[name].extend(item)

    return total_ppl / total_size, result


def save_results(
    params: AttributeDict,
    test_set_name: str,
    ppl,
    results_dict,
):
    recog_path = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
    with open(recog_path, "w") as f:
        for i in range(len(results_dict["y"])):
            print("-" * 30, file=f)
            print(
                f"ppl: {results_dict['ppl'][i]}\nlabel: {results_dict['text'][i]}\nid:{results_dict['label'][i]}",
                file=f,
            )

    pred_path = params.res_dir / f"pred-{test_set_name}-{params.suffix}.txt"
    with open(pred_path, "w") as f:
        for i in range(len(results_dict["y"])):
            print(f"batch_idx: {results_dict['batch_idx'][i]}", file=f)
            print(f"\ty: {results_dict['y'][i]}", file=f)
            print(f"\tlabel: {results_dict['label'][i]}", file=f)
            print(f"\tpredict: {results_dict['predict'][i]}", file=f)
    errs_info = params.res_dir / f"ppl-{test_set_name}-{params.suffix}.txt"
    with open(errs_info, "w") as f:
        print(f"ppl: \t{ppl}", file=f)


@torch.no_grad()
def main():
    parser = get_parser()

    raw_args = sys.argv
    datamodule = None
    for item in raw_args:
        if item == "librispeech_dataset":
            LibriSpeechAsrDataModule.add_arguments(parser)
            datamodule = LibriSpeechAsrDataModule
            break
        elif item == "gigaspeech_dataset":
            GigaSpeechAsrDataModule.add_arguments(parser)
            datamodule = GigaSpeechAsrDataModule
            break
    if datamodule is None:
        raise Exception("dataset not found")

    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / "ppl"

    if params.init_state_dict is not None:
        state_dict_name = params.init_state_dict.replace("/", ".")
        params.suffix = f"init-state-dict-{state_dict_name}-dataset-{params.dataset}"
    else:
        params.suffix = (
            f"epoch-{params.epoch}-avg-{params.avg}-dataset-{params.dataset}"
        )

    setup_logger(f"{params.res_dir}/log-ppl-{params.suffix}")
    logging.info("Computing ppl")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)
    print(params.model_source)
    if params.model_source == "transducer-checkpoint":
        if params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.load_state_dict(
                average_checkpoints(filenames, device=torch.device("cpu"))
            )
        model = AdaptationModel(model.vocab_decoder, model.joiner)
    elif params.model_source == "transducer-dict":
        model.load_state_dict(torch.load(params.init_state_dict))
        model = AdaptationModel(model.vocab_decoder, model.joiner)
    elif params.model_source == "adaptation-checkpoint":
        model = AdaptationModel(model.vocab_decoder, model.joiner)
        if params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.load_state_dict(
                average_checkpoints(filenames, device=torch.device("cpu"))
            )
    elif params.model_source == "adaptation-dict":
        model = AdaptationModel(model.vocab_decoder, model.joiner)
        model.load_state_dict(torch.load(params.init_state_dict))

    model.to(device)
    model.eval()
    model.device = device

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    if datamodule == LibriSpeechAsrDataModule:
        librispeech = LibriSpeechAsrDataModule(args)

        test_clean_cuts = librispeech.test_clean_cuts()
        test_other_cuts = librispeech.test_other_cuts()

        test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
        test_other_dl = librispeech.test_dataloaders(test_other_cuts)

        test_sets = ["test-clean", "test-other"]
        test_dl = [test_clean_dl, test_other_dl]
    else:
        gigaspeech = GigaSpeechAsrDataModule(args)

        dev_cuts = gigaspeech.dev_text()
        test_cuts = gigaspeech.test_text()

        dev_dl = gigaspeech.test_text_dataloaders(dev_cuts, False, False)
        test_dl = gigaspeech.test_text_dataloaders(test_cuts, False, False)

        test_sets = ["dev", "test"]
        test_dl = [dev_dl, test_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        ppl, results_dict = compute_ppl_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
        )

        save_results(
            params=params, test_set_name=test_set, ppl=ppl, results_dict=results_dict
        )

    logging.info("Done!")


# called in train.py and only can be called once
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
