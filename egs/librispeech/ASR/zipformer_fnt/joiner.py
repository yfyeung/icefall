# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

import torch
import torch.nn as nn
from scaling import ScaledLinear, BiasNorm


class Joiner(nn.Module):
    def __init__(
        self,
        joint_dim: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim,
        vocab_size: int,
        blank_id: int,
    ):
        super().__init__()
        self.joint_dim = joint_dim
        self.encoder_embed_dim = encoder_hidden_dim
        self.decoder_embed_dim = decoder_hidden_dim
        self.blank_id = blank_id
        # add blank symbol in output layer
        self.out_dim = vocab_size
        self.out_blank_dim = 1
        self.out_vocab_dim = self.out_dim - self.out_blank_dim

        self.fc_encoder_proj = ScaledLinear(self.encoder_embed_dim, self.joint_dim)
        self.laynorm_proj_encoder = BiasNorm(self.joint_dim)
        self.fc_blank_decoder = ScaledLinear(self.decoder_embed_dim, self.joint_dim)
        self.laynorm_proj_blank_decoder = BiasNorm(joint_dim)
        self.laynorm_proj_vocab_decoder = BiasNorm(decoder_hidden_dim)

        self.fc_out_blank = ScaledLinear(self.joint_dim, self.out_blank_dim)
        self.fc_out_encoder_vocab = ScaledLinear(self.joint_dim, self.out_vocab_dim)
        self.fc_out_decoder_vocab = ScaledLinear(self.joint_dim, self.out_vocab_dim)

        # nn.init.normal_(self.encoder_proj.weight, mean=0, std=self.joint_dim**-0.5)
        # nn.init.normal_(self.blank_decoder_proj.weight, mean=0, std=self.joint_dim**-0.5)
        # nn.init.normal_(self.fc_out_blank.weight, mean=0, std=self.joint_dim**-0.5)
        # nn.init.normal_(self.fc_out_encoder_vocab.weight, mean=0, std=self.joint_dim**-0.5)
        # nn.init.normal_(self.fc_out_decoder_vocab.weight, mean=0, std=self.joint_dim**-0.5)

    # encoder_out: B x T x C
    # decoder_out: B X U x C

    def encoder_proj(self, encoder_out):
        """_summary_

        Args:
            Output from the encoder. Its shape is (N, T, s_range, C).

        Returns:
            project of the encoder
        """
        encoder_out = self.laynorm_proj_encoder(self.fc_encoder_proj(encoder_out))
        return encoder_out

    def blank_decoder_proj(self, blank_decoder_out):
        blank_decoder_out = self.laynorm_proj_blank_decoder(
            self.fc_blank_decoder(blank_decoder_out)
        )
        return blank_decoder_out

    def vocab_decoder_proj(self, vocab_decoder_out):
        vocab_decoder_out = self.laynorm_proj_vocab_decoder(vocab_decoder_out)
        vocab_decoder_out = self.fc_out_decoder_vocab(vocab_decoder_out)
        vocab_decoder_out = torch.nn.functional.log_softmax(vocab_decoder_out, dim=-1)
        return vocab_decoder_out

    def forward(
        self, encoder_out, blank_decoder_out, vocab_decoder_out, project_input=True
    ):
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
           project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        if project_input:
            encoder_out = self.encoder_proj(encoder_out)
            blank_decoder_out = self.blank_decoder_proj(blank_decoder_out)
            vocab_decoder_out = self.vocab_decoder_proj(vocab_decoder_out)

        blank_prob = nn.functional.relu(encoder_out + blank_decoder_out)
        blank_prob = self.fc_out_blank(blank_prob)

        vocab_encoder_prob = self.fc_out_encoder_vocab(encoder_out)
        vocab_decoder_prob = vocab_decoder_out
        vocab_prob = vocab_encoder_prob + vocab_decoder_prob
        out = torch.cat(
            (
                vocab_prob[:, :, :, : self.blank_id],
                blank_prob,
                vocab_prob[:, :, :, self.blank_id :],
            ),
            dim=-1,
        )
        return out, vocab_decoder_prob
