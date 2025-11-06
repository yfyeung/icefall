import torch
import torch.distributed.nn
from torch import distributed as dist
from torch import nn as nn
from torch.nn import functional as F


def gather_features(
    audio_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    # We gather tensors from all gpus
    if gather_with_grad:
        all_audio_features = torch.cat(
            torch.distributed.nn.all_gather(audio_features), dim=0
        )
        all_text_features = torch.cat(
            torch.distributed.nn.all_gather(text_features), dim=0
        )
    else:
        gathered_audio_features = [
            torch.zeros_like(audio_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_audio_features, audio_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_audio_features[rank] = audio_features
            gathered_text_features[rank] = text_features

        all_audio_features = torch.cat(gathered_audio_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_audio_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(
        self,
        audio_features,
        text_features,
        logit_scale,
    ):
        device = audio_features.device

        if self.world_size > 1:
            all_audio_features, all_text_features = gather_features(
                audio_features=audio_features,
                text_features=text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
            )

            if self.local_loss:
                logits_per_audio = logit_scale * audio_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_audio_features.T
            else:
                logits_per_audio = (
                    logit_scale * all_audio_features @ all_text_features.T
                )
                logits_per_text = logits_per_audio.T
        else:
            logits_per_audio = logit_scale * audio_features @ text_features.T
            logits_per_text = logit_scale * text_features @ audio_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_audio.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_audio, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss
