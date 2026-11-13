"""Explicit temporal aggregation for complete contact clips, not forecasting.

The legacy head remains loadable. Changing heads requires a new run/config;
it is not a compatible optimizer-state continuation of an old experiment.
"""
import torch
from torch import nn

REGION_CLIP_HEADS = ("temporal_regions", "temporal_trajectory")
CLIP_HEADS = ("ordered_concat", "temporal_attention", *REGION_CLIP_HEADS)


def ordered_trajectory(states):
    """Invertible mean + signed adjacent changes; never pool away time slots.

    B,T,...,D -> B,...,T*D. A constant sequence has exactly zero changes.
    Reversal negates and reverses changes, while leaving appearance unchanged.
    No normalization of individual differences: tiny noise stays tiny.
    """
    if states.ndim < 3 or states.shape[1] < 2:
        raise ValueError('trajectory requires at least two ordered states')
    blocks = [states.mean(1), *(states[:, i]-states[:, i-1] for i in range(1, states.shape[1]))]
    return torch.cat(blocks, dim=-1)


class TemporalTrajectoryRegionHead(nn.Module):
    """Contextualize frames, then preserve their signed trajectory explicitly.

    Attention has no positional bias: identical observations must not produce
    artificial motion. Temporal order is represented by fixed difference slots.
    The existing three cross-modal losses supervise this one tactile output.
    """
    def __init__(self, width, output=256, frames=4):
        super().__init__()
        if width < 1 or frames < 2 or output % frames or (output//frames) % 4:
            raise ValueError('trajectory output/frames must be divisible by four')
        self.width, self.frames, self.output = width, frames, output
        slot = output//frames
        self.frame = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, slot), nn.GELU())
        layer = nn.TransformerEncoderLayer(slot, 4, slot*2, dropout=0.,
            activation='gelu', batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(slot)

    def forward(self, tokens):
        if tokens.ndim != 4 or tokens.shape[1] != self.frames or tokens.shape[-1] != self.width:
            raise ValueError('expected B x ordered frames x tactile regions x width')
        b, f, r, d = tokens.shape
        x = tokens.permute(0, 2, 1, 3).reshape(b*r, f, d)
        states = self.norm(self.temporal(self.frame(x)))
        return ordered_trajectory(states).reshape(b, r, self.output)


class TemporalClipHead(nn.Module):
    def __init__(self, width, output=256, frames=4):
        super().__init__()
        if width < 1 or output < 4 or output % 4 or frames < 2:
            raise ValueError("temporal head needs positive width, output divisible by 4, and >=2 frames")
        self.width, self.frames = width, frames
        self.frame = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output), nn.GELU())
        self.position = nn.Parameter(torch.empty(1, frames, output))
        self.clip_token = nn.Parameter(torch.empty(1, 1, output))
        layer = nn.TransformerEncoderLayer(output, 4, output*2, dropout=0.,
            activation="gelu", batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(output)
        nn.init.trunc_normal_(self.position, std=.02)
        nn.init.trunc_normal_(self.clip_token, std=.02)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (self.frames, self.width):
            raise ValueError("expected B x complete ordered frames x encoder width")
        x = self.frame(tokens) + self.position
        x = torch.cat([self.clip_token.expand(len(x), -1, -1), x], dim=1)
        # Bidirectional context is intentional: the input is the complete
        # contact recording. Neither future targets nor another modality enter.
        return self.norm(self.temporal(x)[:, 0])


class TemporalRegionHead(TemporalClipHead):
    """One shared temporal head per sensor-image region; one tactile backbone.

    Indices correspond only within the fixed tactile camera recording, never
    across RGB/depth/PC. Across modalities the contrastive scorer soft-matches.
    """
    def forward(self, tokens):
        if tokens.ndim != 4 or tokens.shape[1] != self.frames or tokens.shape[-1] != self.width:
            raise ValueError('expected B x complete frames x tactile regions x width')
        b, f, r, d = tokens.shape
        regions = tokens.permute(0, 2, 1, 3).reshape(b*r, f, d)
        return super().forward(regions).reshape(b, r, -1)


def clip_head_contract(kind, width, output=256, frames=4):
    if kind not in CLIP_HEADS:
        raise ValueError("unregistered tactile clip head: " + str(kind))
    if kind == 'temporal_trajectory':
        return dict(kind=kind, width=width, output=output, frames=frames,
            temporal_positions='explicit_signed_adjacent_change_slots', attention_layers=1,
            attention_heads=4, input='per_frame_patch_tokens', regions=49,
            output_layout='B_regions_mean_then_ordered_changes',
            slot_width=output//frames, extra_supervision=False, future_prediction=False)
    if kind == 'temporal_regions':
        return dict(kind=kind, width=width, output=output, frames=frames,
            temporal_positions='learned_recording_progress', attention_layers=1, attention_heads=4,
            input='per_frame_patch_tokens', regions=49, output_layout='B_regions_D',
            extra_supervision=False, future_prediction=False)
    return dict(kind=kind, width=width, output=output, frames=frames,
        temporal_positions="learned_recording_progress" if kind == "temporal_attention" else "ordered_slots",
        attention_layers=1 if kind == "temporal_attention" else 0,
        attention_heads=4 if kind == "temporal_attention" else 0,
        input="per_frame_CLS", extra_supervision=False, future_prediction=False)


def build_clip_head(kind, width, output=256, frames=4):
    clip_head_contract(kind, width, output, frames)
    if kind == 'temporal_trajectory':
        return TemporalTrajectoryRegionHead(width, output, frames)
    if kind == 'temporal_regions':
        return TemporalRegionHead(width, output, frames)
    if kind == "temporal_attention":
        return TemporalClipHead(width, output, frames)
    from dinov2.layers.multiresource import OrderedClipHead
    return OrderedClipHead(width, output, frames)


def build_controlled_clip_head(kind, width, output=256, frames=4):
    # Both explicit ablations enter with the same CPU RNG state. Different
    # head sizes must not change subsequent RGB/depth/PC projection weights.
    with torch.random.fork_rng(devices=[]):
        return build_clip_head(kind, width, output, frames)
