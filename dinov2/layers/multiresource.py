"""Ordered tactile pooling and a local point-token transformer.

The point encoder uses Point-BERT's group/token/transformer idea, NOT its exact
architecture or official weights. Its initialization is explicitly random.
No rendered point images, custom CUDA extension, or frozen random encoder.
"""
import torch
from torch import nn
from torch.nn import functional as F


class OrderedClipHead(nn.Module):
    def __init__(self, width, output=256, frames=4):
        super().__init__()
        self.frames = frames
        self.frame = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, output), nn.GELU())
        self.ordered = nn.Sequential(nn.Linear(output*frames, output), nn.LayerNorm(output))

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] != self.frames:
            raise ValueError("ordered complete clip required")
        return self.ordered(self.frame(tokens).flatten(1))


class LocalPointTransformer(nn.Module):
    def __init__(self, width=192, groups=16, neighbors=16, depth=4):
        super().__init__()
        self.groups, self.neighbors = groups, neighbors
        self.embed_dim = width
        self.point = nn.Sequential(nn.Linear(3, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, width))
        self.position = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, 6, width*4, dropout=0., batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, points):
        if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < self.neighbors:
            raise ValueError("B,N,3 actual local points required")
        with torch.no_grad():
            p = points.float()
            b, n, _ = p.shape
            chosen = p.square().sum(-1).argmax(-1)
            dist = torch.full((b, n), float("inf"), device=p.device)
            batch = torch.arange(b, device=p.device)
            centers = []
            for _ in range(self.groups):
                center = p[batch, chosen]
                centers.append(center)
                dist = torch.minimum(dist, (p-center[:, None]).square().sum(-1))
                chosen = dist.argmax(-1)
            centers = torch.stack(centers, 1)
            idx = torch.cdist(centers, p).topk(self.neighbors, largest=False).indices
            grouped = p[batch[:, None, None], idx] - centers[:, :, None]
        token = self.point(grouped.to(points.dtype)).amax(2) + self.position(centers.to(points.dtype))
        x = torch.cat([self.cls.expand(len(points), -1, -1), token], 1)
        return self.norm(self.transformer(x))[:, 0]


def symmetric_contact_loss(left, right, ids, temperature=.07, distributed=True):
    """FP32 all-rank InfoNCE, repeated contact IDs are positive sets."""
    left, right = F.normalize(left.float(), dim=-1), F.normalize(right.float(), dim=-1)
    if not 0 < temperature <= 1:
        raise ValueError("invalid temperature")
    ids = ids.reshape(-1)
    a, b, all_ids = left, right, ids
    import torch.distributed as dist
    if distributed and dist.is_initialized() and dist.get_world_size() > 1:
        from torch.distributed.nn.functional import all_gather
        a, b = torch.cat(all_gather(left)), torch.cat(all_gather(right))
        target = [torch.empty_like(ids) for _ in range(dist.get_world_size())]
        dist.all_gather(target, ids); all_ids = torch.cat(target)
    positive = ids[:, None] == all_ids[None]
    if not positive.any(1).all():
        raise ValueError("missing positive contact")
    def loss(logits):
        return (torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), 1)).mean()
    return .5*(loss(left @ b.T / temperature)+loss(right @ a.T / temperature))
