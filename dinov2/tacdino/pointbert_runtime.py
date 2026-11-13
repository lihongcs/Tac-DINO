"""Official pretrained Point-BERT, shared by training and neutral evaluation."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from dinov2.tacdino.file_records import sha, digest

WEIGHT_SHA256 = '4d595a2d4a4d5755d2b10104634f851e2878bc07dc7fc26c0e9f99f162de504f'
UPSTREAM_COMMIT = 'ce5a6e1f35a0d111de12c3a5e803d2b22aeec42f'
PREPROCESSING = 'xyz_center_unit_sphere_repeat10000_direct_group64_neighbors32'
POOL_SIZE = 10000


def observed_point_indices(count):
    """Exactly 10000 measured entries; cap densely, repeat only when sparse."""
    if count < 32:
        raise ValueError('at least 32 observed points required')
    return (np.arange(POOL_SIZE)*count//POOL_SIZE if count > POOL_SIZE
            else np.arange(POOL_SIZE) % count)


def runtime_identity():
    root = Path(__file__).resolve().parents[2]/'third_party/pointbert'
    paths = ['encoder.py', 'LICENSE', 'pointnet2_ops_lib/setup.py']
    paths += [str(p.relative_to(root)) for p in sorted((root/'pointnet2_ops_lib/pointnet2_ops').rglob('*')) if p.is_file() and p.suffix in ('.py', '.cpp', '.cu', '.h')]
    return digest({p: sha(root/p) for p in paths})


def prepare_local_pool(points):
    """Model preprocessing only; never modifies the benchmark's raw XYZ data.

    Fixed pool preserves ALL valid local observations, repeating only existing
    points. It makes FPS independent of batch padding/other samples. Native FPS
    receives all 10000 entries directly. Training and evaluation share this path.
    Repeated entries are padding, not additional measured geometry.
    """
    points = np.asarray(points, dtype=np.float32).copy()
    if points.ndim != 2 or points.shape[1] != 3 or not 32 <= len(points) <= POOL_SIZE:
        raise ValueError('Point-BERT needs 32..10000 valid points; do not drop contacts')
    if not np.isfinite(points).all():
        raise ValueError('nonfinite local XYZ')
    points -= points.mean(0)
    radius = np.linalg.norm(points, axis=1).max()
    if not np.isfinite(radius) or radius <= 1e-9:
        raise ValueError('degenerate local point cloud')
    points /= radius
    return np.ascontiguousarray(points[np.arange(POOL_SIZE) % len(points)])


class PointBertBackbone(nn.Module):
    """Official 384-wide, 12-layer encoder; CLS+max-group output is 768-D."""
    def __init__(self, checkpoint=None):
        super().__init__()
        from third_party.pointbert.encoder import PointTransformer
        config = SimpleNamespace(trans_dim=384, depth=12, drop_path_rate=0.,
            cls_dim=40, num_heads=6, group_size=32, num_group=64, encoder_dims=256)
        self.model = PointTransformer(config)
        self.model.cls_head_finetune = nn.Identity()
        if checkpoint is not None:
            from dinov2.tacdino.official_pointbert import representation_state
            if sha(checkpoint) != WEIGHT_SHA256:
                raise ValueError('not the verified official Point-BERT pretrained checkpoint')
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.model.load_state_dict(representation_state(state['base_model'], self.model.state_dict()), strict=True)

    def forward(self, points, return_regions=False):
        if points.ndim != 3 or points.shape[1:] != (POOL_SIZE, 3) or not points.is_cuda:
            raise ValueError('expected preprocessed local XYZ pool on CUDA')
        # Native PointNet++ kernels require fp32. PC root is also fp32 in FSDP.
        with torch.autocast(device_type='cuda', enabled=False):
            # Do not silently reduce the actual encoder input back to 1024.
            selected = points.float().contiguous()
            if return_regions:
                # Expose the official normalized group tokens; same modules,
                # operations and weights as PointTransformer.forward. Keep the
                # vendored official source unchanged. No CLS broadcast shortcut.
                model = self.model
                neighborhood, center = model.group_divider(selected)
                groups = model.reduce_dim(model.encoder(neighborhood))
                x = torch.cat((model.cls_token.expand(len(groups), -1, -1), groups), 1)
                pos = torch.cat((model.cls_pos.expand(len(groups), -1, -1), model.pos_embed(center)), 1)
                return model.norm(model.blocks(x, pos))[:, 1:]
            return self.model(selected)
