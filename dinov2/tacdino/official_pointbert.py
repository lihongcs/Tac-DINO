"""Load the representation weights from the official Point-BERT checkpoint."""
import torch

def representation_state(state, expected_keys):
    """Load all learned encoder weights; remove only known pretraining heads."""
    selected = {}
    for key, value in state.items():
        key = key.removeprefix('module.')
        if not key.startswith('transformer_q.'):
            continue  # dVAE tokenizer, EMA teacher, and negative queues aren't this encoder.
        key = key[len('transformer_q.'):]
        if key == 'mask_token' or key.startswith(('lm_head.', 'cls_head.')):
            continue
        selected[key] = value
    missing, extra = set(expected_keys)-set(selected), set(selected)-set(expected_keys)
    if missing or extra:
        raise ValueError(f'Point-BERT representation weights mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
    if any(not torch.isfinite(value).all() for value in selected.values()):
        raise ValueError('nonfinite Point-BERT weights')
    return selected


