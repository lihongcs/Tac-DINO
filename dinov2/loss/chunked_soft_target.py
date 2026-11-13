"""Exact first-order soft-target log probability with bounded FP32 temporaries.

The teacher is detached, as in DINO/iBOT. Keep original logits, not a second
full-sized log-softmax activation; recompute small row blocks in backward.
"""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


class _ChunkedLogProbability(torch.autograd.Function):
    @staticmethod
    def forward(ctx, teacher, student, temperature, chunk_rows):
        if teacher.requires_grad or teacher.shape != student.shape or student.ndim != 2:
            raise ValueError("chunked CE requires matching 2D logits and detached teacher")
        if temperature <= 0 or chunk_rows < 1:
            raise ValueError("positive temperature and chunk size required")
        ctx.save_for_backward(teacher, student)
        ctx.temperature, ctx.chunk_rows = temperature, chunk_rows
        output = torch.empty(student.shape[0], device=student.device, dtype=torch.float32)
        for start in range(0, student.shape[0], chunk_rows):
            sl = slice(start, start + chunk_rows)
            logp = F.log_softmax(student[sl].float() / temperature, dim=-1)
            output[sl] = (teacher[sl].float() * logp).sum(-1)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        teacher, student = ctx.saved_tensors
        gradient = torch.empty_like(student)
        for start in range(0, student.shape[0], ctx.chunk_rows):
            sl = slice(start, start + ctx.chunk_rows)
            target = teacher[sl].float()
            p = F.softmax(student[sl].float() / ctx.temperature, dim=-1)
            # Do not assume perfect unit teacher mass after distributed Sinkhorn.
            g = (target - p * target.sum(-1, keepdim=True)) / ctx.temperature
            gradient[sl] = g * grad_output[sl, None]
        return None, gradient, None, None


def chunked_log_probability(teacher, student, temperature, chunk_rows=128):
    return _ChunkedLogProbability.apply(teacher, student, float(temperature), int(chunk_rows))
