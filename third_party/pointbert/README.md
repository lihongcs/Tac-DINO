# Official Point-BERT encoder dependency snapshot

`upstream/` contains unchanged source files from
https://github.com/Julie-tang00/Point-BERT at
`ce5a6e1f35a0d111de12c3a5e803d2b22aeec42f` (MIT; see LICENSE).

Only the encoder definitions are used. DiscreteVAE, its KNN-CUDA/Chamfer
losses, downstream classifier, EMA tokenizer and pretraining queue are not
used by PointTransformer representation inference or fine-tuning.

`pointnet2_ops_lib` is the dependency named by the official Point-BERT install
instructions, from https://github.com/erikwijmans/Pointnet2_PyTorch (UNLICENSE).
Its native sampling/grouping kernels are unchanged. The setup script alone
respects TORCH_CUDA_ARCH_LIST instead of overriding it with obsolete GPUs.

Official pretrained weights (stored outside Git):
https://cloud.tsinghua.edu.cn/f/202b29805eea45d7be92/?dl=1
SHA256: `4d595a2d4a4d5755d2b10104634f851e2878bc07dc7fc26c0e9f99f162de504f`.
