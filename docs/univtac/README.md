# UniVTAC installation and training

This guide trains a Diffusion Policy from GeoTAC visual and tactile encoders using UniVTAC Isaac 4.5 demonstrations. Run the commands from the Tac-DINO repository root.

## 1. Install the training environment

Use Linux, Python 3.11, and a BF16-capable NVIDIA GPU. Install a CUDA build of PyTorch 2.7.1 and torchvision 0.22.1 that matches your NVIDIA driver. The example uses CUDA 12.6. See the [PyTorch wheel installation instructions](https://pytorch.org/get-started/previous-versions/#v271) for other builds.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt 'huggingface_hub>=0.32,<1'
python -m dinov2.tacdino.univtac_dp.cli --help
```

Offline policy training reads recorded HDF5 files directly. The commands in this guide only need the Python environment above. For GeoTAC pretraining and its CUDA extensions, see the [main README](../../README.md#installation).

## 2. Download the demonstrations

Download the HDF5 files for one task from the [pinned UniVTAC dataset revision](https://huggingface.co/datasets/byml/UniVTAC/tree/e1aee7b0c95543b535e0146b2de3ee1bc6ddaabd/isaac45). Set `TASK` to `lift_can`, `lift_bottle`, `grasp_classify`, or `insert_tube`. The example uses `lift_can`.

```sh
TASK=lift_can
DATA_ROOT=./data/univtac
hf download byml/UniVTAC --repo-type dataset \
  --revision e1aee7b0c95543b535e0146b2de3ee1bc6ddaabd \
  --include "isaac45/${TASK}/hdf5/*.hdf5" --local-dir "$DATA_ROOT"
```

Each task needs 100 files, `0.hdf5` through `99.hdf5`, under `$DATA_ROOT/isaac45/$TASK/hdf5/`. Pass `$DATA_ROOT` to the trainer. It splits the episodes into 80 for training and 20 for validation.

## 3. Set the GeoTAC initialization and run preflight

Set `GEOTAC_CKPT` to a `teacher_checkpoint.pth` export from [GeoTAC pretraining](../../README.md#geotac-pretraining). The policy initializes its visual and tactile backbones from that export and fine-tunes both. The path below is the export produced at step 1,848 with the output layout in the main README. Adjust it to point to your export.

```sh
GEOTAC_CKPT=./runs/geotac/training/eval/training_1847/teacher_checkpoint.pth
RUN_ROOT=./runs/univtac

python -m dinov2.tacdino.univtac_dp.cli preflight --task "$TASK" \
  --data-root "$DATA_ROOT" --geotac-checkpoint "$GEOTAC_CKPT" \
  --output "$RUN_ROOT" --stage initial
```

Preflight checks that all 100 episode files are present and runs a training batch from episode 0 on one GPU. It checks gradient flow, action output, and whether changing the tactile input changes the policy conditioning. The report is written to `$RUN_ROOT/preflight/${TASK}_geotac_joint.json`.

## 4. Train and resume

Run the initial stage to 10,000 optimizer steps. After it completes, extend the same run to 30,000 total steps. Keep the task, data root, initialization weights, output root, and training options consistent across the two commands.

```sh
python -m dinov2.tacdino.univtac_dp.cli train --task "$TASK" \
  --data-root "$DATA_ROOT" --geotac-checkpoint "$GEOTAC_CKPT" \
  --output "$RUN_ROOT" --stage initial

python -m dinov2.tacdino.univtac_dp.cli train --task "$TASK" \
  --data-root "$DATA_ROOT" --geotac-checkpoint "$GEOTAC_CKPT" \
  --output "$RUN_ROOT" --stage final --resume --extend-training
```

To resume an interrupted run, repeat the command for its current stage with `--resume`. Once the final stage has saved progress, `--stage final --resume` is sufficient. The trainer resumes from `latest.pt`.

The default microbatch is 24 per GPU and the effective batch size is 192. If preflight runs out of GPU memory, try a smaller `--micro-batch-size` and use it consistently in training. For multiple GPUs, replace `python -m` in the training commands with `torchrun --standalone --nproc_per_node=4 -m`, adjusting the GPU count for your machine. Choose a microbatch and GPU count whose product divides 192 evenly. Run preflight as a standalone command.

The [policy configuration](../../configs/univtac_geotac.json) records the optimizer settings, camera and tactile streams, and diffusion schedule.

## 5. Find the training outputs

Each task writes to `$RUN_ROOT/$TASK/geotac/`.

| File | Contents |
| --- | --- |
| `latest.pt` | Latest saved model, EMA, and optimizer state for resuming training |
| `best.pt` | Saved state with the lowest validation noise prediction MSE |
| `metrics.jsonl` | Training losses and offline validation metrics |
| `checkpoint_status.json` | Latest saved step |
| `complete.json` | Completion record for the requested stage |

The validation metrics measure performance on held-out demonstrations. Keep generated run files local because they can contain resolved paths from the training machine.
