# Representative Guidance (RepG)

Official code for the ICLR 2025 paper
**[Representative Guidance: Diffusion Model Sampling with Coherence](https://openreview.net/forum?id=gWgaypDBs8)**.

Representative Guidance (RepG) reformulates diffusion sampling to move along a *coherent* direction
towards a **representative target** built from **self-supervised representations** (e.g. MoCo v2).
Unlike classic classifier guidance — whose discriminative features highlight only a narrow set of
class-specific cues — RepG treats sampling as a downstream task that refines image detail and corrects
generation errors. It improves vanilla diffusion sampling and, combined with classifier-free guidance,
surpasses state-of-the-art benchmarks.

This is a **minimal, self-contained release** of the main ImageNet sampling pipeline. It builds on
OpenAI's [guided-diffusion](https://github.com/openai/guided-diffusion) (ADM) and runs on a standard
PyTorch stack — no proprietary cluster dependencies.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `off_guided_diffusion/` | Diffusion backbone (UNet + Gaussian diffusion) used during RepG sampling. |
| `guided_diffusion/` | Shared diffusion utilities and self-supervised diffusion features (`gaussian_diffusion_ss`). |
| `scripts_gdiff/selfsup/support/` | Builds / loads the off-the-shelf self-supervised encoders (MoCo v2, SimSiam). |
| `script_odiff/` | The RepG sampler (`mocov2_meanclose_contrastive_outclass_sup_instance_sample_transform.py`). |
| `evaluations/` | FID / Precision / Recall evaluator and the classifier feature models. |

---

## 1. Setup

```bash
git clone -b main https://github.com/dungdinhanh/selfsup-guidance.git
cd selfsup-guidance

conda create -n repg python=3.9 -y
conda activate repg

pip install -e .
pip install -r requirements.txt

# PyTorch matching your CUDA (example: CUDA 11.1)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# needed only for the FID / precision / recall evaluator
pip install tensorflow-gpu==2.9.2 scipy
```

---

## 2. Assets to download

RepG guides a **pretrained diffusion model** with a **pretrained self-supervised encoder** and a set of
**precomputed reference representations**. Place them as follows:

**a) Diffusion checkpoint → `models/`** (from OpenAI guided-diffusion):

```
models/256x256_diffusion.pt      # class-conditional ADM, ImageNet 256
```
Other resolutions (64/128/512) are available on the
[guided-diffusion model page](https://github.com/openai/guided-diffusion#download-pre-trained-models).

**b) Self-supervised encoder → `eval_models/`** (official release):

```
eval_models/moco_v2_800ep_pretrain.pth.tar   # MoCo v2 (facebookresearch/moco)
```

**c) Reference representations → `eval_models/imn256_mocov2/`**

Self-supervised features of the ImageNet reference set that RepG builds its representative target from:

```
eval_models/imn256_mocov2/reps3.npz
```
The sampler reads this via `--features`. Folders follow the pattern `imn{64,128,256,512}_mocov2`.

**d) Evaluation stats → `reference/`** (from guided-diffusion evaluations):

```
reference/VIRTUAL_imagenet256_labeled.npz
```

---

## 3. Sampling with Representative Guidance

The sampler uses `torch.distributed`; for a single process, set the launch env vars shown below.

```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 \
 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
 --use_scale_shift_norm True"

SAMPLE_FLAGS="--batch_size 50 --num_samples 50000 --timestep_respacing 250"

WORLD_SIZE=1 RANK=0 MASTER_IP=127.0.0.1 MASTER_PORT=29510 \
python script_odiff/mocov2_meanclose_contrastive_outclass_sup_instance_sample_transform.py \
  $MODEL_FLAGS $SAMPLE_FLAGS \
  --model_path models/256x256_diffusion.pt \
  --classifier_type mocov2 \
  --classifier_scale 18.0 \
  --joint_temperature 0.5 \
  --k_closest 10 \
  --features eval_models/imn256_mocov2/reps3.npz \
  --logdir runs/repg_imn256_mocov2
```

Key RepG flags:

| Flag | Meaning |
|------|---------|
| `--classifier_type` | Off-the-shelf self-supervised encoder (`mocov2`). |
| `--classifier_scale` | Guidance strength. |
| `--features` | Precomputed reference representations (`reps*.npz`). |
| `--k_closest` | Size of the representative (k-closest) target set. |
| `--joint_temperature`, `--margin_temperature_discount`, `--gamma_factor` | Temperature / margin controls for the representative target. |
| `--timestep_respacing` | Sampling steps (`250`, or `ddim25` with `--use_ddim True`). |

Samples and an `.npz` batch are written under `--logdir`.

---

## 4. Evaluation (FID / Precision / Recall)

```bash
python evaluations/evaluator_tolog.py \
  reference/VIRTUAL_imagenet256_labeled.npz \
  runs/repg_imn256_mocov2/reference/samples_50000x256x256x3.npz
```

---

## Citation

```bibtex
@inproceedings{repg2025,
  title     = {Representative Guidance: Diffusion Model Sampling with Coherence},
  author    = {Dinh, Anh-Dung and others},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=gWgaypDBs8}
}
```
<!-- Please confirm the author list matches the camera-ready version. -->

## Acknowledgements

Built on [openai/guided-diffusion](https://github.com/openai/guided-diffusion) and
[openai/improved-diffusion](https://github.com/openai/improved-diffusion), with self-supervised encoders
from [MoCo](https://github.com/facebookresearch/moco).
