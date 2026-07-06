# Representative Guidance (RepG)

Official code for the ICLR 2025 paper
**[Representative Guidance: Diffusion Model Sampling with Coherence](https://openreview.net/forum?id=gWgaypDBs8)**.

Representative Guidance (RepG) reformulates diffusion sampling to move along a *coherent* direction
towards a **representative target** built from **self-supervised representations** (e.g. MoCo v2 / v3,
SimSiam, BYOL). Unlike classic classifier guidance — whose discriminative features highlight only a
narrow set of class-specific cues — RepG treats sampling as a downstream task that refines image
detail and corrects generation errors. It improves vanilla diffusion sampling and, when combined with
classifier-free guidance, surpasses state-of-the-art benchmarks.

This codebase builds on OpenAI's
[guided-diffusion](https://github.com/openai/guided-diffusion) (ADM) and
[GLIDE](https://github.com/openai/glide-text2im).

---

## Repository layout

| Path | Purpose |
|------|---------|
| `off_guided_diffusion/`, `guided_diffusion/`, `improved_diffusion/` | Diffusion model + Gaussian diffusion (ADM/IDDPM). |
| `glide_text2im/` | GLIDE text-to-image backbone (incl. `CLIPModelRep` representative guidance). |
| `scripts_gdiff/selfsup/` | Training / fine-tuning of self-supervised guidance classifiers. |
| `script_odiff/` | **Off-the-shelf** self-supervised guided sampling (main RepG samplers). |
| `scripts_glide/` | GLIDE sampling with representative guidance. |
| `evaluations/` | FID / Precision / Recall evaluator. |
| `eval_models/` | Self-supervised encoder checkpoints + precomputed reference representations. |
| `models/` | Pretrained ADM / GLIDE diffusion checkpoints. |
| `reference/` | Reference statistics (`VIRTUAL_imagenet*_labeled.npz`) for evaluation. |
| `bash_scripts/`, `bash_scripts_hfai/`, `bash_iclr/` | Ready-to-run example scripts for every experiment in the paper. |

---

## 1. Setup

```bash
git clone https://github.com/dungdinhanh/selfsup-guidance.git
cd selfsup-guidance

# create an environment (Python 3.8–3.10 recommended)
conda create -n repg python=3.9 -y
conda activate repg

# install this package + Python deps
pip install -e .
pip install -r requirements.txt

# PyTorch (pick the build matching your CUDA); example for CUDA 11.1:
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

# TensorFlow + scikit-learn are needed for the FID/precision/recall evaluator
pip install tensorflow-gpu==2.9.2 scikit-learn
```

Convenience installers used on our clusters are also provided:
`install_requirements.sh` (local/CUDA 11.1) and `install_requirements_nci2.sh` (NCI/CUDA 10.2).

Multi-GPU runs use `torch.distributed`; set `NCCL_P2P_DISABLE=1` if your nodes lack P2P.

---

## 2. Assets to download

RepG guides a **pretrained diffusion model** with a **pretrained self-supervised encoder** and a set
of **precomputed reference representations**. Place them as follows:

**a) Diffusion checkpoints → `models/`** (from OpenAI guided-diffusion):

```
models/256x256_diffusion.pt          # class-conditional ADM, ImageNet 256
models/64x64_diffusion.pt            # ImageNet 64
models/128x128_diffusion.pt          # ImageNet 128
models/512x512_diffusion.pt          # ImageNet 512
```
See the [guided-diffusion model page](https://github.com/openai/guided-diffusion#download-pre-trained-models)
for links to every checkpoint (64/128/256/512, upsamplers, LSUN).

**b) Self-supervised encoders → `eval_models/`** (official releases):

```
eval_models/moco_v2_800ep_pretrain.pth.tar   # MoCo v2  (facebookresearch/moco)
eval_models/simsiam_0099.pth.tar             # SimSiam  (facebookresearch/simsiam)
```

**c) Reference representations → `eval_models/<dataset>_<encoder>/`**

These are self-supervised features of the ImageNet reference set plus the per-class
*representative* (k-closest) sets that RepG guides towards, e.g.:

```
eval_models/imn256_mocov2/reps3.npz                       # reference features
eval_models/imn256_mocov2/reps3_mean_sup_closest10_set.npz # k=10 representative set
```
The samplers read this via the `--features` flag. Encoder/dataset folders follow the pattern
`imn{64,128,256,512}_{mocov2,mocov3,simsiam}`.

**d) Evaluation stats → `reference/`** (from guided-diffusion evaluations):

```
reference/VIRTUAL_imagenet256_labeled.npz
```

---

## 3. Sampling with Representative Guidance

**Off-the-shelf self-supervised guidance** (ImageNet 256, MoCo v2). This is the core RepG sampler:

```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 \
 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 \
 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True \
 --use_scale_shift_norm True"

SAMPLE_FLAGS="--batch_size 50 --num_samples 50000 --timestep_respacing 250"

python script_odiff/mocov2_meanclose_sup_sample_transform.py $MODEL_FLAGS $SAMPLE_FLAGS \
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
| `--classifier_type` | Self-supervised encoder: `mocov2`, `mocov3`, `simsiam`, or `resnet50/101`. |
| `--classifier_scale` | Guidance strength. |
| `--features` | Precomputed reference representations (`reps*.npz`). |
| `--k_closest` | Size of the representative (k-closest) target set. |
| `--joint_temperature`, `--margin_temperature_discount`, `--gamma_factor` | Temperature / margin controls for the representative target. |
| `--timestep_respacing` | Sampling steps (`250`, or `ddim25` with `--use_ddim True`). |

**Combined with classifier-free guidance** — see
`bash_scripts/sample_offtheshelf_clsfree/` (e.g. `im256_contrastive/im256_clsfree_contrastive2.sh`).

**GLIDE text-to-image** representative guidance — see `scripts_glide/` and `bash_scripts/sample_offtheshelf_glide/`.

The `bash_scripts/`, `bash_scripts_hfai/` and `bash_iclr/` folders contain the exact commands and
hyperparameters for every configuration reported in the paper — they are the recommended starting point.

---

## 4. Evaluation (FID / Precision / Recall)

```bash
python evaluations/evaluator_tolog.py \
  reference/VIRTUAL_imagenet256_labeled.npz \
  runs/repg_imn256_mocov2/reference/samples_50000x256x256x3.npz
```

---

## 5. Training self-supervised guidance classifiers (optional)

To fine-tune a distance-aware self-supervised classifier for guidance (instead of using an
off-the-shelf encoder):

```bash
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 64 --lr 6e-4 \
 --save_interval 10000 --weight_decay 0.2 --pretrained_cls simsiam"
CLASSIFIER_FLAGS="--image_size 256 --dim 2048 --pred_dim 512"

python scripts_gdiff/selfsup/classifier_train_selfsup_simsiam_samplercontrol_negative_ft.py \
  --data_dir /path/to/imagenet \
  --logdir runs/selfsup_training/psimsiam300k_IM256 \
  $TRAIN_FLAGS $CLASSIFIER_FLAGS
```

See `bash_scripts/train_sscls/` and `bash_scripts/train_sscls_pretrained/` for full examples.

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

Built on [openai/guided-diffusion](https://github.com/openai/guided-diffusion),
[openai/improved-diffusion](https://github.com/openai/improved-diffusion),
[openai/glide-text2im](https://github.com/openai/glide-text2im), and the self-supervised encoders from
[MoCo](https://github.com/facebookresearch/moco) and [SimSiam](https://github.com/facebookresearch/simsiam).
