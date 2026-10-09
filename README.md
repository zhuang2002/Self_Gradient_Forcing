<div align="center">

# 🌀 Self Gradient Forcing

### Native Long-Video Extrapolation

<p>
  <a href='http://zhuang2002.github.io/SelfGradientForcing'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
  <a href="https://arxiv.org/abs/2607.20368"><img src="https://img.shields.io/badge/arXiv-2607.20368-b31b1b.svg"></a> &nbsp;
  <a href="https://huggingface.co/JunhaoZhuang/Self_Gradient_Forcing"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow" alt="Hugging Face"></a> &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="License"></a>
</p>

<p>
  <strong>Junhao Zhuang, Shiyi Zhang, Yuxuan Bian, Yaowei Li, Yawen Luo, Yijun Liu, Weiyang Jin, Songchun Zhang, Xianglong He, Xuying Zhang, Haoran Li, Haoyang Huang, Zeyue Xue, Nan Duan</strong>
</p>

<p>
  Joy Future Academy, JD
</p>

⭐ If Self Gradient Forcing is useful for your research, please consider starring this repository.

</div>

> [!TIP]
> 🚀 Check out our follow-up work **[Self Gradient Forcing Plus (SGF+)](https://zihan-su.github.io/self-gradient-forcing-plus/)**: Decoupling Gradient Flows for Autoregressive Video Generation.

<p align="center">
  <img src="assets/teaser.jpg" width="95%" alt="Self Gradient Forcing teaser">
</p>

## 🔥 News

- **2026-10-07**: 🚀 Introducing **SGF+**, our follow-up work on Self Gradient Forcing! SGF+ decouples context-writing and denoising parameters, improving visual quality and long-horizon consistency and enabling video generation of up to **24 hours** from only **5-second training windows**. [Project Page](https://zihan-su.github.io/self-gradient-forcing-plus/) | [Code](https://github.com/Zihan-Su/Self_Gradient_Forcing_Plus) | [Paper](https://arxiv.org/abs/2610.10429).
- **2026-07-23**: Paper, model checkpoints, inference scripts, and training code are publicly released.

## 🧠 Method Overview

Self Gradient Forcing (SGF) recovers the missing context-gradient path for self-generated causal memory through a bounded two-pass replay, enabling models trained with only a 5-second window to extrapolate to minute-scale videos with stronger identity, layout, and temporal stability.

<p align="center">
  <img src="assets/flowchat.jpg" width="95%" alt="Self Gradient Forcing method overview">
</p>

## 🛠️ Installation

The environment follows the Causal-Forcing setup.

```bash
conda create -n self_gradient_forcing python=3.10 -y
conda activate self_gradient_forcing

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## ⬇️ Download Weights

```bash
bash scripts/download_weights.sh
```

The script uses the Hugging Face CLI command `hf` by default. Set `HF_CLI=huggingface-cli` if your environment still uses the older command name.

It downloads:

- Wan base models to `wan_models/Wan2.1-T2V-1.3B` and `wan_models/Wan2.1-T2V-14B`.
- All [Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) initialization checkpoints under `checkpoints/init/framewise/` and `checkpoints/init/chunkwise/`: `ar_diffusion.pt`, `causal_cd.pt`, and `causal_ode.pt`.
- Released SGF inference checkpoints to `checkpoints/framewise/ar/model.pt` and `checkpoints/chunkwise/ar/model.pt`.
- The training prompt list to `prompts/vidprom_filtered_extended.txt`.

## 🚀 Inference

The default prompt file is `prompts/test_prompt.txt` with 8 prompts. The launcher uses 8 GPUs when at least 8 GPUs are visible; otherwise it falls back to single-GPU serial inference. By default it generates `963` latent frames, which decode to about 240 seconds of video at 16 fps.
The inference script takes the release setting name (`framewise` or `chunkwise`) and selects the matching config and checkpoint automatically:

- framewise config: `configs/self_gradient_forcing_framewise.yaml`
- chunkwise config: `configs/self_gradient_forcing_chunkwise.yaml`

The long-video KV-cache geometry is set in `scripts/infer_self_gradient_forcing.sh`. Framewise defaults to `KV_CACHE_SINK=4`, `KV_CACHE_FIFO_FRAMES=16`, and `KV_CACHE_CURRENT_FRAMES=1`, so the actual `--kv_cache_max_frames` passed to `inference.py` is `4 + 16 + 1 = 21`. Chunkwise defaults to `KV_CACHE_SINK=3`, `KV_CACHE_FIFO_FRAMES=6`, and `KV_CACHE_CURRENT_FRAMES=3`, so `--kv_cache_max_frames` is `12`.

### Framewise

```bash
bash scripts/infer_self_gradient_forcing.sh framewise
```

This uses:

```text
configs/self_gradient_forcing_framewise.yaml
checkpoints/framewise/ar/model.pt
```

### Chunkwise

```bash
bash scripts/infer_self_gradient_forcing.sh chunkwise
```

This uses:

```text
configs/self_gradient_forcing_chunkwise.yaml
checkpoints/chunkwise/ar/model.pt
```

### Custom checkpoint or prompt file

```bash
bash scripts/infer_self_gradient_forcing.sh \
  framewise \
  checkpoints/framewise/ar/model.pt \
  prompts/test_prompt.txt
```

Useful overrides:

```bash
NUM_OUTPUT_FRAMES=963 SEED=42 OUTPUT_ROOT=outputs/demo \
  bash scripts/infer_self_gradient_forcing.sh framewise
```

For trained checkpoints, pass the release setting first and the produced `logs/.../checkpoint_model_*/model.pt` path as the second argument. The script uses EMA weights by default; set `USE_EMA=0` if you explicitly want the non-EMA `generator` weights.

## 🏋️ Training

### Framewise SGF

```bash
bash scripts/train_self_gradient_forcing_framewise.sh
```

Equivalent explicit form:

```bash
bash scripts/train_self_gradient_forcing_framewise.sh \
  configs/self_gradient_forcing_framewise.yaml \
  logs/sgf_framewise
```

### Chunkwise SGF

```bash
bash scripts/train_self_gradient_forcing_chunkwise.sh
```

Equivalent explicit form:

```bash
bash scripts/train_self_gradient_forcing_chunkwise.sh \
  configs/self_gradient_forcing_chunkwise.yaml \
  logs/sgf_chunkwise
```

The launchers accept `[config.yaml] [logdir] [extra train.py args...]`, matching the multi-node launcher convention used by the reference training scripts. They support single-node and multi-node training. For multi-node jobs, run the same command on every node within the gather window. The scripts auto-register nodes through `.rendezvous/` on the shared filesystem and launch static `torchrun` with an IP master address.

Useful overrides:

```bash
GATHER_WINDOW=90 NUM_GPUS=8 MASTER_PORT=29501 ENABLE_WANDB=1 \
  bash scripts/train_self_gradient_forcing_framewise.sh logs/sgf_framewise

NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 NUM_GPUS=8 \
  bash scripts/train_self_gradient_forcing_chunkwise.sh logs/sgf_chunkwise
```

## 🙏 Acknowledgements

This implementation builds on the Wan video model ecosystem and follows the installation conventions of `thu-ml/Causal-Forcing`. We thank the open-source community for the infrastructure that made this release possible.

## 📮 Contact

For questions, please contact Junhao Zhuang at [zhuangjh23@tsinghua.org.cn](mailto:zhuangjh23@tsinghua.org.cn).

## 📜 License

This project is released under the Apache-2.0 license.

### 📜 Citation

```bibtex
@article{zhuang2026self,
  title={Self Gradient Forcing: Native Long Video Extrapolation},
  author={Zhuang, Junhao and Zhang, Shiyi and Bian, Yuxuan and Li, Yaowei and Luo, Yawen and Liu, Yijun and Jin, Weiyang and Zhang, Songchun and He, Xianglong and Zhang, Xuying and others},
  journal={arXiv preprint arXiv:2607.20368},
  year={2026}
}
```
