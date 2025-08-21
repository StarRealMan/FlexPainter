# FlexPainter

[Arxiv 2025] FlexPainter: Flexible and Multi-View Consistent Texture Generation

### [Paper](https://arxiv.org/abs/2506.02620) | [Project Page](https://starydy.xyz/FlexPainter) | [Video](https://www.youtube.com/watch?v=AudeQdTifWY) | [Demo](https://envision-research.hkust-gz.edu.cn/flexpainter)

> FlexPainter: Flexible and Multi-View Consistent Texture Generation <br />
> Dongyu Yan*, Leyi Wu*, Jiantao Lin, Luozhou Wang, Tianshuo Xu, Zhifei Chen, Zhen Yang, Lie Xu, Shunsi Zhang, Yingcong Chen <br />
> Arxiv 2025

<p align="center">
  <img width="100%" src="./docs/static/images/teaser-1.png"/>
</p>

This repository contains code for the paper FlexPainter: Flexible and Multi-View Consistent Texture Generation, an texture generation method that uses flexible, multi-modal inputs.

## Install

```sh
conda create -n flexpainter python==3.10 -y
conda activate flexpainter

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
conda install google-sparsehash -c bioconda -y
pip install -r requirements.txt
```

## Usage

Download checkpoints using huggingface-cli:

```sh
  huggingface-cli login
  huggingface-cli download StarYDY/FlexPainter --local-dir ./ckpts
```

Generate Blank Embeddings for CFG:

```sh
  python gen_blank.py
```

Run texting with image prompt:

```sh
 python test.py \
    --mesh_path ./demo/axe/13b98c410feb42dc940f0b40b96af9c0.obj \
    --image_prompt ./demo/axe/cond.jpg
```

Run testing with text prompt:

```sh
 python test.py \
    --mesh_path ./demo/axe/13b98c410feb42dc940f0b40b96af9c0.obj \
    --prompt "An axe with wooden handle and metal blade"
```

Run joint generation with a lower image strength:

```sh
 python test.py \
    --mesh_path ./demo/axe/13b98c410feb42dc940f0b40b96af9c0.obj \
    --image_prompt ./demo/axe/cond.jpg \
    --prompt "An axe with wooden handle and metal blade" \
    --image_strength 0.1
```

Run stylization:

```sh
 python test.py \
    --mesh_path ./demo/axe/13b98c410feb42dc940f0b40b96af9c0.obj \
    --image_prompt ./demo/styles/icy.jpg \
    --stylize
```

Additional upsampling can be done using RealESRGan and the checkpoint trained on UV space: ```./ckpts/realesrgan/net_g.pth```:

```sh
  # clone the original Real-ESRGAN repo
  git clone https://github.com/xinntao/Real-ESRGAN.git
  cd Real-ESRGAN
  # install
  pip install basicsr
  pip install facexlib
  pip install gfpgan
  pip install -r requirements.txt
  python setup.py develop
  # inference
  cd ./Real-ESRGAN
  python inference_realesrgan.py --model_path ../ckpts/realesrgan/net_g.pth -i <your-input-texture-image>
```

## Acknowledgements

Our work builds upon these excellent repositories:

- [TEXGen](https://github.com/CVMI-Lab/TEXGen)
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{yan2024flexipainter,
  title={FlexPainter: Flexible and Multi-View Consistent Texture Generation},
  author={Dongyu Yan, Leyi Wu, Jiantao Lin, Luozhou Wang, Tianshuo Xu, Zhifei Chen, Zhen Yang, Lie Xu, Shunsi Zhang, Yingcong Chen},
  journal={arXiv preprint arXiv:2506.02620},
  year={2025}
}
```
