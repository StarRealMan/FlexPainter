# FlexPainter

[Arxiv 2025] FlexPainter: Flexible and Multi-View Consistent Texture Generation

### [Paper](https://arxiv.org/abs/2506.02620) | [Project Page](https://starydy.xyz/FlexPainter) | [Video](https://www.youtube.com/watch?v=AudeQdTifWY)

> FlexPainter: Flexible and Multi-View Consistent Texture Generation <br />
> Dongyu Yan*, Leyi Wu*, Jiantao Lin, Luozhou Wang, Tianshuo Xu, Zhifei Chen, Zhen Yang, Lie Xu, Shunsi Zhang, Yingcong Chen <br />
> Arxiv 2025

<p align="center">
  <img width="100%" src="./docs/static/images/teaser-1.png"/>
</p>

This repository contains code for the paper FlexPainter: Flexible and Multi-View Consistent Texture Generation, an texture generation method that uses flexible, multi-modal inputs.

## Install

```sh
conda create -n flexpainter python==3.10
conda activate flexpainter

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

Run texting with image prompt:

```sh
 python test.py --mesh_path ./demo/cases/sofa/model.obj --image_prompt ./demo/cases/sofa/cond.png
```

Run testing with text prompt:

```sh
 python test.py --mesh_path ./demo/cases/sofa/model.obj --text_prompt "A modern sofa with a blue cushion"
```

Run stylization:

```sh
 python test.py --mesh_path ./demo/cases/sofa/model.obj --image_prompt ./demo/styles/icy.jpg --stylization
```

Additional upsampling can be done using RealESRGan and the checkpoint trained on UV space: ```./ckpts/realesrgan/net_g.pth```.

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
