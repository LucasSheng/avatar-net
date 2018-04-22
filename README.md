# Avatar-Net: Multi-scale Zero-shot Style Transfer by Feature Decoration

[teaser]: ./figures/teaser.png
![teaser]
<p align = 'center'>
Exemplar stylized results by the proposed Avatar-Net, which faithfully transfers the lena image by arbitrary style.
</p>

## Overview

Zero-shot artistic style transfer is an important image synthesis problem aiming at transferring arbitrary style into content images. However, the trade-off between the generalization and efficiency in existing methods impedes a high quality zero-shot style transfer in real-time. In this repository, we resolve this dilemma and propose an efficient yet effective Avatar-Net that enables visually plausible multi-scale transfer for arbitrary style.

The key ingredient of our method is a __style decorator__ that makes up the content features by semantically aligned style features from an arbitrary style image, which does not only holistically match their feature distributions but also preserve detailed style patterns in the decorated features.

[style_decorator]: ./figures/style_decorator.png
![style_decorator]
<p align = 'center'>
Comparison of feature distribution transformation by different feature transfer modules. (a) <a href="https://arxiv.org/abs/1703.06868">Adaptive Instance Normalization</a>, (b) <a href="https://arxiv.org/abs/1705.08086">Whitening and Coloring Transform</a>, (c) <a href="https://arxiv.org/abs/1612.04337">Style-Swap</a>, and (d) the proposed style decorator.
</p>

By embedding this module into an image reconstruction network that fuses multi-scale style abstractions, the Avatar-Net renders multi-scale stylization for any style image in one feed-forward pass. 

[network]: ./figures/network_architecture_with_comparison.png
![network]
<p align = 'center'>
(a) Stylization comparison by autoencoder and style-augmented hourglass network. (b) The network architecture of the proposed method.
</p>

## Results

We demonstrate the state-of-the-art effectiveness and efficiency of the proposed method in generating high-quality stylized images, with a series of successive applications including multiple style integration, video stylization and etc.

[image_results]: ./figures/image_results.png
![image_results]

### Comparison with Prior Arts

<p align='center'><img src="figures/closed_ups.png" width="640"></p>

- The result by Avatar-Net receives concrete multi-scale style patterns (e.g. color distribution, brush strokes and circular patterns in _candy_ image).

- WCT distorts the brush strokes and circular patterns. AdaIN cannot even keep the color distribution, while style-swap fails in this example.

#### Execution Efficiency
|Method| Gatys et. al. | AdaIN | WCT | Style-Swap | __Avatar-Net__ |
| :---:  | :---:       | :---: | :---: | :---:    | :---:         |
| __256x256 (sec)__ | 12.18 | 0.053 | 0.62 | 0.064 | __0.071__ |
| __512x512 (sec)__ | 43.25 | 0.11 | 0.93 | 0.23 | __0.28__ |

- Avatar-Net has a comparable executive time as AdaIN and GPU-accelerated Style-Swap, and is much faster than WCT and the optimization-based style transfer by Gatys _et. al._.

- The reference methods and the proposed Avatar-Net are implemented on a same TensorFlow platform with a same VGG network as the backbone.

### Applications
#### Multi-style Interpolation
[style_interpolation]: ./figures/style_interpolation.png
![style_interpolation]

#### Content and Style Trade-off
[trade_off]: ./figures/trade_off.png
![trade_off]

#### Video Stylization (click the image or [the Youtube link](https://youtu.be/amaeqbw6TeA))
[video_snapshot]: ./figures/snapshot.png
[![video_snapshot]](https://youtu.be/amaeqbw6TeA)

## Code

Please refer to the [GitHub repository]() for more details. 

## Publication

Lu Sheng, Ziyi Lin, Jing Shao and Xiaogang Wang, "Avatar-Net: Multi-scale Zero-shot Style Transfer by Feature Decoration", in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.  [[PDF]()]  [[Arxiv]()]

```
@inproceedings{sheng2018avatar,
    Title = {Avatar-Net: Multi-scale Zero-shot Style Transfer by Feature Decoration},
    author = {Sheng, Lu and Lin, Ziyi and Shao, Jing and Wang, Xiaogang},
    Booktitle = {Computer Vision and Pattern Recognition (CVPR), 2018 IEEE Conference on},
    pages={1--9},
    year={2018}
}
