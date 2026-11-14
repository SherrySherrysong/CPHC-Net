# CPHC-Net 论文代码 / Paper Modules
CPHC-Net: Robust Student Behavior Recognition in Smart ClassroomsUsing a Convergent Positional and Hyper-Contextual Network

We propose a Convergent Positional and Hyper-Contextual Network (CPHC-Net) for scale-adaptive and occlusion-aware student behavior recognition in smart classrooms.
CPHC-Net adopts D-FINE as baseline and improves its feature representation capability from multi-scale enhancement to high-order contextual modeling. First, a Position-Dominant Adaptive Fusion unit is designed to enable scale-adaptive spatial feature extraction, allowing the backbone to better handle object scale variations in classroom scenes. Then, a Multi-Scale Context Enhancement module is constructed to compensate for high-frequency information loss during feature extraction and provide more informative multi-scale features for subsequent context modeling. Based on the enhanced features, a Hypergraph Contextual Aggregation module is introduced to establish cross-spatial high-order feature relationships. Finally, GateNCS units are employed to refine multi-scale feature fusion and reduce redundant feature channels introduced by global contextual modeling. Experiments on the public classroom behavior datasets SCB03-S and SCB03-U, as well as the self-built SmartClassroom-U dataset, demonstrate that CPHC-Net outperforms mainstream algorithms. Its generalization performance is further verified on a small-object dataset. {\color{red}Moreover, CPHC-Net improves recognition accuracy and inference speed over the D-FINE baseline with slightly fewer parameters.} These results indicate that CPHC-Net provides an effective intelligent recognition approach for student behavior analysis in complex smart classroom environments.

本目录包含 CPHC-Net 的核心模块及网络配置，用于论文实验和完整检测工程集成。

This directory contains the core CPHC-Net modules and network configuration for paper experiments and integration into a complete detection framework.

## 中文

### 文件说明

| 文件 | 说明 |
| --- | --- |
| `PDAF.py` | 位置与尺度感知特征增强，包含 `PDAF` 和 `GateNCS`。 |
| `MCE.py` | 使用不同膨胀率的并行卷积增强上下文信息。 |
| `HCA.py` | 对三层特征进行超图上下文聚合。 |
| `CPHC-Net.yaml` | CPHC-Net 的主干、编码器和解码器配置。 |

### 依赖

- Python 3.10+
- PyTorch
- `einops`
- 完整工程中的 `.utils.get_activation`（供 `MCE.py` 使用）

### 核心模块

- `PDAF`：输入和输出均为单层特征图。`scale_index` 对应尺寸为 `0→160×160`、`1→80×80`、`2→40×40`、`3→20×20`。
- `MCE`：输入 `[B, c1, H, W]`，输出 `[B, c2, H, W]`。
- `HCA`：输入高、中、低三个尺度的特征列表，输出中间尺度特征。
- `GateNCS`：用于编码器中的卷积特征聚合与全局上下文增强。

`CPHC-Net.yaml` 还依赖完整工程提供的 `Conv`、`DownsampleConv`、`FullPAD_Tunnel`、`SCDown`、`Concat` 和 `DFINETransformer`。使用前需将本目录模块加入工程的模块注册或配置解析逻辑。

## English

### Files

| File | Description |
| --- | --- |
| `PDAF.py` | Position- and scale-aware feature enhancement with `PDAF` and `GateNCS`. |
| `MCE.py` | Context enhancement through parallel convolutions with different dilation rates. |
| `HCA.py` | Hypergraph contextual aggregation over three feature levels. |
| `CPHC-Net.yaml` | Backbone, encoder, and decoder configuration for CPHC-Net. |

### Requirements

- Python 3.10+
- PyTorch
- `einops`
- `.utils.get_activation` from the host project for `MCE.py`

### Core Modules

- `PDAF`: processes one feature map. `scale_index` maps to `0→160×160`, `1→80×80`, `2→40×40`, and `3→20×20`.
- `MCE`: input `[B, c1, H, W]`, output `[B, c2, H, W]`.
- `HCA`: consumes high-, middle-, and low-resolution feature maps and returns the fused middle-resolution feature.
- `GateNCS`: performs convolutional feature aggregation and global-context enhancement in the encoder.

`CPHC-Net.yaml` also requires framework modules such as `Conv`, `DownsampleConv`, `FullPAD_Tunnel`, `SCDown`, `Concat`, and `DFINETransformer`. Register the modules in this directory with the host framework before loading the configuration.
