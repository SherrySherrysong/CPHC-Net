# CPHC-Net

Code for **CPHC-Net: Robust Student Behavior Recognition in Smart Classrooms Using a Convergent Positional and Hyper-Contextual Network**.

We propose a Convergent Positional and Hyper-Contextual Network (CPHC-Net) for scale-adaptive and occlusion-aware student behavior recognition in smart classrooms. CPHC-Net adopts D-FINE as baseline and improves its feature representation capability from multi-scale enhancement to high-order contextual modeling. First, a Position-Dominant Adaptive Fusion unit is designed to enable scale-adaptive spatial feature extraction, allowing the backbone to better handle object scale variations in classroom scenes. Then, a Multi-Scale Context Enhancement module is constructed to compensate for high-frequency information loss during feature extraction and provide more informative multi-scale features for subsequent context modeling. Based on the enhanced features, a Hypergraph Contextual Aggregation module is introduced to establish cross-spatial high-order feature relationships. Finally, GateNCS units are employed to refine multi-scale feature fusion and reduce redundant feature channels introduced by global contextual modeling. Experiments on the public classroom behavior datasets SCB03-S and SCB03-U, as well as the self-built SmartClassroom-U dataset, demonstrate that CPHC-Net outperforms mainstream algorithms. Its generalization performance is further verified on a small-object dataset. Moreover, CPHC-Net improves recognition accuracy and inference speed over the D-FINE baseline with slightly fewer parameters. These results indicate that CPHC-Net provides an effective intelligent recognition approach for student behavior analysis in complex smart classroom environments.

## Core modules

| Component | Role |
| --- | --- |
| [PDAF](PDAF.py) | Position-Dominant Adaptive Fusion uses scale-specific positional embeddings at feature-map sizes of 160, 80, 40, and 20 pixels per side. |
| [MCE](MCE.py) | Multi-Scale Context Enhancement enriches each feature level with parallel convolutions at different dilation rates. |
| [HCA](HCA.py) | Hypergraph Contextual Aggregation combines three feature levels and returns a context-enriched feature at the middle resolution. |
| [GateNCS](PDAF.py) | Refines the fused encoder features through convolutional aggregation and global-context recalibration. |

## Installation

The paper used Python 3.10, PyTorch 2.2.2, and an NVIDIA GeForce RTX 4090. Install PyTorch 2.2.2 with its matching torchvision 0.17.2 build for your system. See the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v222).

```bash
git clone https://github.com/SherrySherrysong/CPHC-Net.git
cd CPHC-Net
python -m pip install -r requirements.txt
```

Run the following commands from the repository root.

## Dataset preparation

| Dataset | Classes | Configuration |
| --- | ---: | --- |
| SCB03-S, the school-classroom subset of SCB-Dataset3 | 3 | [`configs/scb03-s.yml`](configs/scb03-s.yml) |
| SCB03-U, the university-classroom subset of SCB-Dataset3 | 6 | [`configs/scb03-u.yml`](configs/scb03-u.yml) |
| SmartClassroom-U, collected from university classrooms | 6 | [`configs/smartclassroom-u.yml`](configs/smartclassroom-u.yml) |
| VisDrone2019-DET, for supplementary small-object evaluation | 10 | [`configs/visdrone2019.yml`](configs/visdrone2019.yml) |

Download the public datasets from [SCB-Dataset](https://github.com/Whiffe/SCB-dataset) and [VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset). Convert native VisDrone annotations to COCO detection JSON before using its config. The SmartClassroom-U config requires authorized access to the dataset. The paper used an 80/20 split for SmartClassroom-U.

Set `img_folder` and `ann_file` for `train_dataloader.dataset` and `val_dataloader.dataset` in the chosen YAML. Use COCO detection JSON with contiguous category IDs starting at `0` and the same category mapping in both splits. For a separate test split, point `val_dataloader.dataset` to its images and JSON when evaluating.

For YOLO-format data, keep the intended split and place `images/`, `labels/`, and a `classes.txt` listing one class per line in label-ID order under each split directory. The converter treats images without label files as background examples. For example:

```bash
python tools/yolo2coco.py data/scb03-s/train data/scb03-s/train/annotations/train.json
python tools/yolo2coco.py data/scb03-s/val data/scb03-s/val/annotations/val.json
```

## Training and evaluation

[`configs/cphc_net.yml`](configs/cphc_net.yml) sets the shared training recipe. It starts from randomly initialized weights and uses 640 × 640 inputs, a batch size of 4, and 300 epochs. AdamW uses learning rates of `8e-4` overall and `4e-4` for the backbone, with weight decay `1e-3` and zero decay for normalization parameters and biases. The staged augmentation schedule uses Mosaic and Mixup with probability `0.5` during their active stages. The paper used seeds `0`, `10`, and `20`.

Train on SCB03-S. Substitute the corresponding dataset config for other datasets and use a separate output directory for each seed.

```bash
python train.py -c configs/scb03-s.yml --device cuda:0 --seed 0 --output-dir outputs/scb03-s/seed0
```

Evaluate a checkpoint produced by training. `best_stg1.pth` is the best checkpoint before the final training stage. Use `best_stg2.pth` if training produced it.

```bash
python train.py -c configs/scb03-s.yml --device cuda:0 --test-only -r outputs/scb03-s/seed0/best_stg2.pth --output-dir outputs/scb03-s/eval
```

Evaluation prints COCO bounding-box AP, AP50, and AP75 and writes `pred.json` and `eval.pth` to the evaluation directory.

## Model Configuration

[CPHC-Net.yaml](CPHC-Net.yaml) defines the backbone, encoder, and D-FINE decoder for the 192-channel model. PDAF and GateNCS are implemented in [PDAF.py](PDAF.py), MCE in [MCE.py](MCE.py), and HCA in [HCA.py](HCA.py).

[engine/extre_module/tasks.py](engine/extre_module/tasks.py) resolves the layer names and assembles the model. The shared training config [configs/cphc_net.yml](configs/cphc_net.yml) selects the architecture through `DEIM_MG.yaml_path`. Keep the input size at 640 × 640 to match the PDAF positional embeddings.

## Acknowledgements

The training code draws on [DEIM](https://github.com/ShihuaHuang95/DEIM), [D-FINE](https://github.com/Peterande/D-FINE), and [RT-DETR](https://github.com/lyuwenyu/RT-DETR).
