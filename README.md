# CIMDL

## Introduction
Source code of paper: "Online Opinion Conflict Interaction Recognition Based on Dependent Multi-Task Deep Learning".

Part of code (particularly data processing and loading, as well as model running) is based on the [Multi-architecture Open-source Chinese Text Classification] (https://github.com/649453932/Chinese-Text-Classification-Pytorch) project and has undergone significant modifications and adjustments.

## Dataset

`ScienceNet_dataset` and `zhihu_dataset` are two benchmark datasets we have constructed, which have been divided into train/dev/test sets.

We crawled users' opinion interaction records and manually annotated them.

## Environments

- Python 3.10

- PyTorch 2.1.0

- RTX 3090 GPU 

- CUDA 12.1

  Recommended to use GPU for training (such as Google Colab, Kaggle, AutoDL, Aliyun, etc.)

## File Structure 

The `CIMDL.py` file located in the `models` folder is the model file, which contains the definition scheme of the model. 

The `run.py`, `train_eval.py`, and `utils.py` files are responsible for the project's operation, training, and processing respectively. 

Additionally, the pre-trained model we used was downloaded from: [https://huggingface.co/hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) and [https://huggingface.co/DMetaSoul/sbert-chinese-qmc-domain-v1.](https://huggingface.co/DMetaSoul/sbert-chinese-qmc-domain-v1)

## Running
```
python run.py --model CIMDL
```
