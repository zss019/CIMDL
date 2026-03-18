# CIMDL

## Introduction
Source code of paper: "**Online Opinion Conflict Interaction Recognition Based on Dependent Multi-Task Deep Learning**". 
Accepted by **ICWSM** 2026 (*The 20th International AAAI Conference on Web and Social Media*, Los Angeles).

## Online Appendix
This is the online appendix https://github.com/zss019/CIMDL/blob/main/online_appendix.pdf for the paper, which includes “Appendix A. Prompts for LLM in the Experiment” and “Appendix B. Case Analysis”. These appendices constitute the supplementary material​ for the experimental analysis in the paper.

## Dataset

`ScienceNet_dataset` and `zhihu_dataset` are two benchmark datasets we have constructed, which have been divided into train/dev/test sets. We crawled users' opinion interaction records and manually annotated them.

Zhihu dataset consists of comments and corresponding replies in Q&A thread on selected controversial topics. These interactions were annotated as “Support” and “Conflict”.
ScienceNet dataset includes comments and replies on the web-blogs about scientific or academic topics. In addition to “Support” and “Conflict” , we have also annotated “Neutral” for some smooth interaction.

## Environments

- Python 3.10

- PyTorch 2.1.0

- RTX 3090 GPU 

- CUDA 12.1

  Recommended to use GPU for training (such as Google Colab, AutoDL, Aliyun, etc.)

## File Structure 

The `CIMDL.py` file located in the `models` folder is the model file, which contains the definition scheme of the model. 

The `run.py`, `train_eval.py`, and `utils.py` files are responsible for the project's operation, training, and processing respectively. 

Additionally, the pre-trained model we used was downloaded from: [https://huggingface.co/hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) and [https://huggingface.co/DMetaSoul/sbert-chinese-qmc-domain-v1.](https://huggingface.co/DMetaSoul/sbert-chinese-qmc-domain-v1)

Part of code (particularly data processing and loading) references the [Multi-architecture Open-source Chinese Text Classification] (https://github.com/649453932/Chinese-Text-Classification-Pytorch) project. Readers interested in our paper can refer to that project's documentation for details..

## Running
```
python run.py --model CIMDL
```
