# Rethinking Model Prototyping through the MedMNIST+ Dataset Collection
This repository contains the official codebase for the paper titled "Rethinking Model Prototyping through the MedMNIST+ Dataset Collection".

## Overview 🧠
Deep learning (DL) has witnessed remarkable advancements across diverse domains, including image classification and natural language processing. These strides have been driven by the development of sophisticated models and advanced training methodologies, such as self-supervised contrastive methods, which offer cost-effective labeling strategies. Despite the considerable progress and DL models achieving performance levels comparable to experts in medical tasks, their integration into daily clinical practice has been comparatively slow.  One major obstacle hindering this adoption is the scarcity of appropriate datasets in medical imaging, characterized by limited sample sizes and heterogeneous image acquisition conditions. Additionally, there's a concerning trend in DL research prioritizing incremental performance improvements on influential benchmarks over addressing clinically relevant needs. This trend, partially fueled by academic incentives favoring quantity over relevance, leads to increased computational requirements without necessarily improving real-world applicability. The limitations of scaling alone, evidenced by larger models experiencing challenges in trustworthiness or task-specific performance, further impede DL's utility in clinical environments.

Hence, the paper underscores the necessity for qualitative enhancements alongside quantitative scaling in DL research, particularly within real-world medical contexts. It advocates for the creation of larger and more diverse datasets and benchmarks, incorporating additional inductive biases and fostering the development of more sophisticated approaches. In this context, the paper introduces a comprehensive benchmark for the [MedMNIST+ database](https://zenodo.org/records/10519652) to re-evaluate traditional convolutional neural networks (CNNs), Transformer-based architectures as well as training schemes for medical image classification. The evaluation highlights the potential of computationally efficient training schemes while reaffirming the competitiveness of convolutional models compared to Vision Transformer-based architectures. Furthermore, the standardized evaluation framework aims to enhance transparency, reproducibility, and comparability in medical image classification research.

Subsequent sections outline the paper's [key contributions](#key-contributions-), showcase the [obtained results](#results-), and offer instructions on [accessing and utilizing the accompanying codebase](#getting-started-) to replicate the findings and train or evaluate your own models.

## Key Contributions 🔑
- Presentation of a solid baseline performance for MedMNIST+ and a standardized evaluation framework for assessing future model performance in medical image classification.
- Identification of systematic strengths and weaknesses inherent in traditional models within the context of medical image classification.
- Reevaluation of prevalent assumptions with respect to model design, training schemes and input resolution requirements.
- Formulation of recommendations and take-aways for model development and deployment.

## Results 📊
First and foremost, the paper presents the very first solid baseline evaluation for the MedMNIST+ dataset collection. Furthermore, the findings suggest that computationally efficient training schemes and modern foundation models hold promise in bridging the gap between expensive end-to-end training and more resource-refined approaches. Contrary to prevailing assumptions, the authors observe that higher resolutions may not consistently improve performance beyond a certain threshold, advocating for the use of lower resolutions, particularly in prototyping stages, to expedite processing. Additionally, the analysis reaffirms the competitiveness of convolutional models compared to Vision Transformer (ViT)-based architectures, emphasizing the importance of understanding the intrinsic capabilities of different model architectures.

### 1. MedMNIST+ Baseline

![github_benchmark_300dpi](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/assets/98497332/0b18ce89-f095-4187-8842-893255f47808)

Figure 1: Side-by-side comparison of the 12 2D datasets included in MedMNIST+, showcasing diverse primary data modalities and classification tasks across four image resolutions (left). Benchmark outcomes summarizing the average mean and standard deviation of accuracy
(ACC) and area under the receiver operating characteristic curve (AUC) across all datasets for all training scheme-model-image resolution combinations, derived from three independent random seeds (right).

### 2. Potential of Computationally Efficient Training Schemes

<p align="middle">
  <img src="https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/assets/98497332/2286c7d1-77c6-4767-8217-52d3656af536" width="750" />
</p>

Figure 2: Ranking analysis showcasing the frequency of model placements among the top-5 performers in terms of accuracy (ACC) across all training schemes and resolutions (top), for each training scheme separately (center), and for both training schemes and resolutions, respectively (bottom) across all
datasets.

### 3. Performance Improvement Cap of Higher Resolutions

<p align="middle">
  <img src="https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/assets/98497332/859d9e66-a786-44af-9ebf-aab096be12e8" width="750" />
</p>

Figure 3: Analysis of model performance (ACC) improvement with increasing input resolution across all $12$ datasets. The figure illustrates the frequency of performance enhancements as input resolutions progress from $28 \times 28$ to $64 \times 64$, $64 \times 64$ to $128 \times 128$, and $128 \times 128$ to $224 \times 224$, encompassing all models and training schemes. Each bar signifies for how many datasets the model performance, in terms of the mean accuracy across the three random seeds, is superior at the next higher resolution compared to the preceding lower one, with a maximum of 12 improvements per transition.

### 4. Competitiveness of CNNs compared to ViTs

![box_dpi300](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/assets/98497332/b85c7442-e713-4455-83ce-ecd83a24eae4)

Figure 4: Illustrating the accuracy (ACC) distributions exhibited by each model averaged across all 12 datasets, delineated by training scheme (subplots) and input resolution (color coded) (left). Percentile statistics for each model performance in terms of averaged accuracy (ACC)
across all training schemes and input resolutions across all 12 datasets (right).

## Getting Started 🚀
### Project Structure
- [`config.yaml`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/config.yaml): Training and evaluation configuration
- [`environment.yaml`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/environment.yaml): Requirements
- [`evaluate.py`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/evaluate.py): Evaluation script
- [`feature_extraction.py`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/feature_extraction.py): Feature embedding extractor
- [`main.py`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/main.py): Main script
- [`train.py`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/train.py): Training script
- [`utils.py`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/utils.py): Helper functions

### Installation and Requirements
### Quick Start

# Citation 📖
If you find this work useful in your research, please consider citing our paper:
- [Publication](...)
- [Preprint](https://arxiv.org/abs/2404.15786)

```
@article{doerrich2024rethinking,
      title="Rethinking Model Prototyping through the MedMNIST+ Dataset Collection", 
      author="Sebastian Doerrich and Francesco Di Salvo and Julius Brockmann and Christian Ledig",
      year="2024",
      eprint="2404.15786",
      archivePrefix="arXiv",
      primaryClass="eess.IV"
}
```
