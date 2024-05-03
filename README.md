# Rethinking Model Prototyping through the MedMNIST+ Dataset Collection

## Overview 🧠
This repository contains the official code for the paper titled "Rethinking Model Prototyping through the MedMNIST+ Dataset Collection". The paper aims to address the challenges hindering the integration of deep learning based systems into clinical practice by re-emphasizing the importance of clinically relevant innovations over marginal performance improvements on narrowly scoped benchmarks. It introduces a comprehensive benchmark for the MedMNIST+ database to evaluate common convolutional neural networks (CNNs) and Transformer-based architectures for medical image classification, highlighting the potential of computationally efficient training schemes and modern foundation models. The findings suggest that lower resolutions may expedite processing without compromising performance, reaffirming the competitiveness of convolutional models compared to Vision Transformer-based architectures. The standardized evaluation framework aims to enhance transparency, reproducibility, and comparability in medical image classification research.

## Key Features 🔑
- Presentation of a solid baseline performance for MedMNIST+ and a standardized evaluation
framework for assessing future model performance in medical image classification.
- Identification of systematic strengths and weaknesses inherent in traditional models within
the context of medical image classification.
- Reevaluation of prevalent assumptions with respect to model design, training schemes and
input resolution requirements.
- Formulation of recommendations and take-aways for model development and deployment.

## Results 📊
TODO

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

## Acknowledgements 👏
TODO

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
