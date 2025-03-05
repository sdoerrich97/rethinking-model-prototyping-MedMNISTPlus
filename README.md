# Rethinking model prototyping through the MedMNIST+ dataset collection
<p align="center">
    [<a href="https://arxiv.org/abs/2404.15786">Preprint</a>]
    [<a href="https://www.nature.com/articles/s41598-025-92156-9">Publication</a>]
    [<a href="#citation-">Citation</a>]
</p>

## Overview 🧠
The integration of deep learning based systems in clinical practice is often impeded by challenges rooted in limited and heterogeneous medical datasets. In addition, the field has increasingly prioritized marginal performance gains on a few, narrowly scoped benchmarks over clinical applicability, slowing down meaningful algorithmic progress. This trend often results in excessive fine-tuning of existing methods on selected datasets rather than fostering clinically relevant innovations. In response, this work introduces a comprehensive benchmark for the [MedMNIST+ dataset collection](https://zenodo.org/records/10519652), designed to diversify the evaluation landscape across several imaging modalities, anatomical regions, classification tasks and sample sizes. We systematically reassess commonly used Convolutional Neural Networks (CNNs) and Vision Transformer (ViT) architectures across distinct medical datasets, training methodologies, and input resolutions to validate and refine existing assumptions about model effectiveness and development. Our findings suggest that computationally efficient training schemes and modern foundation models offer viable alternatives to costly end-to-end training. Additionally, we observe that higher image resolutions do not consistently improve performance beyond a certain threshold. This highlights the potential benefits of using lower resolutions, particularly in prototyping stages, to reduce computational demands without sacrificing accuracy. Notably, our analysis reaffirms the competitiveness of CNNs compared to ViTs, emphasizing the importance of comprehending the intrinsic capabilities of different architectures. Finally, by establishing a standardized evaluation framework, we aim to enhance transparency, reproducibility, and comparability within the MedMNIST+ dataset collection as well as future research.

Subsequent sections outline the paper's [key contributions](#key-contributions-), showcase the [obtained results](#results-), and offer instructions on [accessing and utilizing the accompanying codebase](#getting-started-) to replicate the findings and train or evaluate your own models.

## Key Contributions 🔑
- Systematic benchmarking of a wide range of commonly used models across diverse medical datasets, accounting for variations in resolutions, tasks, sample sizes, and class distributions.
- Identification of systematic strengths and weaknesses inherent in traditional models within the context of medical image classification.
- Reevaluation of prevalent assumptions with respect to model design, training schemes and input resolution requirements.
- Presentation of a solid baseline performance for MedMNIST+ and a standardized evaluation framework for assessing future model performance in medical image classification.
- Formulation of recommendations and take-aways for model development and deployment.

## Results 📊
First and foremost, the paper presents the very first solid baseline evaluation for the MedMNIST+ dataset collection. Furthermore, the findings suggest that computationally efficient training schemes and modern foundation models hold promise in bridging the gap between expensive end-to-end training and more resource-refined approaches. Contrary to prevailing assumptions, the authors observe that higher resolutions may not consistently improve performance beyond a certain threshold, advocating for the use of lower resolutions, particularly in prototyping stages, to expedite processing. Additionally, the analysis reaffirms the competitiveness of convolutional models compared to Vision Transformer (ViT)-based architectures, emphasizing the importance of understanding the intrinsic capabilities of different model architectures.

### MedMNIST+ Baseline

![github_benchmark_300dpi](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/assets/98497332/0b18ce89-f095-4187-8842-893255f47808)

Figure 1: Side-by-side comparison of the 12 2D datasets included in MedMNIST+, showcasing diverse primary data modalities and classification tasks across four image resolutions (left). Benchmark outcomes summarizing the average mean and standard deviation of accuracy
(ACC) and area under the receiver operating characteristic curve (AUC) across all datasets for all training scheme-model-image resolution combinations, derived from three independent random seeds (right).

### Potential of Computationally Efficient Training Schemes

<p align="middle">
  <img src="https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/assets/98497332/2286c7d1-77c6-4767-8217-52d3656af536" width="750" />
</p>

Figure 2: Ranking analysis showcasing the frequency of model placements among the top-5 performers in terms of accuracy (ACC) across all training schemes and resolutions (top), for each training scheme separately (center), and for both training schemes and resolutions, respectively (bottom) across all
datasets.

### Performance Improvement Cap of Higher Resolutions

<p align="middle">
  <img src="https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/assets/98497332/859d9e66-a786-44af-9ebf-aab096be12e8" width="750" />
</p>

Figure 3: Analysis of model performance (ACC) improvement with increasing input resolution across all $12$ datasets. The figure illustrates the frequency of performance enhancements as input resolutions progress from $28 \times 28$ to $64 \times 64$, $64 \times 64$ to $128 \times 128$, and $128 \times 128$ to $224 \times 224$, encompassing all models and training schemes. Each bar signifies for how many datasets the model performance, in terms of the mean accuracy across the three random seeds, is superior at the next higher resolution compared to the preceding lower one, with a maximum of 12 improvements per transition.

### Competitiveness of CNNs compared to ViTs

![box_dpi300](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/assets/98497332/9a58cbec-4a2b-4a0c-b04b-e51958501911)

Figure 4: Illustrating the accuracy (ACC) distributions exhibited by each model averaged across all 12 datasets, delineated by training scheme (subplots) and input resolution (color coded) (left). Percentile statistics for each model performance in terms of averaged accuracy (ACC) across all training schemes and input resolutions across all 12 datasets (right).

### Significant Pair-Wise Model Performance Differences

<p align="middle">
  <img src="https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/assets/98497332/e6aae3b9-1d30-4dfc-90b3-65d77cf74574" width="650" />
</p>

Figure 5: llustration of pair-wise significant differences between model performance in terms of averaged accuracy across all training schemes, input resolutions, and all 12 datasets using the results of the pair-wise Wilcoxon signed-rank tests with a Bonferroni correction (adjusted significance level of p < 0.0011). (_Green Diamond_: significant difference favoring the model in the row, _Green Trophy_ : significant difference favoring the model in the column, _Red Cross_ : no significant difference).

## Getting Started 🚀
### Project Structure
- [`config.yaml`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/config.yaml): Training and evaluation configuration
- [`environment.yaml`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/environment.yaml): Package Requirements
- [`evaluate.py`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/evaluate.py): Evaluation script
- [`feature_extraction.py`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/feature_extraction.py): Feature embedding extractor
- [`main.py`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/main.py): Main script
- [`train.py`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/train.py): Training script
- [`utils.py`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/utils.py): Helper functions

### Installation and Requirements
#### Clone this Repository:
To clone this repository to your local machine, use the following command:
```
git clone https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus.git
```

#### Set up a Python Environment Using Conda (Recommended) 
If you don't have Conda installed, you can download and install it from [here](https://conda.io/projects/conda/en/latest/index.html).
Once Conda is installed, create a Conda environment with Python 3 (>= 3.11) in your terminal:
```
conda create --name rethinkingPrototyping python=3.11
```
Of course, you can use a standard Python distribution as well.

#### Install Required Packages From the Terminal Using Conda (Recommended)
All required packages are listed in [`environment.yaml`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/environment.yaml).

Activate your Conda environment in your terminal:
```
conda activate rethinkingPrototyping
```

Once Conda is activated, install PyTorch depending on your system's configuration. For example for Linux using Conda and CUDA 12.1 use the following command. For all other configurations refer to the official [PyTorch documentation](https://pytorch.org/):
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install required Python packages via Conda:
```
conda install fastai::timm
conda install anaconda::scikit-learn
```

Additionally, navigate to your newly created Conda environment within your Conda install and install the remaining Python Packages from [PyPI](https://pypi.org/):
```
cd ../miniconda3/envs/rethinkingPrototyping/Scripts
pip install medmnist
```

If you use a standard Python distribution instead, you need to adjust the installation steps accordingly.

### Quick Start
Once all requirements are installed, make sure the Conda environment is active and navigate to the project directory:
```
cd ../rethinking-model-prototyping-MedMNISTPlus
```

You can adjust the parameters and hyperparameters of each training/evaluation run within your copy of [`config.yaml`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/config.yaml). These are
```
# Fixed Parameters
data_path: # Where the dataset is stored (' ')
output_path: # Where the trained model shall be stored. (' ')
epochs: # How many epochs to train for. (100)
learning_rate: # Learning rate (0.0001)
batch_size: # Batch size for the training. (64)
batch_size_eval: # Batch size for the evaluation. (256)
device: # Which device to run the computations on. ('cuda:0')

# Modifiable Parameters
dataset: # Which dataset to use. ('bloodmnist', 'breastmnist', 'chestmnist', 'dermamnist', 'octmnist', 'organamnist', 'organcmnist', 'organsmnist', 'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist')
img_size: # Height and width of the input image. (28, 64, 128, 224)
training_procedure: # Which training procedure to use. ('endToEnd', 'linearProbing', 'kNN')
architecture: # Which model to use. ('vgg16', 'alexnet', 'resnet18', 'densenet121', 'efficientnet_b4', 'vit_base_patch16_224', 'vit_base_patch16_clip_224', 'eva02_base_patch16_clip_224', 'vit_base_patch16_224.dino', 'samvit_base_patch16')
k: # Number of neighbors for the kNN.
seed: # Seed for random operations for reproducibility. (9930641, 115149041, 252139603)
```

Once the config file is all set, you can run a combined training and evaluation run using:
```
python main.py --config_file './config.yaml'
```

Additionally, you can adjust the _#Modifiable Parameters_ (not the  _#Fixed Parameters_!) on the fly, using for example:
```
python main.py --config_file './config.yaml' --dataset 'bloodmnist' --img_size 224 --training_procedure 'endToEnd' --architecture 'vgg16' --seed 9930641
```

If you only want to run either training or evaluation, you can run the respective scripts independently:
```
python train.py --config_file './config.yaml'
python evaluate.py --config_file './config.yaml'
```

Lastly, you can use [`feature_extraction.py`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/feature_extraction.py) to extract the latent embeddings of all used models before the classification head:
```
python feature_extraction.py --data_path '<DATAPATH>' --output_path '<OUTPUTPATH>'
```
Please replace `<DATAPATH>` and `<OUTPUTPATH>` with the respective paths to your files.

You will find all parameters (model architectures, number of epochs, learning rate, etc.) we used for our benchmark within the provided [`config.yaml`](https://github.com/sdoerrich97/rethinking-model-prototyping-MedMNISTPlus/blob/main/config.yaml) in case you want to reproduce our results. If you want to use our evaluation framework for your own models and datasets, you only need to adjust the config-file, respectively.

# Citation 📖
If you find this work useful in your research, please consider citing our paper:
- [Publication](https://www.nature.com/articles/s41598-025-92156-9)
- [Preprint](https://arxiv.org/abs/2404.15786)

```
@article{doerrich2025rethinking,
author={Doerrich, Sebastian
and {Di Salvo}, Francesco
and Brockmann, Julius
and Ledig, Christian},
title={Rethinking model prototyping through the MedMNIST+ dataset collection},
journal={Scientific Reports},
year={2025},
month={Mar},
day={05},
volume={15},
number={1},
pages={7669},
issn={2045-2322},
doi={10.1038/s41598-025-92156-9},
url={https://doi.org/10.1038/s41598-025-92156-9}
}
```
