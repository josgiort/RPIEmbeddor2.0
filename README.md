# RPIEmbeddor

Official implementation of **RPIEmbeddor**, a transformer-based framework for RNA--protein interaction prediction using pretrained RNA and protein foundation-model embeddings.

This repository accompanies the Master's thesis:

> **Evaluating foundation model embeddings in RNA-Protein Interaction classification with RPIembeddor**

## Repository Structure

```text
.
├── src/                        Model architecture, training and inference
├── baselines/                  Logistic Regression, CNN and MLP baselines
├── scripts/
│   └── embedding_generation/   Embedding generation utilities
├── figures/
├── data/
└── requirements.txt
```

## Installation

Create the environment:

```bash
conda create -n rpi python=3.8 pip
conda activate rpi
pip install -r requirements.txt
```

Install ESM-2:

```bash
pip install git+https://github.com/facebookresearch/esm.git
```

Install RNA-FM:

```bash
pip install rna-fm
```

## Data

The datasets, pretrained embeddings, checkpoints, and experiment logs are not included because of their size and possible redistribution constraints.

Expected directory structure:

```text
data/
├── annotations/
├── interactions/
└── embeddings/
```

The paths to these directories can be configured in the training and inference scripts.

## Embedding Generation

Scripts for generating RNA and protein embeddings are provided under:

```text
scripts/embedding_generation/
```

Supported representations include RNA-FM, ESM-2, ESM-C, LaMAR, LCPLM, and one-hot sequence representations.

## Training

Training scripts for the experiments reported in the thesis are available under:

```text
src/
```

These include experiments on eCLIP2 and RNAInterAct using RNA-disjoint, protein-disjoint, and random data splits, as well as frozen-embedding and one-hot ablations.

## Baseline Models

The repository contains the following baseline models:

- Logistic Regression
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)

The baselines operate on mean-pooled RNA and protein foundation-model embeddings.

## Inference

Inference can be performed with:

```bash
python src/inference.py
```

For protein sequences, the supported amino acid alphabet is:

```text
A R N D C Q E G H I L K M F P S T W Y V
```

For RNA sequences, the supported nucleotide alphabet is:

```text
A C G U
```

## Requirements

The required Python packages are listed in `requirements.txt`.

## Citation

If you use this repository, please cite the accompanying Master's thesis.
