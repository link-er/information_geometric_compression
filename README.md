# Geometric and Information Compression of Representations in Deep Learning

Code for the paper: [Geometric and Information Compression of Representations in Deep Learning](https://arxiv.org/pdf/2606.21593)

Checkpoints: [https://doi.org/10.17877/RCTRUST-2026-DWMJTZ]

## Overview

This repository includes experiments for two settings:

- **CEB experiments**: models and tasks used to estimate conditional mutual information (MI) on the fly during training, followed by computing neural collapse (NC), plotting and summarization.
- **Gaussian experiments**: a more involved pipeline with three stages:
  1. estimate MI and NC,
  2. plot the results,
  3. generate summaries.
- **Toy example**: a separate notebook with a minimal illustrative example.

The datasets used in the paper are standard public datasets referenced in the manuscript. Training checkpoints are available in the open repository.

## Repository structure

```text
.
├── ceb/
│   ├── run.sh # running training script train.py for the set of random seeds
│   ├── visualize_training.py # computes NC for the particular run and visualizes training curves
│   ├── correlation_plots.py # aggregates runs into the correlation (MI-NC and others) plots
│   ├── correlation_coefs.py # computes correlation coefficients reported in the manuscript
├── gaus_dropout/
│   ├── train_with_NC.py # training a neural network with NC regularizer
│   ├── process_step1_mi_nc.py # compute MI and NC for the saved checkpoints
│   ├── process_step2_plotting.py # plot the training curves and correlations in one setup
│   ├── process_step3_correlation_coefs.py # compute correlation coefficients reported in the paper
├── toy_example.ipynb # computation and visualization used as theoretical explanation for mismatch between MI compression and NC
```
Checkpoints of the training have to be placed in "checkpoints" folder in gaus_dropout - the scripts are expecting to find it there and the training script saves runs there.

## Requirements

- Python
- PyTorch
- Standard scientific Python libraries
- GPU recommended for practical training time

## Reproducing the experiments

### 1. CEB experiments

For the CEB experiments, MI and NC are collected during training. After training, run the plotting and summarization scripts.

### 2. Gaussian dropout experiments

For the Gaussian dropout experiments, reproduction happens in four stages:

1. Train networks or download checkpoints from [https://doi.org/10.17877/RCTRUST-2026-DWMJTZ].
2. Estimate MI (the current code uses DoE estimator from the corresponding repository of the paper) and NC.
3. Plot the results.
4. Produce summaries.

### 3. Toy example

The toy example is provided as a notebook.

```bash
jupyter notebook toy_example.ipynb
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{adilovageometric,
  title={Geometric and Information Compression of Representations in Deep Learning},
  author={Adilova, Linara and Petzka, Henning and Fischer, Asja and Geiger, Bernhard C},
  booktitle={Joint European conference on machine learning and knowledge discovery in databases},
  year={2026}
}
```

## Contact

For questions, please contact [linara.adilova@tu-dortmund.de].