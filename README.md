# grm-paper
This is a duplicate of the repository for the project on glucose regulatory measure. The link to the original repository is https://github.com/KlickInc/klickhealth-labs-papers-grm-public

# Introduction
This repository contains the code required to replicate the results of the manuscript titled "Model-Based Extraction of T2D Diagnostic Information from Continuous Glucoses Monitoring" submitted to IEEE Open Journal of Engineering in Medicine and Biology (OJEMB) currently under peer review.

We develop a new approach for the prediction of type 2 diabetes (T2D) using data collected by continuous glucose monitors (CGMs). The algorithm first detects and extracts hyperglycemic episodes (i.e. peaks) from time series glucose data, then a glucose homeostasis model is fit to each extracted peak. Glucose homeostasis is modelled as a negative feedback system via proportional-integral (PI) control, which loosely mimics the biphasic pattern of insulin secretion. Each extracted peak is then characterized by a set of model parameters from model fitting. The distribution of these model parameters are used to determine the diabetic statuses of each individual.

# Structure of Framework
The code here is separated into two main steps. The first step is the so-called peak extraction process where peaks in glucose data are selected from raw CGM data. The PI model is then fit to each selected peak, where the fitted model parameters are saved for each individuals. The second step uses the model parameter sets from the first step to define the Glucose Regulatory Measure (GRM), a new metric for classification of T2D statuses. 

# Prerequisites
Python 3 (3.8.5)
## Modules
- yaml (5.1)
- matplotlib (3.5.3)
- scipy (1.5.2)
- deprecation (2.1.0)
- yacs (0.1.7)
- pandas (1.1.0)
- numpy (1.23.5)
- scikit-learn (0.23.1)
- pyrcca (0.2)

# Installation
- Clone this repo:
```bash
git clone https://github.com/eshn/grm-paper.git
cd grm-paper
```
<!--- Install Python Requirements:
```bash
pip install -r requirements.txt
```-->

# Datasets
Raw CGM data is not available due to proprietary ownership. Fitted model parameters are available via https://dx.doi.org/10.21227/bx9e-2b60 (IEEE Dataport subscription required).

# Usage
To be updated
<!--- ## 1) Setup
### Dataset Placement
### Experiment Configuration (optional)
## 2) Model Configurations
-->

# License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

# Citation
To be updated upon the completion of peer review
