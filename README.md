# Industrial Machine Failure Prediction

This project is a machine learning model that predicts industrial machine maintenance and possible failures.

## Table of Contents

- [Overview](#overview)
- [Required Libraries](#required-libraries)
- [About the Dataset](#about-the-dataset)
- [Folders Structure](#folders-structure)
- [Visualizations & Outputs](#visualizations--outputs)

---

## Overview

Predictive maintenance is a critical part of industrial systems, where identifying potential machine failures in advance can save cost, time, and resources.  
In this project, we build a machine learning pipeline to detect possible failures using sensor and operational data from industrial machines.

We focus on:

- Preprocessing and engineering meaningful features
- Handling class imbalance using SMOTE
- Training and evaluating different classifiers (Logistic Regression, Random Forest, XGBoost)
- Visualizing correlations and model outputs

---

## Required Libraries

We used this libraries in our project:

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
```

## About the Dataset

Predictive Maintenance dataset is modeled after a real milling machine and contains **10,000 records** with **14 features** each.

### Features:

- `UID`: Unique identifier (1–10,000)
- `Product ID`: Contains a product quality label (`L`, `M`, `H`) and a serial number
- `Type`: Product type extracted from the Product ID (`L`, `M`, `H`)
- `Air temperature [K]`: Simulated using a normalized random walk (std = 2K) around 300K
- `Process temperature [K]`: Air temp + 10K + random noise (std = 1K)
- `Rotational speed [rpm]`: Simulated from 2860 W power with added Gaussian noise
- `Torque [Nm]`: Normally distributed around 40 Nm (std = 10 Nm), no negative values
- `Tool wear [min]`: Wear time varies by product type (H/M/L adds 5/3/2 minutes)
- `Machine failure`: Binary label (1: failure, 0: normal)

### Failure Types (used to set `Machine failure` = 1 if any of the following are true):

- **TWF (Tool Wear Failure):** Tool fails between 200–240 mins of wear
- **HDF (Heat Dissipation Failure):** Occurs if temp difference < 8.6K and speed < 1380 rpm
- **PWF (Power Failure):** Power outside 3500–9000 W range
- **OSF (Overstrain Failure):** Torque × Tool Wear exceeds threshold (11,000–13,000 depending on type)
- **RNF (Random Failure):** 0.1% chance of failure independent of input features

Note: The dataset only includes the final failure label and **does not specify** which of the five modes caused the failure.

## Folders Structure

```bash

├── dataset/                  # dataset path
├── plots/                    # visualizations & plots
├── preprocessing-data.ipynb  # dataset analyzing & preprocessing codes
├── predict-maintenance.ipynb # training and evaluating models codes
├── X_train.csv               # processed training data
├── X_test.csv                # processed test data
├── y_train.csv               # labels for training data
├── y_test.csv                # labels for test data

```

## Preprocessing dataset

### Step 1

In this stage we at first read the dataset & check some info like shape of dataset that means the count of data and features using df.shape
after that we check the first 5 data row to analysis the data values and types, using df.head()
we can also check the df.info() and df.description() to discover more info about dataset same as null values count in each features and the type of each features and etc...

### Step 2

In the next step :

1. We will check the count of all missing values in dataset for each column
2. Calculate the percent of all type of failures in machines
3. After that we will calculate the failures base on types, here is the result :

```bash
machine failures: 3.39%
TWF     46
HDF    115
PWF     95
OSF     98
RNF     19
```

4. Then we will filter the numeric data and visualize the correlation matrix heat map to analyze the relation between data
5. We also plots the distribution histograms of the main numerical features and colors them based on whether the machine is operational or faulty, so that the distribution differences in the two states can be compared.

## Visualizations & Outputs

### Citation:

> S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications,"  
> 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), 2020, pp. 69-74.  
> DOI: [10.1109/AI4I49448.2020.00023](https://doi.org/10.1109/AI4I49448.2020.00023)
