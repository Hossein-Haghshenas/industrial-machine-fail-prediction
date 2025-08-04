# Industrial Machine Failure Prediction

This project is a machine learning model that predicts industrial machine maintenance and possible failures.

## Table of Contents

- [Overview](#overview)
- [Required Libraries](#required-libraries)
- [Folders Structure](#folders-structure)
- [About the Dataset](#about-the-dataset)
- [Preprocessing dataset](#preprocessing-dataset)
- [Dataset Analysis](#dataset-analysis)
- [Data Imbalance](#data-imbalance)
- [Visualizations & Outputs](#visualizations--outputs)
- [Citation](#citation)

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

---

## Folders Structure

```bash

├── dataset/                                # dataset path
├───────── ai4i2020.csv                     # original data
├───────── /preprocessing                   # preprocessed dataset path
├─────────────────────── /X_train.csv       # processed training data
├─────────────────────── /X_test.csv        # processed test data
├─────────────────────── /y_train.csv       # labels for training data
├─────────────────────── /y_test.csv        # labels for test data
├── plots/                                  # visualizations & plots
├── preprocessing-data.ipynb                # dataset analyzing & preprocessing codes
├── predict-maintenance.ipynb               # training and evaluating models codes
├── requirements.txt                        # requirements libraries for installing


```

---

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

---

## Preprocessing dataset

### Step 1

In this stage we at first read the dataset & check some info like shape of dataset that means the count of data and features using df.shape
after that we check the first 5 data row to analysis the data values and types, using df.head()
we can also check the df.info() and df.description() to discover more info about dataset same as null values count in each features and the type of each features and etc...

---

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

---

### Step 3

In this step, we create **new meaningful features** derived from the existing data to enhance our machine learning model’s performance. Feature engineering helps uncover deeper patterns and relationships in the data that are not immediately visible in the raw form.

We define three new features:

#### 1. power\_[W]

This feature estimates the **mechanical power** based on torque and rotational speed using the following formula:

```python
# Convert RPM to radians per second
rad_per_sec = rpm * 0.10472

# Power in Watts
power_[W] = Torque [Nm] × Rotational speed [rad/s]
```

This gives us a more direct insight into the machine's mechanical workload

#### 2. temperature*difference*[K]

This feature calculates the difference between the process temperature and the surrounding air temperature:

```python
temperature_difference_[K] = Process temperature [K] - Air temperature [K]
```

This helps capture thermal stress within the system that may lead to overheating or component degradation.

#### 3. tool_wear_strain

This feature combines tool wear and torque to reflect mechanical degradation under stress:

```python
tool_wear_strain = Tool wear [min] × Torque [Nm]
```

Higher values may indicate machines operating under risky mechanical conditions.

After creating these features, we compute their correlation with machine failures to evaluate their predictive power.
Finally, we visualize their distributions using histogram plots, allowing us to better interpret their relationship with failures.

---

### Step 4

In this step, we preprocess the dataset to prepare it for machine learning models. This includes feature selection, encoding, scaling, and saving the processed data for training and testing.

#### 1. Separate Features and Target

We drop the following columns:

- UDI, Product ID: Unique identifiers, not useful for modeling.

- Machine failure, TWF, HDF, PWF, OSF, RNF: Target labels that should not be part of input features.

#### 2. Split the Dataset

We split the dataset into training and testing sets using train_test_split():

- test_size=0.2: 20% of the data goes to testing.

- stratify=y_data: Ensures the class distribution (failure vs non-failure) is maintained across train and test sets.

- random_state=42: Ensures reproducibility.

#### 3. Define Feature Transformation Strategy

We define two feature groups:

- Categorical: ['Type']

- Numerical: All columns except 'Type'

#### 4. Create Preprocessing Pipeline

We use ColumnTransformer to apply:

- StandardScaler to numerical features (for normalization)

- OneHotEncoder to the categorical feature 'Type' (dropping the first to avoid multicollinearity)

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])
```

#### 5. Fit and Transform Data

- Fit the preprocessor on the training set (fit_transform)

- Apply the same transformation on the test set (transform) without leaking test data

#### 6. Retrieve and Clean Feature Names

After transformation, we reconstruct feature names:

- Numerical features retain original names

- One-hot encoded features get expanded names like Type_H, Type_L, etc.

#### 7. At the end we will clean column names from problematic characters such as brackets ([]) or comparison symbols (<, >) and save ready data.

## Dataset Analysis

## Failure Type Analysis Based on Feature Correlation

In this dataset, machine failures are categorized into **five types**, each with distinct correlations to system features. Here's a detailed summary of the analysis based on the correlation matrix:

### 1. TWF – Tool Wear Failure

- **36%** of all machine failures are due to this type.
- **Most influential feature**: `Tool Wear [min]`

### 2. HDF – Heat Dissipation Failure

- This is the **most common failure type**.
- **Key contributing features**:
  - `Torque [Nm]`
  - `Air temperature [K]`
  - `Rotational speed [rpm]`
- **Least relevant feature**: `Tool Wear [min]` and `Process temperature [K]`

### 3. PWF – Power Failure

- Occurs when calculated power is outside the safe operating range.
- **Strongest correlated features**:
  - `Torque [Nm]`
  - `Rotational speed [rpm]`

### 4. OSF – Overstrain Failure

- Caused by excessive pressure or load (torque × tool wear).
- **Most impactful features**:
  - `Torque [Nm]`
  - `Tool Wear [min]`
  - `Rotational speed [rpm]`
- **Least impactful feature**: `Air temperature [K]` and `Process temperature [K]`

### 5. RNF – Random Failure

- **Rare and unpredictable**, likely caused by external/random factors.
- **No significant correlation** with any measured features.

### Overall Machine Failure Correlations

When considering the **general machine failure** (`Machine failure` label), the top influencing features are:

1. `Torque [Nm]`
2. `Tool Wear [min]`
3. `Process temperature [K]`

---

## Feature Distributions in Failed vs. Non-Failed Machines

Below is an analysis of the distribution of numerical features with respect to machine failures (1 = failure, 0 = no failure). The distributions help us understand which value ranges are more likely to lead to failures:

### Torque [Nm]

- Failures are **more frequent** when torque is between **45 to 65 Nm**.
- Higher torque levels put more mechanical stress on components, leading to increased failure risk.

### Tool Wear [min]

- Machines with **tool wear between 180 to 240 minutes** show a **higher likelihood of failure**, particularly due to wear-related issues.
- The wear accumulation over time plays a direct role in TWF (Tool Wear Failure).

### Rotational Speed [rpm]

- Most failures occur in the **1270 to 1380 rpm** range.
- This may indicate that machines operating in this speed band are more prone to instability or stress-related breakdowns.

### Air Temperature [K]

- Increased failures are seen when **air temperature ranges between 301.8K and 303.7K**.
- This temperature zone may cause inadequate cooling, contributing to heat dissipation failure (HDF).

### Process Temperature [K]

- A high concentration of failures is visible when **process temperature lies between 309.8K and 312K**.
- Overheating in this range could affect internal components and directly impact the system's operational integrity.

---

## Data Imbalance

How did we deal with data imbalance ?
we used SMOTE (Synthetic Minority Oversampling Technique). SMOTE works by generating synthetic examples for the minority class by interpolating between existing minority class samples.

```python
smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### Effect of SMOTE

- SMOTE increased the number of minority class samples by generating new synthetic examples.

- It helped improve the classifier's performance by giving it a more balanced dataset to learn from.

- As a result, recall and F1-score for the minority class improved significantly during evaluation.

## Visualizations & Outputs

---

### Citation

> S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications,"
> 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), 2020, pp. 69-74.
> DOI: [10.1109/AI4I49448.2020.00023](https://doi.org/10.1109/AI4I49448.2020.00023)

```

```
