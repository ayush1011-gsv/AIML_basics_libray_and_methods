# Pima Indians Diabetes: Data Loading, Descriptive Statistics, Visualization and Preparation

A Google Colab notebook that walks through the standard first steps of a machine learning project on the Pima Indians Diabetes dataset: loading the data in several ways, understanding it with descriptive statistics and plots, and preparing it with common scaling transforms.

<a href="https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/YOUR_NOTEBOOK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Contents

1. [Loading the CSV](#1-loading-the-csv)
2. [Descriptive statistics](#2-descriptive-statistics)
3. [Data visualization](#3-data-visualization)
4. [Preparing the data](#4-preparing-the-data)

## Dataset

`pima-indians-diabetes.data.csv` has **768 rows and 9 columns** and no header row. The first eight columns are inputs and the last one is the class label.

| Column | Meaning |
|--------|---------|
| `preg` | Number of times pregnant |
| `plas` | Plasma glucose concentration |
| `pres` | Diastolic blood pressure (mm Hg) |
| `skin` | Triceps skinfold thickness (mm) |
| `test` | 2-hour serum insulin (mu U/ml) |
| `mass` | Body mass index |
| `pedi` | Diabetes pedigree function |
| `age` | Age in years |
| `class` | 1 = diabetes, 0 = no diabetes |

All columns are integers except `mass` and `pedi`, which are floats.

## Requirements

- Python 3.8 or later
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`

Colab already includes all of them. To run locally:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## How to run

1. Click the **Open in Colab** badge above.
2. Put `pima-indians-diabetes.data.csv` in your Google Drive at `MyDrive/GSVPRML/`, or change the `filename` in each cell to wherever the file is.
3. Run the first cell to mount Google Drive and allow access when asked.
4. Choose **Runtime → Run all**.

---

## 1. Loading the CSV

The same file is loaded in three ways:

| Method | Notes | Result |
|--------|-------|--------|
| Python `csv` module + NumPy | Reads rows as text and converts to a float array | shape `(768, 9)` |
| `numpy.loadtxt` | Needs no header row and all values in one format | shape `(768, 9)` |
| `pandas.read_csv` | Most flexible, returns a DataFrame with named columns | shape `(768, 9)` |

Pandas is used in the rest of the notebook. A cell for loading directly from a URL is included but commented out.

## 2. Descriptive statistics

Seven checks are made to understand the data:

| Step | Code | Finding |
|------|------|---------|
| Peek at the data | `data.head(20)` | First 20 rows look as expected |
| Data types | `data.dtypes` | 7 integer columns, 2 float columns |
| Summary statistics | `data.describe()` | Count, mean, std, min, quartiles and max per column |
| Class distribution | `groupby('class').size()` | 500 without diabetes (65.1%), 268 with diabetes (34.9%) |
| Correlations | `data.corr(method='pearson')` | See below |
| Skew | `data.skew()` | See below |

**Correlations.** The strongest relationships with `class` are `plas` (0.467) and `mass` (0.293). Among the inputs, `preg` and `age` are the most correlated pair (0.544), followed by `skin` and `test` (0.437).

**Skew.** `test` (2.27), `pedi` (1.92) and `age` (1.13) are strongly right-skewed. `pres` (-1.84) is skewed the other way, which is mostly caused by the zero values described below.

**Zeros that look like missing values.** `plas`, `pres`, `skin`, `test` and `mass` all have a minimum of 0, which is not physically possible for glucose, blood pressure or BMI. These zeros are most likely missing readings. They are left as they are in this notebook, but they should be handled (for example replaced with the median) before training a model.

## 3. Data visualization

**Univariate plots** (one attribute at a time):

- **Histograms**: `age`, `pedi` and `test` look exponential, while `mass`, `pres` and `plas` look roughly Gaussian.
- **Density plots**: a smooth version of each histogram.
- **Box and whisker plots**: show the median, the middle 50% of the values and the outliers.

**Multivariate plots** (relationships between attributes):

- **Correlation matrix plot**: a colour-coded version of the correlation table above, with a colour bar from -1 to 1. It is drawn once with labelled axes and once in a generic form.
- **Scatter plot matrix**: every pair of attributes plotted against each other. A second version colours the points by class (blue = no diabetes, orange = diabetes) and uses a larger figure.

## 4. Preparing the data

Each transform is applied to the eight input columns (`X`), with the class column (`Y`) kept separate. The first five rows are printed.

| Transform | scikit-learn class | What it does | Useful for |
|-----------|--------------------|--------------|------------|
| Rescale | `MinMaxScaler(feature_range=(0, 1))` | Scales each column to the range 0 to 1 | Gradient descent, neural networks, k-NN |
| Standardize | `StandardScaler` | Gives each column mean 0 and standard deviation 1 | Linear and logistic regression, LDA |
| Normalize | `Normalizer` | Scales each **row** to unit length (norm 1) | Sparse data with attributes on different scales |
| Binarize | `Binarizer(threshold=0.0)` | Sets values above 0 to 1 and all others to 0 | Turning values into crisp yes/no features |

Note that `MinMaxScaler` and `StandardScaler` work on columns, while `Normalizer` works on rows. For a real model, fit the scaler on the training data only and then apply it to the test data.

---

## Project structure

```
.
├── README.md
├── YOUR_NOTEBOOK.ipynb
└── pima-indians-diabetes.data.csv   (stored in Google Drive: MyDrive/GSVPRML/)
```
