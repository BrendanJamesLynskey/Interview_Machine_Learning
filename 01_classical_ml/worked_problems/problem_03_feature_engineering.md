# Problem 03: Feature Engineering

**Difficulty**: Intermediate to Advanced  
**Topics covered**: Feature scaling, encoding categorical variables, handling missing data, feature creation, feature selection, pipelines

---

## Background

Feature engineering is often the highest-leverage activity in a machine learning project. Raw data rarely comes in a form that classical ML algorithms can directly use. This problem walks through a realistic feature engineering pipeline for a structured tabular dataset, motivating each decision and demonstrating the implementation.

---

## The Dataset: House Price Prediction

We use a variant of the Ames Housing dataset. The task is to predict `SalePrice` given features of residential properties.

**Key challenge features:**
- Numeric with skew: `GrLivArea` (gross living area), `LotArea`, `TotalBsmtSF`
- Categorical (nominal): `Neighborhood`, `HouseStyle`, `SaleCondition`
- Categorical (ordinal): `ExterQual`, `KitchenQual`, `OverallQual` (Poor to Excellent)
- Temporal: `YearBuilt`, `YrSold`
- Missing values: many columns have NaN for valid "not applicable" reasons

---

## Part A: Understanding Your Data Before Engineering

The first step is always exploratory data analysis (EDA). Features cannot be engineered without understanding their distributions, relationships, and meaning.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('ames_housing.csv')

# 1. Basic structure
print(df.shape)
print(df.dtypes.value_counts())

# 2. Missing value audit
missing = df.isnull().sum()
missing_pct = missing / len(df) * 100
missing_summary = pd.DataFrame({
    'count':      missing[missing > 0],
    'percentage': missing_pct[missing > 0]
}).sort_values('percentage', ascending=False)
print(missing_summary.head(20))

# 3. Target variable distribution
print(f"Target skewness: {df['SalePrice'].skew():.3f}")
# Typical result: ~1.88 (right-skewed -- expensive houses create a long tail)

# 4. Numeric feature correlations with target
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
correlations = df[numeric_cols].corrwith(df['SalePrice']).abs().sort_values(ascending=False)
print(correlations.head(15))
```

**Key findings from EDA (typical for Ames dataset):**

- `SalePrice` is right-skewed ($\text{skew} \approx 1.88$): a log transformation will make it more Gaussian, which benefits linear models
- `GrLivArea` and `LotArea` are also right-skewed
- ~19 features have missing values; most are missing because the feature is "not applicable" (e.g., `PoolQC` is NaN for houses with no pool)
- Strong correlations with `SalePrice`: `OverallQual` ($r = 0.79$), `GrLivArea` ($r = 0.71$), `GarageArea` ($r = 0.62$)

---

## Part B: Handling Missing Values

Missing values fall into three categories with different treatments:

### Category 1: Structural NA -- "Feature does not apply"

Many NaN values encode the absence of a feature (e.g., no basement, no garage). These are **not random** and should be filled with a domain-appropriate value, not imputed.

```python
# Features where NA means "None/Not present"
# Fill with meaningful string for categoricals, 0 for numerics
none_fill_cats = [
    'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish',
    'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'
]
zero_fill_nums = [
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageArea',
    'GarageCars', 'MasVnrArea'
]

for col in none_fill_cats:
    df[col] = df[col].fillna('None')
for col in zero_fill_nums:
    df[col] = df[col].fillna(0)
```

### Category 2: Random missingness -- imputation appropriate

A small number of features are missing at random (measurement error, data entry gaps).

```python
# LotFrontage: missing ~18% -- impute with neighbourhood median
# Rationale: lot frontage correlates strongly with neighbourhood
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# MSZoning: 4 missing -- fill with mode (most common zoning)
df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])

# Electrical, Functional: 1-2 missing -- fill with mode
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
df['Functional'] = df['Functional'].fillna('Typ')   # domain knowledge default
```

### Category 3: Target-informed imputation (beware of leakage)

Never impute using the target variable in a way that leaks test information. Fit imputers on training data only.

```python
from sklearn.impute import SimpleImputer, KNNImputer

# For pipelines: use sklearn transformers to prevent leakage
# KNN imputer uses k nearest neighbours for numeric features
knn_imputer = KNNImputer(n_neighbors=5)
# Fit ONLY on training data:
X_train_imputed = knn_imputer.fit_transform(X_train_numeric)
X_test_imputed  = knn_imputer.transform(X_test_numeric)  # transform only, no fit
```

**Critical concept: train/test leakage in imputation**

A common mistake is computing the global mean (or median) over the entire dataset and using it to fill missing values. This leaks test information into the imputed training features. The correct approach: compute all statistics on the training set alone and apply to both train and test. Sklearn's `Pipeline` automates this correctly.

---

## Part C: Encoding Categorical Variables

### Ordinal encoding -- order matters

For features with a natural ordering, encode with integers that preserve the ranking:

```python
# Quality scale: None < Po < Fa < TA < Gd < Ex
quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

ordinal_quality_cols = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'
]
for col in ordinal_quality_cols:
    df[col] = df[col].map(quality_map)

# OverallQual and OverallCond are already 1-10 integers -- keep as-is
```

### One-hot encoding -- no natural order

For nominal categoricals with no order, use one-hot encoding (dummy variables):

```python
# Get dummies and drop first level to avoid multicollinearity (dummy trap)
nominal_cols = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
    'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
    'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
    'CentralAir', 'Electrical', 'Functional', 'GarageType', 'GarageFinish',
    'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
]

df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
```

**Warning on high-cardinality categoricals:**

`Neighborhood` has 25 unique values. One-hot encoding creates 24 dummy variables. For a dataset of 1460 rows, many neighbourhoods will have very few examples, making these dummies noisy.

**Alternative: target encoding (mean encoding)**

Replace the category with the mean of the target variable for that category:

```python
# Target encoding -- MUST be done with cross-validation to prevent leakage
from sklearn.model_selection import KFold

def target_encode_cv(df_train, df_test, col, target, n_splits=5, smoothing=10):
    """
    Target encode col using cross-validation on training data.
    smoothing parameter blends category mean towards global mean for rare categories.
    """
    global_mean = df_train[target].mean()
    n_cat       = df_train.groupby(col)[target].count()
    cat_mean    = df_train.groupby(col)[target].mean()
    
    # Smoothing: blends local mean with global mean based on category frequency
    smooth = n_cat / (n_cat + smoothing)
    smoothed_mean = smooth * cat_mean + (1 - smooth) * global_mean
    
    # For test: use smoothed means from full training set
    df_test_encoded = df_test[col].map(smoothed_mean).fillna(global_mean)
    
    # For train: use out-of-fold encoding to prevent leakage
    df_train_encoded = pd.Series(np.zeros(len(df_train)), index=df_train.index)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(df_train):
        fold_mean = df_train.iloc[train_idx].groupby(col)[target].mean()
        df_train_encoded.iloc[val_idx] = df_train.iloc[val_idx][col].map(fold_mean).fillna(global_mean)
    
    return df_train_encoded.values, df_test_encoded.values
```

**Why cross-validation for target encoding**: if you compute the mean of `SalePrice` per `Neighborhood` on the full training set and use that as the feature, the encoding for each training example has already "seen" its own target value. This is a subtle form of target leakage. Cross-validation ensures each training example's encoding is computed without its own label.

---

## Part D: Feature Transformation

### Log-transforming skewed features

Linear models and neural networks benefit from features with approximately Gaussian distributions. The log transform reduces right skew and makes multiplicative relationships additive.

```python
# Check skewness and log-transform highly skewed numeric features
from scipy.stats import skew

numeric_feats = df.select_dtypes(include=np.number).columns.tolist()
feat_skewness = df[numeric_feats].apply(skew).sort_values(ascending=False)

# Apply log1p (log(1+x)) to features with |skew| > 0.75
# log1p handles zero values (no undefined log(0))
skewed_feats = feat_skewness[abs(feat_skewness) > 0.75].index
df[skewed_feats] = np.log1p(df[skewed_feats])

# Also log-transform the target
df['SalePrice'] = np.log1p(df['SalePrice'])
# This converts RMSE on log scale back to RMSLE, which is more interpretable
# for price prediction (penalises relative errors equally across the price range)
```

**Why log-transform the target?**

A $\$10{,}000$ error on a $\$100{,}000$ house ($10\%$ relative error) should be penalised differently from a $\$10{,}000$ error on a $\$1{,}000{,}000$ house ($1\%$ relative error). Standard MSE treats them equally. RMSLE (root mean squared log error) penalises proportional errors equally:

$$\text{RMSLE} = \sqrt{\frac{1}{n}\sum_i (\log(1+\hat{y}_i) - \log(1+y_i))^2}$$

Training on log-transformed targets with MSE loss optimises RMSLE.

### Feature scaling

Always scale features for gradient-based models and distance-based models:

```python
from sklearn.preprocessing import StandardScaler, RobustScaler

# StandardScaler: mean=0, std=1
# Good for: logistic regression, neural networks, SVMs, Ridge
std_scaler = StandardScaler()

# RobustScaler: median=0, IQR=1
# Good for: data with outliers (less affected by extreme values)
rob_scaler = RobustScaler()

# Tree-based models (Random Forest, XGBoost) do NOT require scaling
# They are invariant to monotone feature transformations
```

---

## Part E: Feature Creation

The most impactful feature engineering often involves domain knowledge -- combining raw features into new ones that more directly capture the underlying relationship.

```python
# Total square footage (combines multiple area features)
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

# Total bathrooms (full bath counts twice as much as half bath)
df['TotalBath'] = (df['FullBath'] + df['BsmtFullBath'] +
                   0.5 * (df['HalfBath'] + df['BsmtHalfBath']))

# Age of house at time of sale
df['HouseAge'] = df['YrSold'] - df['YearBuilt']

# Years since last remodel
df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']

# Has basement, garage, pool (binary indicators from numeric)
df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
df['HasGarage']   = (df['GarageArea']  > 0).astype(int)
df['HasPool']     = (df['PoolArea']    > 0).astype(int)

# Overall quality * living area (interaction: quality premium scales with size)
df['QualxArea'] = df['OverallQual'] * df['GrLivArea']

# Neighbourhood price tier (group rare neighbourhoods)
# Compute quartiles of mean sale price by neighbourhood
nbhd_price  = df.groupby('Neighborhood')['SalePrice'].mean()
df['NbhdTier'] = pd.cut(
    df['Neighborhood'].map(nbhd_price),
    bins=4,
    labels=['low', 'med_low', 'med_high', 'high']
)
```

**Validation: verify each new feature adds signal**

```python
from sklearn.feature_selection import mutual_info_regression

# Compare correlation of new features with target
new_features = ['TotalSF', 'TotalBath', 'HouseAge', 'QualxArea']
for feat in new_features:
    corr = df[feat].corr(df['SalePrice'])
    print(f"{feat}: r = {corr:.3f}")

# Typical output:
# TotalSF: r = 0.782   (higher than any individual area feature)
# TotalBath: r = 0.645
# HouseAge: r = -0.559
# QualxArea: r = 0.801 (highest single feature correlation!)
```

---

## Part F: Feature Selection

With engineered features, the total feature count may exceed the original. Pruning uninformative features reduces overfitting, speeds training, and improves interpretability.

### Method 1: Variance threshold

Remove features with near-zero variance -- they contain almost no information:

```python
from sklearn.feature_selection import VarianceThreshold

# Remove features with variance below threshold
# After one-hot encoding, rare dummy variables are common culprits
var_selector = VarianceThreshold(threshold=0.01)
X_reduced = var_selector.fit_transform(X_train)
retained = var_selector.get_support()
print(f"Retained {retained.sum()} of {len(retained)} features")
```

### Method 2: Mutual information

Measures non-linear dependence between each feature and the target:

```python
from sklearn.feature_selection import mutual_info_regression, SelectKBest

mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
mi_frame  = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

# Select top 50 features by MI
selector = SelectKBest(mutual_info_regression, k=50)
X_selected = selector.fit_transform(X_train, y_train)
```

### Method 3: Lasso-based selection

Train a Lasso model and use its non-zero weights as selected features:

```python
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train_log)

lasso_selector = SelectFromModel(lasso_cv, prefit=True)
X_lasso_selected = lasso_selector.transform(X_train_scaled)
print(f"Lasso selected {X_lasso_selected.shape[1]} features")
selected_names = [feature_names[i] for i in lasso_selector.get_support(indices=True)]
```

### Method 4: Feature importance from tree models

Random Forest or Gradient Boosting feature importances:

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=feature_names)
top_features = importances.nlargest(30)
```

**Warning on impurity-based importances**: Scikit-learn's `feature_importances_` use mean decrease in impurity (MDI), which is biased towards high-cardinality features (those with many possible split values). Permutation importance (`permutation_importance`) is more reliable and directly measures the impact on test performance.

```python
from sklearn.inspection import permutation_importance

perm_result = permutation_importance(
    rf, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
)
perm_importance = pd.Series(perm_result.importances_mean, index=feature_names)
```

---

## Part G: Complete Pipeline with Sklearn

A production-grade feature engineering pipeline must:
1. Apply all transformations only on training data (fit) and then apply to test (transform)
2. Be reproducible and serialisable
3. Handle edge cases (unseen categories, missing values at inference time)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

# Define feature groups
numeric_features  = ['GrLivArea', 'LotArea', 'TotalBsmtSF', 'GarageArea',
                     'TotalSF', 'TotalBath', 'HouseAge', 'QualxArea']
ordinal_features  = ['ExterQual', 'KitchenQual', 'BsmtQual', 'OverallQual']
nominal_features  = ['Neighborhood', 'HouseStyle', 'SaleCondition']

# Preprocessing for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('log',     FunctionTransformer(np.log1p)),   # custom transformer
    ('scaler',  StandardScaler())
])

# Preprocessing for ordinal features
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# Preprocessing for nominal features
nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
])

# Combine all transformers
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer,  numeric_features),
    ('ord', ordinal_transformer,  ordinal_features),
    ('nom', nominal_transformer,  nominal_features)
])

# Full pipeline including model
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model',        GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                               max_depth=4, random_state=42))
])

# Fit on training data -- ALL transformations are learned from training data only
full_pipeline.fit(X_train_raw, y_train_log)

# Evaluate
from sklearn.metrics import mean_squared_error
y_pred_log = full_pipeline.predict(X_test_raw)
rmsle = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
print(f"RMSLE: {rmsle:.5f}")
```

---

## Part H: Interview Questions and Model Answers

### Q1. What is target leakage and how can feature engineering introduce it?

**Answer:**

Target leakage occurs when information about the target variable $y$ is encoded in a feature $x_j$ at training time in a way that will not be available at inference time (or that uses future information). The model learns a spurious, non-causal relationship.

**Feature engineering sources of leakage:**

1. **Imputation with test data statistics**: computing the mean of a feature over both train and test data and using it to fill missing values allows test labels to influence training imputation.

2. **Target encoding without cross-validation**: replacing a categorical feature with its mean target value, computed on the full training set, means each training example's feature value incorporates its own label.

3. **Computing aggregations that include the current row**: creating a feature "average house price in this neighbourhood" using the same dataset on which you are training.

4. **Temporal features with future information**: in time-series data, creating a 7-day rolling average that includes the current and future days for a training example that is in the past.

5. **Normalisation using test statistics**: fitting a `StandardScaler` on train + test and then transforming both. The scaler learns the mean and std of the combined set, which contains test labels (implicitly through the feature distribution).

**Detection**: if a new feature has suspiciously high predictive power (mutual information or correlation), investigate whether it contains target information.

---

### Q2. A feature has $30\%$ missing values. Describe three different imputation strategies and when each is appropriate.

**Answer:**

**Strategy 1: Domain-informed constant fill**

If missing values have a known meaning ("not applicable"), fill with a domain-appropriate constant. For `PoolQC`, missing means "no pool" -- fill with `None` or `0`.

- When to use: when missingness is **not random** (Missing Not At Random, MNAR) and has a known interpretation
- Advantage: preserves the semantic meaning; does not introduce bias from statistical imputation
- Risk: if some true missing values exist (data entry error), they are masked

**Strategy 2: Model-based imputation (KNN or iterative)**

Impute using the $k$ most similar examples (measured by other features). `KNNImputer` fills missing values with the mean of the $k$ nearest neighbours. `IterativeImputer` models each feature as a function of all others in a round-robin fashion.

- When to use: when missingness is **at random** (MAR) -- missing values correlate with other observed features but not with the missing value itself
- Advantage: uses feature relationships; more accurate than global mean
- Risk: computationally expensive; requires fitting on training data; can introduce bias if the imputation model is misspecified

**Strategy 3: Missing indicator + mean imputation**

Add a binary indicator feature `feature_j_was_missing` (1 if missing, 0 otherwise), then fill the original with the column mean. This allows the model to learn a different relationship for examples where the value was observed vs. missing.

- When to use: when the fact of missingness is itself predictive (e.g., expensive houses may report more features); when you want the model to distinguish "was missing" from "has a small value"
- Advantage: preserves missingness signal; robust; simple
- Risk: doubles the feature count for each affected column; the model must be complex enough to use the indicator

---

### Q3. You create a feature "OverallQual $\times$ GrLivArea" (quality $\times$ area). How do you decide if this interaction term adds value?

**Answer:**

An interaction feature is valuable if the effect of one variable on the target depends on the value of the other. For house prices, the premium for higher quality should scale with house size (a larger high-quality house is worth more than a smaller high-quality house -- the quality "multiplies" the value of the area).

**Evaluation steps:**

**1. Correlation analysis**: compute the correlation of `QualxArea` with `SalePrice` and compare to the individual correlations of `OverallQual` and `GrLivArea`. If the interaction is higher, it may add signal.

**2. Partial regression plot**: regress out `OverallQual` and `GrLivArea` from both the feature and target, then plot the residuals. A clear trend in the residual vs. `QualxArea_residual` confirms the interaction effect.

**3. Model comparison**: train two models -- one with and one without the interaction term -- using cross-validated RMSE. If adding the interaction term improves CV RMSE beyond the noise level (confidence interval of the estimate), include it.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

X_without_interaction = X_baseline
X_with_interaction    = np.column_stack([X_baseline, df['QualxArea']])

scores_without = cross_val_score(Ridge(alpha=1.0), X_without_interaction, y,
                                  cv=5, scoring='neg_root_mean_squared_error')
scores_with    = cross_val_score(Ridge(alpha=1.0), X_with_interaction, y,
                                  cv=5, scoring='neg_root_mean_squared_error')

print(f"Without interaction: {-scores_without.mean():.5f} ± {scores_without.std():.5f}")
print(f"With interaction:    {-scores_with.mean():.5f} ± {scores_with.std():.5f}")
# If the improvement exceeds one standard deviation, the term is valuable
```

**4. Permutation importance**: after including the term, measure its permutation importance on a validation set. Importance near zero means it did not help the model.

**5. Redundancy check**: if `QualxArea` is highly correlated with `OverallQual` alone (e.g., $r > 0.95$), it adds little new information and may cause multicollinearity in linear models.

---

## Summary: Feature Engineering Checklist

```
1. EDA
   [ ] Check distributions (histograms, box plots)
   [ ] Identify skewed features (skewness > 0.75: consider log transform)
   [ ] Audit missing values (count, percentage, MCAR/MAR/MNAR classification)
   [ ] Check target distribution (log-transform if skewed)

2. Missingness
   [ ] Structural NA (domain meaning): fill with constant
   [ ] Random missingness: impute (median/KNN/iterative) + consider indicator
   [ ] Always fit imputers on training data only

3. Encoding
   [ ] Ordinal: integer encoding preserving order
   [ ] Nominal (low cardinality): one-hot encoding
   [ ] Nominal (high cardinality): target encoding with CV, or embedding

4. Transformation
   [ ] Log-transform skewed numeric features
   [ ] Scale numeric features (StandardScaler for gradient-based; not needed for trees)

5. Feature creation
   [ ] Sum relevant components (TotalSF, TotalBath)
   [ ] Temporal features (age, time since event)
   [ ] Ratios (price per sqft, quality per age)
   [ ] Interaction terms (quality × area)
   [ ] Binary indicators (has pool, has garage)

6. Feature selection
   [ ] Remove near-zero variance features
   [ ] Compare mutual information / Lasso selection / permutation importance
   [ ] Validate each added feature improves CV score

7. Pipeline
   [ ] Use sklearn Pipeline to prevent leakage
   [ ] Fit only on training data; transform test data
   [ ] Serialise the fitted pipeline for deployment
```
