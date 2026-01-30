# Country predictors
The data is an open source dataset from statistic canada. Namly contry indicators as predictors and test prediction as the response variable. 

## Abstract
The main purpose of this file is the comparison between the three models(ffnn\XGboost\Transformer), but the first three quarter of the file is the process of cleaning the data, using the function one-hot, fitting the model and shrinking the parameters by manual backwards selection. 
The reason for why this is, is because I didn't really know what I was truly looking for at the time. It would be adhearent that the goal of the file is not to train and improve the models' performence even if there are clearly backward selection and ANOVA test done on the model, 
It is to understand, explain and express where each model structure fails at and why. 

below is a generated steps of what I think i had done, since i believe that it can explain it better than me at this moment. 
(note, this will not be a recurring event as the other projects are more recent and this is quite long ago)
## What This Script Does

### 1. Integrates model predictions with country indicators
- Loads precomputed probabilistic predictions and ground-truth labels from **Transformer**, **FFNN**, and **XGBoost** models.
- Merges these predictions with country-level indicator data using **ISO-3 country codes**, providing geographic and contextual information for analysis.

### 2. Defines a unified error metric
- Computes **absolute prediction error** for each model as:

  \[
  \left| y_{\text{true}} - y_{\text{predicted probability}} \right|
  \]

- This metric enables direct, apples-to-apples comparison of **probabilistic performance** across different model families.

### 3. Builds a stacked analytical dataset
- Combines predictions from all three model families into a **single design matrix**.
- Adds indicator variables for:
  - model type,
  - prediction outcome (0 vs 1),
  - geographic region.
- Constructs **interaction terms** to capture how model performance varies by region and prediction type.

### 4. Explains model behavior using statistical regression
- Fits multiple **OLS regression models** with prediction error as the response variable.
- Evaluates which factors (model family, region, prediction outcome, and interactions) are associated with higher or lower error.
- Iteratively simplifies models using **significance thresholds and diagnostic checks** to improve interpretability.

### 5. Validates robustness with train–test evaluation
- Uses held-out **train–test splits** to compute RMSE on prediction errors.
- Ensures observed performance patterns are not driven by overfitting or random noise.
