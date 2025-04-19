# Regression Analysis Methodology

## Overview
This document outlines the methodology used for regression analysis to identify demographic factors associated with GAA club performance in Cork. The analysis implements both Ordinary Least Squares (OLS) regression and Geographically Weighted Regression (GWR) to account for spatial heterogeneity.

## Data Preparation

### Variable Selection
The regression analysis included the following categories of variables:

1. **Dependent Variables:**
   - Overall club performance score
   - Football-specific performance score
   - Hurling-specific performance score

2. **Demographic Independent Variables:**
   - Youth population metrics (percentage ages 0-4, 5-12, 13-18)
   - Education levels (third level, secondary, primary education rates)
   - Employment indicators (employment rate, labor force participation)
   - Social class distribution (professional, managerial, skilled, semi-skilled)
   - Housing characteristics (ownership rates, household sizes)
   - Age structure (dependency ratio, average age)

3. **Spatial Control Variables:**
   - Population density
   - Distance to nearest urban center
   - Urban/rural classification
   - Competition density (clubs within 10km)

### Variable Transformation
- Log transformation for population density
- Standardization of all variables (Z-scores)
- Ratio transformations for demographic percentages
- Creation of composite indices where appropriate

### Multicollinearity Treatment
- Variance Inflation Factor (VIF) analysis
- Correlation matrix assessment
- Stepwise variable removal (threshold VIF > 10)
- Principal Component Analysis for highly correlated groups

## Modeling Approach

### Model Selection Strategy
1. **Baseline OLS Models**
   - Simple linear regression for initial assessment
   - Multiple linear regression with demographic variables
   - Full model with demographic and spatial variables

2. **Stepwise Model Building**
   - Forward selection based on AIC
   - Backward elimination based on p-values
   - LASSO regression for variable selection

3. **Spatial Regression Models**
   - Spatial lag models (SLM)
   - Spatial error models (SEM)
   - Geographically Weighted Regression (GWR)

### Model Validation
- K-fold cross-validation (k=5)
- Leave-one-out cross-validation for spatial models
- Monte Carlo significance testing
- Spatial autocorrelation assessment of residuals

## OLS Regression Implementation

### Model Specification
The general form of the OLS model was:

$Y_i = \beta_0 + \sum_{j=1}^{p} \beta_j X_{ij} + \varepsilon_i$

Where:
- $Y_i$ is the performance score for club $i$
- $X_{ij}$ is the value of predictor $j$ for club $i$
- $\beta_0$ is the intercept
- $\beta_j$ is the coefficient for predictor $j$
- $\varepsilon_i$ is the error term for club $i$

### Implementation Details
- Software: Python with statsmodels package
- Significance level: α = 0.05
- Robust standard errors (HC3)
- Outlier detection and treatment

### Model Assessment
- R² and Adjusted R²
- F-statistic and p-value
- AIC and BIC values
- Residual diagnostics (normality, homoscedasticity)

## GWR Implementation

### Model Specification
The GWR model takes the form:

$Y_i = \beta_0(u_i, v_i) + \sum_{j=1}^{p} \beta_j(u_i, v_i) X_{ij} + \varepsilon_i$

Where:
- $(u_i, v_i)$ represents the spatial coordinates of club $i$
- $\beta_j(u_i, v_i)$ is the spatially varying coefficient

### Implementation Details
- Software: Python with mgwr package
- Kernel function: Adaptive bi-square
- Bandwidth selection: Optimal AICc
- Coordinate reference system: EPSG:2157 (ITM)

### Bandwidth Optimization
- Golden section search algorithm
- Cross-validation approach
- AICc minimization
- F1 optimization for mixed GWR

### Model Assessment
- Local R² values
- Global AICc
- Moran's I of residuals
- Monte Carlo significance tests

## Key Findings

### OLS Results
1. **Youth Population**
   - Significant positive effect on overall performance
   - Stronger effect for football than hurling
   - Non-linear relationship with optimal ranges

2. **Education Levels**
   - Third level education consistently significant
   - Positive association with overall performance
   - Stronger effect in urban areas

3. **Social Class**
   - Professional and managerial classes positively associated
   - Mixed effects for other social classes
   - Interaction with urban/rural context

4. **Spatial Factors**
   - Population density significant but non-linear
   - Urban proximity positive for hurling performance
   - Competition density negative for lower-grade clubs

### GWR Results
1. **Spatial Variation**
   - Significant spatial non-stationarity in relationships
   - Local R² values ranging from 0.42 to 0.78
   - Higher model fit in urban areas

2. **Parameter Variation**
   - Youth population effect strongest in suburban areas
   - Education effect strongest in urban areas
   - Social class effects vary significantly across space

3. **Model Comparison**
   - GWR outperforms OLS (lower AICc)
   - Significant reduction in spatial autocorrelation of residuals
   - Better prediction accuracy in cross-validation

## Limitations and Considerations
1. **Temporal Limitations**
   - Cross-sectional analysis (2022 data)
   - Cannot establish causality
   - Historical patterns not captured

2. **Scale Issues**
   - Modifiable Areal Unit Problem (MAUP)
   - Catchment area definition affects results
   - Edge effects at study area boundaries

3. **Unmeasured Factors**
   - Club history and tradition
   - Internal club governance
   - Infrastructure quality
   - Volunteer engagement

4. **Statistical Considerations**
   - Multiple testing issues
   - Ecological fallacy
   - Sample size limitations for GWR 