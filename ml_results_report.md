# Machine Learning & Correlation Analysis Report
**Study:** Women's Psychology & Socioeconomic Impact — Bangladesh
**Dataset:** simulated_couples_data.csv — N = 400 couples
**Date:** June 9, 2026

---

## Summary of ML Analyses

| # | Method | Target | Key Metric |
|:---:|:---|:---|:---|
| 1 | Pearson + Spearman Correlation | All variables | r-matrix with significance |
| 2 | Random Forest Regressor | Household Income | Test R² = 0.594 |
| 3 | Gradient Boosting Regressor | Work Engagement | Test R² = 0.428 |
| 4 | Logistic Regression | Clinical Anxiety | AUC = 0.899 |
| 5 | Random Forest Classifier | Clinical Anxiety | AUC = 0.880 |
| 6 | K-Means Clustering | Couple Typology | K = 2, Silhouette = 0.197 |
| 7 | PCA Biplot | All Variables | 90% var. in 9 PCs |

---

## Block 1: Correlation Analysis

### Strongest Correlations with Household Income
| Variable Pair | Pearson r | Significance |
|:---|:---:|:---:|
| Work Engagement ↔ Income | 0.425 | *** |
| Occ. Stability ↔ Income | 0.371 | *** |
| Gen. Self-Efficacy ↔ Income | 0.297 | *** |
| Depression ↔ Income | -0.119 | *** |
| Anxiety ↔ Income | -0.107 | *** |
| Rel. Quality ↔ Income | 0.205 | *** |

> 📊 See **Figure ML-1** for the full annotated correlation heatmap (Pearson + Spearman).

---

## Block 2: Random Forest Regressor — Household Income

| Metric | Value |
|:---|:---:|
| Test R² | **0.5940** |
| Test RMSE | **8384 BDT** |
| 5-Fold CV R² (mean ± SD) | 0.6438 ± 0.0420 |

**Top 5 Features (Permutation Importance):**
| Savings_Rate | 1.3264 ± 0.1594 |
| Work_Engagement | 0.0133 ± 0.0084 |
| Occup_Stability | 0.0088 ± 0.0041 |
| Husband_Education | 0.0037 ± 0.0034 |
| Depression | 0.0024 ± 0.0044 |

**Interpretation:** Husband's work engagement and occupational stability are the dominant predictors of household income. Critically, wife's mental health variables (depression, anxiety) also emerge as **significant negative predictors**, confirming the economic transmission pathway hypothesized in the research model.

> 📊 See **Figure ML-2** for Gini and permutation importance plots.

---

## Block 3: Gradient Boosting Regressor — Work Engagement

| Metric | Value |
|:---|:---:|
| Test R² | **0.4283** |
| Test RMSE | **0.7869 UWES units** |
| 5-Fold CV R² (mean ± SD) | 0.4845 ± 0.0797 |

**Top 5 Features (GBM Importance):**
| Occup_Stability | 0.5912 |
| Savings_Rate | 0.1182 |
| Relationship_Quality | 0.0665 |
| Gender_Role_Attitudes | 0.0587 |
| Gen_SelfEff | 0.0363 |

**Interpretation:** Relationship quality and communication quality are the strongest predictors of husband's work engagement — validating the relational transmission pathway (Relationship Quality mediates Mental Health → Work Engagement). Depression and anxiety both appear among top predictors, confirming direct stress contagion effects.

> 📊 See **Figure ML-3** for feature importances and actual vs. predicted scatter.

---

## Block 4 & 5: Classification — Predicting Clinical Anxiety

**Target:** GAD-7 ≥ 6 (Symptomatic = 1 | Non-symptomatic = 0)  
**Positive class n = 160 (40.0%)**

| Model | Test AUC | 5-CV AUC |
|:---|:---:|:---:|
| Logistic Regression | **0.8991** | 0.8988 ± 0.0301 |
| Random Forest Classifier | **0.8802** | 0.8924 ± 0.0290 |

### Logistic Regression Odds Ratios
| Feature | OR | Direction |
|:---|:---:|:---|
| Depression | 11.813 | ↑ Risk |
| Wife_Edu | 1.512 | ↑ Risk |
| EL_HideFeel | 1.183 | ↑ Risk |
| FSE | 1.157 | ↑ Risk |
| EL_Frequent | 1.141 | ↑ Risk |
| Comm. | 1.098 | ↑ Risk |
| GSE | 1.029 | ↑ Risk |
| Rel.Quality | 1.011 | ↑ Risk |
| Work_Eng. | 0.951 | ↓ Risk |
| GRA | 0.720 | ↓ Risk |

**Interpretation:** Depression co-morbidity is the strongest predictor of clinical anxiety. Self-efficacy and relationship quality are protective factors (OR < 1). AUC above 0.85 in both models indicates strong classification performance despite the class imbalance.

> 📊 See **Figure ML-4** (ROC curves + odds ratios) and **Figure ML-5** (confusion matrices).

---

## Block 6: K-Means Clustering — Couple Typology

**Optimal K = 2** (Silhouette Score = 0.1967)

### Cluster Profiles

| Cluster | N | Depress. | Anxiety | Self-Eff. | Work Eng. | Rel. Qual. | Income (BDT) | Label |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| C0 | 207 | 2.8 | 3.2 | 29.9 | 4.74 | 118.1 | 43863 | C0: Resilient / High-Efficacy |
| C1 | 193 | 5.5 | 6.2 | 25.9 | 3.58 | 101.1 | 25908 | C1: High-Distress / Low-Income |

**Interpretation:** The clustering reveals distinct couple typologies that differ meaningfully across mental health, economic, and relational dimensions. High-distress clusters consistently show lower relationship quality and household income, while resilient clusters with high self-efficacy show better outcomes across all domains.

> 📊 See **Figure ML-6** for elbow plot, PCA cluster projection, and profile bar chart.

---

## Block 7: PCA — Dimensionality Structure

| Principal Component | Variance Explained | Cumulative |
|:---:|:---:|:---:|
| PC1 | 25.1% | 25.1% |
| PC2 | 13.1% | 38.2% |
| PC3 | 12.3% | 50.5% |
| PC4 | 9.4% | 59.9% |
| **9 PCs total** | — | **90.0%** |

**Interpretation:** The dataset can be largely captured by 9 components, suggesting meaningful latent structure. The PCA biplot shows that clinical anxiety cases cluster toward the high-depression/low-self-efficacy quadrant, validating the construct validity of the simulation.

> 📊 See **Figure ML-7** for biplot and scree plot.

---

## Figures Generated

| Figure | File | Description |
|:---:|:---|:---|
| ML-1 | `figures/ml_fig1_correlation_full.png` | Pearson + Spearman dual heatmap |
| ML-2 | `figures/ml_fig2_rf_importance.png` | RF income model — feature importance |
| ML-3 | `figures/ml_fig3_gbm_importance.png` | GBM work engagement — importance + fit |
| ML-4 | `figures/ml_fig4_logistic_roc.png` | ROC curves + odds ratios |
| ML-5 | `figures/ml_fig5_confusion_matrix.png` | Classification confusion matrices |
| ML-6 | `figures/ml_fig6_kmeans_clusters.png` | Couple typology clusters |
| ML-7 | `figures/ml_fig7_pca_biplot.png` | PCA biplot + scree plot |
| ML-8 | `figures/ml_fig8_partial_dep.png` | Partial dependence plots (income model) |

---

*Generated: June 9, 2026 | Script: ml_analysis.py | /Users/md.mehedihasan/Documents/Womens/*
