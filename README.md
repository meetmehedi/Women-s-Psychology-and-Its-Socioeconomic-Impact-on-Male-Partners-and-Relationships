# Women's Psychology and Its Socioeconomic Impact on Male Partners and Relationships

> **An Interdisciplinary Quantitative Research Project | Bangladesh Context**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

This research project investigates how the **psychological well-being of married women** in Bangladesh — including depression, anxiety, self-efficacy, and emotional labor — transmits into their **male partners' work engagement** and **household socioeconomic outcomes**.

The study combines:
- 📄 National survey data from **BDHS 2022** (N = 19,987 ever-married women)
- 📚 Peer-reviewed literature (4 key 2024–2025 papers from Europe PMC)
- 🧪 Simulated dyadic dataset of **400 couples (N = 800 individuals)**
- 📊 Full statistical analysis: SEM, OLS regression, moderated mediation, ML models

---

## 🔬 Research Questions

1. How does wife's mental health (depression/anxiety) predict husband's work engagement?
2. Does relationship quality mediate this pathway?
3. Does women's self-efficacy independently predict household economic outcomes?
4. Does gender role attitudes moderate the anxiety → relationship quality → work engagement pathway?

---

## 📁 Repository Structure

```
├── data.md                           # Data sources, literature links & legal disclaimer
├── womens_psychology_research_proposal.md  # Full research proposal (APA format)
├── measurement_toolkit.md            # Survey instruments (PHQ-9, GAD-7, ECR-R, etc.)
├── manuscript_draft.md               # Full academic manuscript draft
│
├── simulated_couples_data.csv        # Dyadic dataset: 400 couples, 15 variables
├── bdhs_mental_health_baseline.json  # BDHS 2022 extracted baseline statistics
│
├── sem_simulation.py                 # SEM simulation script (semopy)
├── full_analysis.py                  # Full statistical analysis (OLS, correlations, t-tests)
├── ml_analysis.py                    # ML analysis (RF, GBM, Logistic, K-Means, PCA)
│
├── sem_analysis_report.md            # SEM path results & model fit
├── moderated_mediation_report.md     # PROCESS Model 7 moderated mediation
├── findings_report.md                # Full statistical findings report
├── ml_results_report.md              # Machine learning results report
│
├── figures/                          # All generated charts (14 PNG files)
│   ├── fig1_prevalence.png
│   ├── fig2_correlation_heatmap.png
│   ├── fig3_regression_forest.png
│   ├── fig4_group_comparison.png
│   ├── fig5_moderated_mediation.png
│   ├── fig6_distributions.png
│   ├── ml_fig1_correlation_full.png
│   ├── ml_fig2_rf_importance.png
│   ├── ml_fig3_gbm_importance.png
│   ├── ml_fig4_logistic_roc.png
│   ├── ml_fig5_confusion_matrix.png
│   ├── ml_fig6_kmeans_clusters.png
│   ├── ml_fig7_pca_biplot.png
│   └── ml_fig8_partial_dep.png
```

---

## 📊 Key Findings

### Structural Equation Modeling (SEM)
| Hypothesis | Path | β | p | Supported |
|:---:|:---|:---:|:---:|:---:|
| H1 | Wife's Mental Health → Relationship Quality | −0.475 | <.001 | ✅ |
| H2 | Wife's Mental Health → Work Engagement (direct) | −0.135 | .035 | ✅ |
| H3 | Self-Efficacy → Household Economy | +0.248 | <.001 | ✅ |
| H4 | Work Engagement → Household Economy | +0.455 | <.001 | ✅ |
| H5 | Indirect effect stronger in traditional households | Δβ = 0.015 | sig | ✅ |

**Model Fit:** CFI = 0.996 | RMSEA = 0.021 | TLI = 0.995 ✅ Excellent

### Machine Learning Results
| Model | Target | Performance |
|:---|:---|:---:|
| Random Forest Regressor | Household Income | R² = 0.594 |
| Gradient Boosting Regressor | Work Engagement | R² = 0.428 |
| Logistic Regression | Clinical Anxiety | AUC = 0.899 |
| Random Forest Classifier | Clinical Anxiety | AUC = 0.880 |
| K-Means (K=2) | Couple Typology | Silhouette = 0.197 |

### Cluster Typology (K-Means)
| Cluster | Type | Anxiety | Rel. Quality | Income (BDT/mo.) |
|:---:|:---|:---:|:---:|:---:|
| C0 | Resilient / High-Efficacy | 3.21 | 118 | 43,863 |
| C1 | High-Distress / Low-Income | 6.16 | 101 | 25,908 |

> **BDT 17,955/month income difference** between resilient and distressed couple clusters.

---

## 🛠️ Setup & Run

### Requirements
```bash
uv run --with matplotlib --with seaborn --with scikit-learn --with pandas --with numpy --with scipy ml_analysis.py
```

### Scripts
```bash
# 1. Generate simulated data + SEM
uv run --with semopy --with pandas --with numpy --with scipy sem_simulation.py

# 2. Full statistical analysis (OLS, correlations, moderated mediation)
uv run --with matplotlib --with seaborn --with scikit-learn --with pandas --with numpy --with scipy full_analysis.py

# 3. Machine learning + correlation analysis
uv run --with matplotlib --with seaborn --with scikit-learn --with pandas --with numpy --with scipy ml_analysis.py
```

---

## 📚 Data Sources

1. **BDHS 2022** — Bangladesh Demographic and Health Survey (NIPORT & ICF, 2023)
2. Islam et al. (2025) — Marital control, domestic violence & mental health, BMC Psychiatry
3. Akter et al. (2025) — Determinants of help-seeking, PLOS Mental Health
4. Hossain et al. (2024) — Climate change & women's mental health, PLOS Global Public Health
5. Rahman et al. (2025) — Women's mental health help-seeking behavior, Frontiers in GWH

> See [`data.md`](data.md) for full citations and Europe PMC links.

---

## 📖 Citation

```
[Author Name(s)] (2026). Women's Psychological Well-Being and Its Dyadic
Socioeconomic Transmission: An Empirical Investigation Among Couples in Bangladesh.
[University/Research Institute]. GitHub: https://github.com/meetmehedi/Women-s-Psychology-and-Its-Socioeconomic-Impact-on-Male-Partners-and-Relationships
```

---

## ⚖️ License

MIT License — see [LICENSE](LICENSE) for details.

---

*Research generated: June 2026 | Tools: Python 3.10+, semopy, scikit-learn, matplotlib, seaborn*
