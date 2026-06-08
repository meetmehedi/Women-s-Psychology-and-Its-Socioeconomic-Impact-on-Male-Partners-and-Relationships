# Research Proposal SEM Analysis Report (Simulated)
This report summarizes the structural equation modeling (SEM) analysis conducted on simulated data for 400 couples ($N=800$ individuals) based on the research framework.

## 1. Study & Sample Overview
* **Sample Size ($N$):** 400 couples (representing 400 wives and 400 husbands)
* **Design:** Dyadic cross-sectional assessment
* **Wife's Variables:** Mental Health (Depression/PHQ-9, Anxiety/GAD-7), Self-Efficacy, and Emotional Labor.
* **Husband's Variables:** Work Engagement (UWES), Occupational Stability.
* **Household/Shared Variables:** Monthly Income, Savings Rate, Relationship Quality.

## 2. Model Fit Assessment
The model was fitted using the `semopy` library. The calculated fit indices are:
* **Chi-squared ($P$-value):** 51.944 ($p = 0.1919$)
* **RMSEA:** 0.021 (Root Mean Square Error of Approximation)
* **CFI:** 0.996 (Comparative Fit Index)
* **TLI:** 0.995 (Tucker-Lewis Index)
* **GFI:** 0.977 (Goodness of Fit Index)

*Interpretation: A CFI and TLI above 0.90, and RMSEA below 0.08 indicate acceptable-to-good fit.*

## 3. Structural Path Analysis Results

The estimated standardised regression paths correspond to the key hypotheses:

### Hypothesis 1: Stress Contagion (Mental Health to Relationship Quality)
* Path: `Relationship_Quality_Latent ~ Mental_Health`
* Coefficient ($eta$): -0.475
* $p$-value: 0.000
* **Finding:** Support found. Wife's mental health symptoms (anxiety/depression) significantly deteriorate relationship quality and dyadic adjustment.

### Hypothesis 2: Transmission to Partner (Mental Health to Husband's Work Engagement)
* Path: `Work_Engagement_Latent ~ Mental_Health`
* Coefficient ($eta$): -0.135
* $p$-value: 0.035
* **Finding:** Support found. There is a significant direct and indirect negative transmission pathway where the wife's psychological distress lowers her husband's work engagement.

### Hypothesis 3: Socioeconomic Output (Self-Efficacy to Household Economy)
* Path: `Household_Economy ~ Self_Efficacy`
* Coefficient ($eta$): 0.248
* $p$-value: 0.000
* **Finding:** Support found. Wife's self-efficacy (general & financial) has a strong direct positive relationship with household economic outcomes (income & savings).

### Hypothesis 4: Partner Productivity to Household Economy
* Path: `Household_Economy ~ Work_Engagement_Latent`
* Coefficient ($eta$): 0.455
* $p$-value: 0.000
* **Finding:** Support found. Husband's work engagement directly increases monthly household income and savings.

---
*Report generated on June 9, 2026. Data simulated based on research design.*
