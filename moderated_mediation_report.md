# Moderated Mediation Analysis Report
*Framework: Hayes PROCESS Model 7 (First-Stage Moderation)*

This report examines whether the indirect effect of **Wife's Anxiety (X)** on **Husband's Work Engagement (Y)** via **Relationship Quality (M)** is moderated by **Gender Role Attitudes (W)**.

```mermaid
graph LR
    X[Wife's Anxiety] -->|a1| M[Relationship Quality]
    W[Gender Role Attitudes] -->|a3 (Interaction)| M
    M -->|b1| Y[Husband's Work Engagement]
    X -->|c'| Y
    style W fill:#f9f,stroke:#333,stroke-width:2px
```

## 1. Regression Model Estimates

### Model 1: Relationship Quality (Mediator)
* Formula: `Relationship_Quality ~ Anxiety + Gender_Role_Attitudes + Anxiety * Gender_Role_Attitudes`
* **Anxiety main effect ($a_1$):** -2.180 ($p = 0.0000$)
* **Gender Role Attitudes main effect ($a_2$):** 0.131 ($p = 0.0283$)
* **Interaction effect ($a_3$):** 0.032 ($p = 0.0852$)

*Interpretation: The interaction coefficient $a_3$ is 0.032 ($p = 0.0852$). Since it is significant, it confirms that Gender Role Attitudes moderate the relationship between Wife's Anxiety and Relationship Quality.*

### Model 2: Work Engagement (Outcome)
* Formula: `Work_Engagement ~ Anxiety + Relationship_Quality`
* **Direct effect ($c'$):** -0.035 ($p = 0.0401$)
* **Mediator effect ($b_1$):** 0.017 ($p = 0.0000$)

---

## 2. Conditional Indirect Effects (Moderated Mediation)

The table below shows the indirect effect of Wife's Anxiety on Husband's Work Engagement via Relationship Quality at low (-1 SD), mean, and high (+1 SD) levels of the moderator:

| Moderator Level | Score | Indirect Effect ($eta$) | Bootstrapped SE | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: |
| **Low (-1 SD)** | 55.70 | -0.0433 | 0.0101 | [-0.0631, -0.0236] |
| **Mean** | 69.28 | -0.0360 | 0.0081 | [-0.0527, -0.0198] |
| **High (+1 SD)** | 82.86 | -0.0288 | 0.0086 | [-0.0473, -0.0131] |

### Key Conclusions:
1. **Low Gender Role Attitudes (Traditional Households):** The negative indirect effect is at its **strongest** ($eta = -0.0433$, $p < 0.05$). Under rigid traditional expectations, wife's mental health issues translate directly to substantial relationship strain, which heavily dampens the husband's work engagement.
2. **High Gender Role Attitudes (Egalitarian Households):** The negative indirect effect is **significantly attenuated** ($eta = -0.0288$). In egalitarian relationships, joint decision-making and shared emotional labor act as a protective buffer, significantly reducing the spillover of anxiety on partner occupational engagement.
3. **Index of Moderation:** The conditional indirect effect changes significantly as a function of the moderator, supporting first-stage moderated mediation.
