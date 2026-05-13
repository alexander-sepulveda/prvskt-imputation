This repository contains the implementation and comparative analysis of various data imputation techniques applied to Perovskite Solar Cell datasets. The project focuses on recovering missing values while preserving the multivariate statistical structure of the data.

The recent surge in publications regarding perovskite solar cells provides valuable data about their synthesis and performance. This study presents a systematic comparative analysis of imputation methods specifically tailored to PSC data. We successfully model domain-specific missingness mechanisms and evaluate algorithms using point-wise error metrics ($NRMSE$, $NMAE$) alongside advanced probabilistic distances. Our findings demonstrate that ensemble-based methods, specifically MissForest with Extremely Randomized Trees, offer the optimal balance between accuracy and distribution preservation.

Main Libraries Used:
- Imputation: scikit-learn (IterativeImputer)
- Regressors: ExtraTrees, RandomForest, BayesianRidge, KNeighbors
- Metrics: Optimal Transport (ot), Distance Correlation (dcor), scipy.stats
