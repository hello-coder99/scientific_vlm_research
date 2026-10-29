### Statistical Validation: Lie Detector Performance

| Metric | Empirical Value | Description |
| :--- | :---: | :--- |
| **Total Test Claims** | **75** | Balanced evaluation set (50 clean, 50 hallucinated) |
| **True Positives (TP)** | **46** | Hallucinations correctly detected and flagged |
| **True Negatives (TN)** | **25** | Clean veridical claims confirmed as SUPPORTED |
| **False Positives (FP)** | **0** | Clean claims falsely accused of hallucination |
| **False Negatives (FN)** | **4** | Hallucinations that went undetected |
| **Precision** | **100.00%** | Positive Predictive Value: TP / (TP + FP) |
| **Recall / Sensitivity** | **92.00%** | True Positive Rate: TP / (TP + FN) |
| **Specificity** | **100.00%** | True Negative Rate: TN / (TN + FP) |
| **F1-Score** | **95.83%** | Harmonic mean of Precision and Recall |
| **Overall Classification Accuracy** | **94.67%** | Overall empirical correctness |

#### Category-Wise Hallucination Detection Rates:
- **Numerical Drift (5% to 15%)**: 78.6% caught
- **Extreme Hallucinations (30% to 60%)**: 100.0% caught
- **Sign Inversions (-y to +y)**: 100.0% caught
- **Fabricated Points (Invented x)**: 92.3% caught
