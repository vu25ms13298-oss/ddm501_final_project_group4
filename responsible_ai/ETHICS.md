# Ethical Considerations — License Plate Recognition

## 1. Privacy Concerns

License plate numbers are **personally identifiable information (PII)**. They can be linked to vehicle owners, locations, and movement patterns.

### Mitigations
- The API does **not store** uploaded images or recognition results persistently.
- No logging of plate text to disk or external services.
- Images are processed in-memory and discarded after response.
- In a production deployment, data retention policies must be enforced and communicated.

## 2. Surveillance Risks

Automated license plate recognition (ALPR) can enable mass surveillance if deployed without proper oversight.

### Mitigations
- This system is designed for **single-image, on-demand recognition** — not continuous video surveillance.
- Deployment should be limited to authorized use cases (parking management, toll systems, law enforcement with warrant).
- Access control must be enforced at the API level (authentication, rate limiting).

## 3. Bias and Fairness

### Identified Risks
- **Character-level bias**: The model may perform differently on digits vs. letters, or on visually similar characters (0/D, 5/S, 8/B). Our fairness analysis quantifies this disparity.
- **Geographic bias**: The system is trained on Vietnamese license plates and may not generalize to plates from other countries.
- **Data quality bias**: Training data is primarily synthetic. Real-world plates with dirt, damage, unusual fonts, or non-standard formats may be misrecognized.

### Mitigations
- Fairness analysis (see `responsible_ai/fairness_analysis.py`) measures and reports accuracy disparities.
- The model uses format-based post-processing to correct common digit/letter confusion.
- Regular retraining with diverse real-world data is recommended.

## 4. Misuse Potential

- **Stalking/harassment**: Tracking individuals via their plate numbers.
- **Unauthorized surveillance**: Deploying in areas without legal authorization.
- **Discrimination**: Using plate recognition to profile individuals.

### Mitigations
- Access to the API should require authentication.
- Usage should be logged (who accessed, when) for audit purposes.
- Terms of service should prohibit unauthorized tracking.

## 5. Transparency

- The model's decision process is explainable through SHAP and LIME analyses.
- Users of the API can inspect which HOG features drive classification decisions.
- Model metadata (accuracy, training data composition) is documented.

## 6. Recommendations for Production Deployment

1. **Data minimization**: Process images on-device when possible; avoid cloud storage of plates.
2. **Consent**: In consumer-facing applications, users should be informed that plate recognition is occurring.
3. **Audit trail**: Log API access (not plate content) for accountability.
4. **Regular review**: Periodically re-evaluate model fairness as new data becomes available.
5. **Legal compliance**: Ensure deployment complies with local privacy regulations (PDPA, GDPR equivalent).
