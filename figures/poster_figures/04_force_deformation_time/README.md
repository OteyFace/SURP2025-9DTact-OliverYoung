# Force-Deformation over Time

## Overview
This section contains figures demonstrating the temporal analysis of force-deformation relationships in the enhanced 9DTact sensor system. The figures show the monotonic relationship between applied force and gel deformation, supporting stiffness estimation and material characterization.

## Figures

### force_deformation_over_time.png
**Caption**: Temporal analysis showing applied force against gel deformation over the acquisition sequence (negative Z; start depth 21.25 mm). The plot demonstrates monotonic increase of deformation δ with force and supports a first-order stiffness estimate, providing insights into the sensor's mechanical response characteristics.

**Technical Details**:
- **Deformation Calculation**: δ = d_start - d_t (where d_start = 21.25 mm)
- **Force Range**: Approximately 0.5-20 N applied force range
- **Temporal Resolution**: High-frequency sampling during force application
- **Stiffness Estimation**: Linear fit F = k·δ + b over quasi-linear region

**Key Metrics**:
- **Effective Stiffness**: k (N/mm) from linear regression
- **Confidence Intervals**: Statistical uncertainty in stiffness measurement
- **Linearity**: Residual analysis to verify linear relationship
- **Hysteresis**: Up/down force sweeping to assess material behavior

**Material Characterization**:
- **Stiffness Constant**: Summarizes material/system resistance to deformation
- **Comparison**: Enables comparison across sensors or experimental sessions
- **Validation**: Provides ground truth for force estimation algorithms
- **Quality Control**: Monitors sensor performance over time

**Research Context**: This temporal analysis provides crucial validation data for the force estimation algorithms and demonstrates the sensor's ability to accurately track force-deformation relationships in real-time. The stiffness estimation serves as a key parameter for calibrating the force estimation models.
