# Master Experiment Summary: ADK vs Single-Shot (Iris Dataset)

## 1. Experiment Overview
- **Objective**: Compare the performance of the ADK MSE-STAR Agent against various Gemini Single-Shot models on the Iris dataset.
- **Methodology**:
    - **ADK**: Used existing artifacts (`iris_gbdt` task), wrapped them, and ran ablation studies.
    - **Single-Shot**: Generated pipelines using multiple Gemini models (`flash`, `lite`, `pro`, `live`), wrapped them, and ran ablation studies.
    - **Ablation**: Evaluated each pipeline under 6 configurations (full, no_scaling, no_feature_engineering, no_tuning, no_ensemble, minimal).

## 2. Results Summary

### ADK Agent Performance
| Pipeline Variant | Mean Accuracy (Full Config) | Best Config Accuracy |
| :--- | :--- | :--- |
| `run1_train0` | 0.944 | 0.944 (All) |
| `run1_train0_0` | 0.944 | 0.944 (All) |
| `run1_train0_improve0` | 0.944 | 0.944 (All) |

*Observation*: The ADK agent produced consistent pipelines with stable performance across all ablation configurations.

### Single-Shot Gemini Performance (Selected Models)
| Model | Mean Accuracy (Full Config) | Best Config Accuracy | Best Config Name |
| :--- | :--- | :--- | :--- |
| `gemini_live_pro` | **0.978** | **0.978** | Full, No Scaling, etc. |
| `gemini_live_flash` | 0.967 | **0.989** | No Scaling, Minimal |
| `gemini_2.5_flash_lite` | 0.933 | **0.989** | Minimal |
| `gemini_2.5_pro` | 0.933 | **0.989** | Minimal |
| `gemini_2.5_flash` | 0.911 | 0.967 | Minimal, No FE |

*Observation*: 
- Single-Shot models, particularly `gemini_live_pro` and `gemini_live_flash`, outperformed the ADK agent on this dataset.
- Single-Shot pipelines often performed *better* with "Minimal" configuration (removing scaling/tuning/ensembling), suggesting the generated "Full" pipelines might have been slightly over-engineered for the simple Iris dataset.

## 3. Conclusion
- **Winner**: Single-Shot (`gemini_live_pro`) for peak accuracy on Iris.
- **Stability**: ADK pipelines were more stable across configurations (less variance between "Full" and "Minimal"), whereas Single-Shot pipelines showed significant sensitivity to pipeline components (often benefiting from simplification).
- **Recommendation**: For simple datasets like Iris, Single-Shot generation with a powerful model like `gemini_live_pro` yields excellent results. The ADK agent provides a robust baseline but didn't reach the same peak accuracy in this specific run.

## 4. Artifacts
- **Detailed Stats**: `reports/aggregate_summary_stats.csv`
- **Visual Comparison**: `reports/master_comparison.png`
