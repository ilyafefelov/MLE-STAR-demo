# MLE-STAR Customer Churn Prediction

This project implements an advanced machine learning pipeline using the **MLE-STAR agentic workflow** to automate end-to-end customer churn prediction for telecom companies.

## 🌟 Features

- **Autonomous Pipeline**: Self-optimizing ML pipeline with minimal human intervention
- **State-of-the-Art Approaches**: Integrates latest churn prediction methodologies
- **Iterative Refinement**: Component-wise optimization and ablation studies
- **Ensemble Strategies**: Multiple model combination for improved performance
- **Robustness Checking**: Built-in data leakage and error detection
- **Comprehensive Reporting**: Detailed performance metrics and process logs

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Dependencies (automatically installed):
  - scikit-learn, pandas, numpy, matplotlib, seaborn
  - xgboost, lightgbm (for advanced models)
  - Optional: langchain, crewai, fastapi (for full agentic framework)

### Installation
```bash
git clone <your-repo>
cd MLE-STAR-demo
pip install -r requirements.txt
```

### Usage
1. **Place your dataset**: Add `telecom_churn.csv` to the `data/` folder (optional - sample data will be generated)
2. **Run the pipeline**:
   ```bash
   python solution.py
   ```
3. **Review results**: Check `report.md` and `process_log.md` for detailed analysis

## 📊 Latest Results

### Model Performance
- **F1-Score**: 0.4889 (Primary metric)
- **Accuracy**: 65.50%
- **Precision**: 55.93%
- **Recall**: 43.42%

### Best Performing Models
1. **XGBoost**: F1-Score = 0.5278
2. **Tuned Random Forest**: F1-Score = 0.4812
3. **Weighted Ensemble**: F1-Score = 0.4889 (Final model)

## 🏗️ Project Structure

```
MLE-STAR-demo/
├── solution.py           # Main MLE-STAR pipeline
├── data/
│   └── telecom_churn.csv # Dataset (auto-generated if missing)
├── report.md             # Performance summary
├── process_log.md        # Detailed process log
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 MLE-STAR Components

### 1. Web Search Simulation
- Analyzes state-of-the-art approaches
- Identifies optimal ML strategies

### 2. Targeted Refinement
- Component-wise optimization
- Preprocessing pipeline tuning
- Feature selection strategies

### 3. Ensemble Generation
- Multiple candidate model creation
- Voting and weighted ensemble strategies
- Performance-based model selection

### 4. Robustness Checking
- Data leakage detection
- Error handling and debugging
- Data usage validation

### 5. Ablation Studies
- Feature importance analysis
- Component impact assessment
- Performance attribution

## 📈 Key Features

- **Automated Feature Engineering**: Creates interaction terms, encodings, and derived features
- **Multiple Algorithm Support**: Random Forest, XGBoost, LightGBM, Gradient Boosting, SVM, Logistic Regression
- **Hyperparameter Optimization**: Grid search and random search strategies
- **Cross-Validation**: Stratified k-fold validation for robust evaluation
- **Interpretability Focus**: Prioritizes explainable models when performance is comparable

## 🎯 Optimization Goals

- **Primary**: F1-Score maximization (handles class imbalance)
- **Secondary**: Accuracy, Precision, Recall reporting
- **Tertiary**: Model interpretability and business insights

## 📋 Process Log Highlights

The MLE-STAR pipeline automatically logs:
- Each refinement iteration
- Model performance comparisons  
- Feature engineering decisions
- Ensemble strategy evolution
- Data quality checks
- Ablation study results

## 🔍 Data Requirements

Expected CSV format with churn prediction features:
- Customer demographics (Age, Gender)
- Service usage (Tenure, Monthly Charges)
- Contract details (Contract type, Payment method)
- Target variable: `Churn` (0/1)

## 🛠️ Customization

Modify `solution.py` to:
- Change target column name
- Adjust feature engineering strategies
- Add new model types
- Modify ensemble approaches
- Update evaluation metrics

## 📚 References

- **MLE-STAR Framework**: [Google ADK Samples](https://github.com/google/adk-samples/tree/main/python/agents/machine-learning-engineering)
- **Agent Development Kit**: Google's ADK for agentic ML workflows
- **Multi-Agent Orchestration**: CrewAI framework integration
- **LangChain Integration**: For advanced agent reasoning capabilities

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add new MLE-STAR components
4. Test with sample datasets
5. Submit pull request

## 📄 License

This project is licensed under the MIT License.

---

**Built with the MLE-STAR agentic workflow for autonomous machine learning engineering.**
