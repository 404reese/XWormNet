# Project Audit: Explainable LNN Framework for Real-Time Zero-Day Worm Detection

## Project Overview
This codebase implements an explainable liquid neural network (LNN) framework for real-time zero-day worm detection in IoT and enterprise networks, using the IoT and Ent Worm 2025 dataset (GothamDataset2025). The project includes implementations of eight models (LNN, RF, LSTM, GRU, Traffic Transformer, GAN, AR, RNN), training and evaluation pipelines with leakage prevention measures, model comparison tools, explainability modules (SHAP, LIME, T-SHAP), and a Streamlit web application for interactive model training, comparison, and interpretation. The codebase emphasizes rigorous data preprocessing to avoid duplication and shortcut features, ensuring reliable generalization estimates.

## Requirements Checklist
The following table maps the professor's requirements against the current implementation:

| Requirement | Status | Details |
|-------------|--------|---------|
| **Models Executed**: LNN, RNN, LSTM, GRU, Autoregressive, Traffic Transformer, GAN, RF | ✅ Covered | All eight models are implemented: `lnn_model.py`, `rnn_model.py`, `lstm_model.py`, `gru_model.py`, `autoregressive_model.py`, `transformer_model.py`, `gan_model.py`, and `rf_baseline.py`. |
| **Metrics Evaluated**: Precision, Recall, F1-score, Accuracy, Latency, Model Size | ⚠️ Partial | Precision, recall, F1, latency (inference time), and model size are recorded in comparison results (`outputs/csv/comparison_results.csv`). Accuracy is computed internally in `train_and_test.py` (classification report) but not saved to result CSVs or displayed in the web app. |
| **XAI Integration**: TShap (TreeSHAP/SHAP) and LIME | ✅ Covered | SHAP and LIME are implemented in `src/explainability.py` and used in `app_explainability.py`. T-SHAP (time-series SHAP) is available for sequence models (LSTM, GRU, RNN, Transformer, AR). |
| **Visualizations**: Infographs for every model, loss analysis curves, model test accuracy plots | ⚠️ Partial | LNN-specific visualizations (liquid parameters, hidden state, LNN vs LSTM) are generated via `visualize_liquid_params.py`. A SHAP summary plot (`outputs/images/shap_summary.png`) exists. Loss curves are shown during training in the Streamlit app (neural models only) but not saved as standalone plots. No model test accuracy plots are generated. Infographs for non-LNN models (RF, LSTM, etc.) are missing. |
| **Web Application**: Dataset selection (single/cross-dataset), plotting graphs (loss, accuracy), XAI explanations, application performance recommendations | ⚠️ Partial | - Dataset selection: single dataset ( upload or default) supported; cross-dataset selection missing (available only in `multi_dataset_pipeline.py` but not integrated). <br> - Plotting graphs: loss curves shown during training (neural models); accuracy plots missing. <br> - XAI explanations: fully implemented via SHAP/LIME/T-SHAP tabs. <br> - Application performance recommendations: limited insights provided in `app_comparison.py` (fastest, highest F1, smallest model) and hints in the UI, but no explicit recommendations based on conclusions for deployment scenarios. |
| **Documentation for Paper**: Hyperparameters, tech stack, findings, future scope, conclusion, contribution, novelty | ❌ Missing | No dedicated documentation sections exist for hyperparameter tuning, tech stack, findings, future scope, conclusion, contribution, or novelty. The `README.md` contains only a brief project aim. |

## Deep Dive: What is Covered

### Models
All eight required models are implemented:
- **LNN**: Liquid Neural Network (`src/lnn_model.py`), a continuous-time dynamical system.
- **RNN**: Vanilla recurrent neural network (`src/rnn_model.py`).
- **LSTM**: Long Short-Term Memory (`src/lstm_model.py`).
- **GRU**: Gated Recurrent Unit (`src/gru_model.py`).
- **Autoregressive (AR)**: Autoregressive classifier (`src/autoregressive_model.py`).
- **Traffic Transformer**: Transformer-based model for traffic sequences (`src/transformer_model.py`).
- **GAN**: Generative Adversarial Network detector (`src/gan_model.py`).
- **RF**: Random Forest baseline (`rf_baseline.py`).

Each model follows a consistent interface for training and evaluation, with sequence models (LSTM, GRU, RNN, Transformer, AR) using a sliding window approach.

### Metrics
The following metrics are computed and recorded:
- **Precision**, **Recall**, **F1-score**: Calculated via `compute_binary_metrics` in `train_and_test.py` and `sklearn.metrics` in `app_train.py`.
- **Latency (inference time)**: Measured as average latency per sample in milliseconds (e.g., `Avg_Latency_ms` in result CSVs).
- **Model Size (memory footprint)**: Computed as file size of saved model weights (e.g., `Size (MB)` in `comparison_results.csv`).
- **Accuracy**: Computed internally in `train_and_test.py` (`accuracy = (tp + tn) / len(y_true)`) and printed in classification reports, but not persisted to result CSVs or exposed in the web app.

### XAI Integration
- **SHAP**: Implemented for all model types using appropriate explainers (TreeExplainer for RF, KernelExplainer for LNN, GradientExplainer for sequence models).
- **LIME**: Implemented via `LimeTabularExplainer` for all models, with sequence-specific handling.
- **T-SHAP**: Time-series SHAP via `GradientExplainer` on full sequence inputs for sequence models, providing timestep-wise feature importance.

The explainability module (`src/explainability.py`) returns feature importance values, which are visualized in the Streamlit app (`app_explainability.py`) via bar charts, heatmaps, and comparison plots.

### Web Application
The Streamlit web app (`app.py`) consists of five tabs:
1. **Home**: Project description and feature overview.
2. **Model Comparison**: Displays aggregated model comparison table (`outputs/csv/comparison_results.csv`) with interactive charts (F1, latency, size, precision vs recall).
3. **Train Model**: Allows users to select a model, upload a dataset (or use default), set hyperparameters, train the model, view metrics (precision, recall, F1, latency, size), and see training loss curves (for neural models).
4. **Explainability**: Enables SHAP/LIME/T-SHAP explanations for a selected model and input sample, with visualizations and insight panels.
5. **Export**: Provides downloadable CSV and markdown versions of the comparison table.

### Additional Features
- **Leakage Prevention**: Scripts include duplicate removal, shortcut feature removal (dropping IP/port/proto/mac/flow_id columns), and scenario-based train/test splitting (e.g., `multi_dataset_pipeline.py`).
- **Multi-Dataset Training**: `multi_dataset_pipeline.py` supports training LNN on multiple datasets and evaluating on a held-out test set.
- **Model Persistence**: Trained models are saved to the `models/` directory (e.g., `models/lnn_model.pth`).
- **Result Logging**: Training and evaluation results are saved as CSV files in `outputs/csv/`.

## Deep Dive: What is Missing or Incomplete

### Models
- All eight models are implemented, but some lack full hyperparameter tuning capabilities in the web app (e.g., only LNN exposes epochs and hidden size; other models use fixed hyperparameters).

### Metrics
- **Accuracy**: Not stored in result CSVs or displayed in the web app, despite being computed in `train_and_test.py`. This prevents easy comparison of accuracy across models in the summary table.
- **Additional Metrics**: Requirements only called for the listed metrics, but other useful metrics (e.g., ROC-AUC, FPR, FNR) are computed but not included in the comparison table.

### Visualizations
- **Infographs for Every Model**: Only LNN has dedicated visualizations (liquid parameters, hidden state evolution, LNN vs LSTM). No infographs exist for RF, LSTM, GRU, Transformer, GAN, or AR.
- **Loss Analysis Curves**: Loss curves are shown during training in the Streamlit app for neural models (LSTM, GRU, RNN, Transformer, AR, LNN) but are not saved as standalone plots. Non-neural models (RF) do not have training loss in the same sense.
- **Model Test Accuracy Plots**: No scatter plots, bar charts, or other visualizations comparing test accuracy across models or datasets are generated.

### Web Application
- **Cross-Dataset Selection**: The app only supports single dataset selection (upload or default). The capability to train on multiple datasets and test on another (as in `multi_dataset_pipeline.py`) is not integrated.
- **Plotting Graphs (Accuracy)**: No accuracy-related plots (e.g., accuracy vs epochs, accuracy bar chart per model) are present.
- **Application Performance Recommendations**: While the comparison tab provides insights (fastest model, highest F1, smallest model), it does not offer context-aware recommendations (e.g., "For IoT edge deployment with latency < 10ms, choose LNN or AR").

### Documentation
- No documentation exists for:
  - Hyperparameters used for tuning (e.g., learning rates, window sizes, epochs per model).
  - Tech stack (libraries and versions; partially inferable from `requirements.txt` but not elaborated).
  - Findings from experiments (e.g., which model performed best, key trade-offs).
  - Future scope (planned improvements, extensions).
  - Conclusion (summary of results and implications).
  - Contribution (novel aspects of the work).
  - Novelty (what distinguishes this framework from prior work).

## Visuals & Infographs Audit
### Generated Plots
- `figures/lnn_liquid_parameters.png`: LNN gate evolution over time.
- `figures/lnn_hidden_heatmap.png`: LNN hidden state evolution heatmap.
- `figures/lnn_vs_lstm_parameters.png`: LNN liquid parameters vs LSTM fixed weights.
- `outputs/images/shap_summary.png`: SHAP summary plot (likely from a sequence model).

### Required Plots Missing
- Infographs for RF, LSTM, GRU, Transformer, GAN, AR models (e.g., model architecture diagrams, feature importance summaries).
- Loss analysis curves for all models (saved as PNG/SVG).
- Model test accuracy plots (e.g., bar chart of accuracy per model, accuracy vs false positive rate).
- SHAP summary plots for each model (currently only one generic SHAP summary).
- ROC curves, precision-recall curves, or confusion matrix visualizations.

## Documentation & Paper Readiness
The codebase lacks the essential documentation sections required for a research paper. While the code is functional and results are generated, the following must be written to satisfy the professor's requirements:
- **Hyperparameters**: Document the exact values used for each model (e.g., LNN: hidden_dim=16, num_steps=6, dt=0.1, lr=0.001; LSTM: hidden_size=64, window_size=10, epochs=20; etc.).
- **Tech Stack**: Elaborate on the libraries and versions (e.g., Python 3.x, PyTorch, scikit-learn, SHAP, LIME, Streamlit) and their roles.
- **Findings**: Summarize the experimental results (e.g., accuracy, F1, latency, size trade-offs) based on the generated CSV files.
- **Future Scope**: Discuss potential improvements (e.g., integrating cross-dataset selection in the web app, adding attention visualization for Transformer, exploring other XAI methods).
- **Conclusion**: Provide a concise summary of whether the LNN framework meets the real-time zero-day worm detection goal and recommendations for practitioners.
- **Contribution**: Articulate the novel contributions (e.g., leakage-resistant pipeline, comprehensive explainability suite, benchmarking of eight models on IoT worm dataset).
- **Novelty**: Highlight what is new (e.g., application of LNN to zero-day worm detection, combination of T-SHAP with LNN for temporal explainability, rigorous anti-leakage preprocessing).

## Action Plan
To achieve 100% completion before writing the paper, follow these steps:

### 1. Metrics Completeness
- Modify training and evaluation scripts to save **accuracy** to result CSVs (e.g., add an "Accuracy" column to `outputs/csv/*_results.csv`).
- Update `compare_models.py` to include accuracy in the comparison table.
- Ensure the web app displays accuracy in the Train Model tab and exports it in the comparison table.

### 2. Visualizations Generation
- Create a script to generate infographs for each model:
  - **RF**: Feature importance bar chart (using `feature_importances_`).
  - **LSTM/GRU/RNN**: Architecture diagram (simplified) and training loss curve.
  - **Transformer**: Attention visualization (if applicable) or loss curve.
  - **GAN**: Discriminator loss curve or generated samples (if applicable).
  - **AR**: Autoregressive coefficients plot or loss curve.
- Save loss analysis curves for all models during training (e.g., plot loss vs epochs and save to `outputs/images/`).
- Generate model test accuracy plots:
  - Bar chart comparing accuracy across all eight models.
  - If multiple datasets are used, grouped bar chart showing accuracy per model per dataset.
- Ensure all plots are saved with clear filenames and stored in `outputs/images/` or `figures/`.

### 3. Web Application Enhancements
- **Cross-Dataset Selection**: Add a option in the Train Model tab to select multiple training datasets and a separate test dataset (or integrate `multi_dataset_pipeline.py` as a separate tab).
- **Plotting Graphs (Accuracy)**:
  - In the Train Model tab, after training, display an accuracy vs epochs curve (for iterative models) or a single accuracy value.
  - Add a new tab or section for "Model Performance Plots" that shows infographs and accuracy plots for all trained models.
- **Application Performance Recommendations**:
  - In the Model Comparison tab, add a panel that provides context-aware recommendations (e.g., "For low-latency IoT deployment: LNN, AR, GAN"; "For highest accuracy: [model]"; "For smallest memory footprint: [model]").
  - Base recommendations on the data in `comparison_results.csv` (latency, size, F1).

### 4. Documentation Writing
Create a dedicated documentation section (e.g., in `docs/` or as expandable sections in the README) covering:
- **Hyperparameters**: List per-model hyperparameters used in the final experiments.
- **Tech Stack**: Detail libraries, versions, and hardware used.
- **Findings**: Summarize key results from `comparison_results.csv` and any ablation studies.
- **Future Scope**: Brainstorm enhancements (e.g., online learning, edge deployment optimization, additional XAI methods).
- **Conclusion**: Answer the research question: Does the explainable LNN framework provide effective real-time zero-day worm detection?
- **Contribution**: Bullet-point list of contributions (e.g., "First to apply LNN to IoT worm detection with temporal explainability via T-SHAP").
- **Novelty**: Highlight unique aspects (e.g., "Combines leakage-resistant preprocessing, eight-model benchmarking, and T-SHAP for sequential explainability in security context").

### 5. Verification
- Run the full pipeline: train all eight models (via `compare_models.py` prerequisites or the web app), generate comparison results, verify accuracy is included.
- Check that all required visualizations are present in `outputs/images/` and `figures/`.
- Review the web app to ensure all features work as specified.
- Compile the documentation into a coherent draft for the paper.

By addressing these gaps, the project will fully satisfy the professor's requirements and be ready for paper writing.