import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
outputs = root / 'outputs'
outputs_csv = outputs / 'csv'
outputs_images = outputs / 'images'
outputs_reports = outputs / 'reports'
models_dir = root / 'models'

mappings = {
    'comparison_results.csv': outputs_csv / 'comparison_results.csv',
    'results.csv': outputs_csv / 'results.csv',
    'rf_results.csv': outputs_csv / 'rf_results.csv',
    'lstm_results.csv': outputs_csv / 'lstm_results.csv',
    'fixed_results.csv': outputs_csv / 'fixed_results.csv',
    'predictions.csv': outputs_csv / 'predictions.csv',
    'confusion_matrix.csv': outputs_csv / 'confusion_matrix.csv',
    'multi_results.csv': outputs_csv / 'multi_results.csv',
    'lime_explanations.txt': outputs_reports / 'lime_explanations.txt',
    'shap_summary.png': outputs_images / 'shap_summary.png',
}

# model files
model_files = [
    'lnn_model.pth',
    'lnn_fixed.pth',
    'lnn_multi.pth',
    'lnn_strict.pth',
    'lstm_model.pth',
    'rf_model.pkl',
]

for src_name, dst in mappings.items():
    src = root / src_name
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(src), str(dst))
            print(f"Moved {src_name} -> {dst}")
        except Exception as e:
            print(f"Failed to move {src_name}: {e}")
    else:
        print(f"Source not found: {src}")

print('Done')
for mf in model_files:
    src = root / mf
    if src.exists():
        dst = models_dir / mf
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(src), str(dst))
            print(f"Moved {mf} -> {dst}")
        except Exception as e:
            print(f"Failed to move {mf}: {e}")
    else:
        print(f"Model file not found: {src}")

print('Model moves done')
