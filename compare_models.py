import os
import pandas as pd

def main():
    if not os.path.exists('results.csv'):
        print("LNN results ('results.csv') not found. Make sure to run the LNN pipeline first.")
        return
        
    if not os.path.exists('rf_results.csv'):
        print("RF results ('rf_results.csv') not found. Run rf_baseline.py first.")
        return
        
    if not os.path.exists('lstm_results.csv'):
        print("LSTM results ('lstm_results.csv') not found. Run train_lstm.py first.")
        return
        
    lnn_df = pd.read_csv('results.csv').iloc[0]
    rf_df = pd.read_csv('rf_results.csv').iloc[0]
    lstm_df = pd.read_csv('lstm_results.csv').iloc[0]
    
    # Get model file sizes
    lnn_size_mb = os.path.getsize('lnn_model.pth') / (1024 * 1024) if os.path.exists('lnn_model.pth') else 0.0
    rf_size_mb = os.path.getsize('rf_model.pkl') / (1024 * 1024) if os.path.exists('rf_model.pkl') else 0.0
    lstm_size_mb = os.path.getsize('lstm_model.pth') / (1024 * 1024) if os.path.exists('lstm_model.pth') else 0.0
    
    comp_df = pd.DataFrame({
        'Model': ['LNN', 'RF', 'LSTM'],
        'Precision': [lnn_df['Precision'], rf_df['Precision'], lstm_df['Precision']],
        'Recall': [lnn_df['Recall'], rf_df['Recall'], lstm_df['Recall']],
        'F1': [lnn_df['F1_Score'], rf_df['F1_Score'], lstm_df['F1_Score']],
        'Latency (ms)': [lnn_df['Avg_Latency_ms'], rf_df['Avg_Latency_ms'], lstm_df['Avg_Latency_ms']],
        'Size (MB)': [lnn_size_mb, rf_size_mb, lstm_size_mb]
    })
    
    # Format for display
    print("\n" + "="*80)
    print("                              MODEL COMPARISON")
    print("="*80)
    print(f"| {'Model':<5} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'Latency (ms)':<15} | {'Size (MB)':<10} |")
    print("|" + "-"*7 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*12 + "|" + "-"*17 + "|" + "-"*12 + "|")
    
    for _, row in comp_df.iterrows():
        print(f"| {row['Model']:<5} | {row['Precision']:<10.4f} | {row['Recall']:<10.4f} | {row['F1']:<10.4f} | {row['Latency (ms)']:<15.4f} | {row['Size (MB)']:<10.4f} |")
        
    print("="*80 + "\n")
    
    comp_df.to_csv('comparison_results.csv', index=False)
    print("Saved comparison table to 'comparison_results.csv'")

if __name__ == "__main__":
    main()
