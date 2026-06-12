import os
import pandas as pd

def main():
    if not os.path.exists('outputs/csv/results.csv'):
        print("LNN results ('results.csv') not found. Make sure to run the LNN pipeline first.")
        return
        
    if not os.path.exists('outputs/csv/rf_results.csv'):
        print("RF results ('rf_results.csv') not found. Run rf_baseline.py first.")
        return
        
    if not os.path.exists('outputs/csv/lstm_results.csv'):
        print("LSTM results ('lstm_results.csv') not found. Run train_lstm.py first.")
        return
        
    if not os.path.exists('outputs/csv/gru_results.csv'):
        print("GRU results ('gru_results.csv') not found. Run train_gru.py first.")
        return
        
    if not os.path.exists('outputs/csv/transformer_results.csv'):
        print("Transformer results ('transformer_results.csv') not found. Run train_transformer.py first.")
        return
        
    if not os.path.exists('outputs/csv/gan_results.csv'):
        print("GAN results ('gan_results.csv') not found. Run train_gan.py first.")
        return
        
    if not os.path.exists('outputs/csv/autoregressive_results.csv'):
        print("AR results ('autoregressive_results.csv') not found. Run train_autoregressive.py first.")
        return
        
    lnn_df = pd.read_csv('outputs/csv/results.csv').iloc[0]
    rf_df = pd.read_csv('outputs/csv/rf_results.csv').iloc[0]
    lstm_df = pd.read_csv('outputs/csv/lstm_results.csv').iloc[0]
    gru_df = pd.read_csv('outputs/csv/gru_results.csv').iloc[0]
    tf_df = pd.read_csv('outputs/csv/transformer_results.csv').iloc[0]
    gan_df = pd.read_csv('outputs/csv/gan_results.csv').iloc[0]
    ar_df = pd.read_csv('outputs/csv/autoregressive_results.csv').iloc[0]
    
    # Get model file sizes
    lnn_size_mb = os.path.getsize('models/lnn_model.pth') / (1024 * 1024) if os.path.exists('models/lnn_model.pth') else 0.0
    rf_size_mb = os.path.getsize('models/rf_model.pkl') / (1024 * 1024) if os.path.exists('models/rf_model.pkl') else 0.0
    lstm_size_mb = os.path.getsize('models/lstm_model.pth') / (1024 * 1024) if os.path.exists('models/lstm_model.pth') else 0.0
    gru_size_mb = os.path.getsize('models/gru_model.pth') / (1024 * 1024) if os.path.exists('models/gru_model.pth') else 0.0
    tf_size_mb = os.path.getsize('models/transformer_model.pth') / (1024 * 1024) if os.path.exists('models/transformer_model.pth') else 0.0
    gan_size_mb = os.path.getsize('models/gan_model.pth') / (1024 * 1024) if os.path.exists('models/gan_model.pth') else 0.0
    ar_size_mb = os.path.getsize('models/autoregressive_model.pth') / (1024 * 1024) if os.path.exists('models/autoregressive_model.pth') else 0.0
    
    comp_df = pd.DataFrame({
        'Model': ['LNN', 'RF', 'LSTM', 'GRU', 'Transformer', 'GAN', 'AR'],
        'Precision': [lnn_df['Precision'], rf_df['Precision'], lstm_df['Precision'], gru_df['Precision'], tf_df['Precision'], gan_df['Precision'], ar_df['Precision']],
        'Recall': [lnn_df['Recall'], rf_df['Recall'], lstm_df['Recall'], gru_df['Recall'], tf_df['Recall'], gan_df['Recall'], ar_df['Recall']],
        'F1': [lnn_df['F1_Score'], rf_df['F1_Score'], lstm_df['F1_Score'], gru_df['F1_Score'], tf_df['F1_Score'], gan_df['F1_Score'], ar_df['F1_Score']],
        'Latency (ms)': [lnn_df['Avg_Latency_ms'], rf_df['Avg_Latency_ms'], lstm_df['Avg_Latency_ms'], gru_df['Avg_Latency_ms'], tf_df['Avg_Latency_ms'], gan_df['Avg_Latency_ms'], ar_df['Avg_Latency_ms']],
        'Size (MB)': [lnn_size_mb, rf_size_mb, lstm_size_mb, gru_size_mb, tf_size_mb, gan_size_mb, ar_size_mb]
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
    
    comp_df.to_csv('outputs/csv/comparison_results.csv', index=False)
    print("Saved comparison table to 'outputs/csv/comparison_results.csv'")

if __name__ == "__main__":
    main()
