import os
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ExperimentLogger:
    """Automated tracker for Q1 journal reproducibility standards with plotting."""
    
    def __init__(self, experiment_name, base_dir="logs"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, f"{self.timestamp}_{experiment_name}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Raw Data
        self.metrics_file = os.path.join(self.exp_dir, "metrics.csv")
        self.config_file = os.path.join(self.exp_dir, "config.json")
        self.report_file = os.path.join(self.exp_dir, "final_report.txt")
        
        # Visual Figures
        self.cm_image_file = os.path.join(self.exp_dir, "confusion_matrix.png")
        self.curve_image_file = os.path.join(self.exp_dir, "training_curves.png")
        
        self.metrics_headers_written = False
        print(f"\n[Logger] Directory initialized: {self.exp_dir}")

    def log_config(self, config_dict):
        with open(self.config_file, "w") as f:
            json.dump(config_dict, f, indent=4)
            
    def log_metric(self, metric_dict):
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metric_dict.keys())
            if not self.metrics_headers_written:
                writer.writeheader()
                self.metrics_headers_written = True
            writer.writerow(metric_dict)
            
    def log_report(self, report_text):
        with open(self.report_file, "w") as f:
            f.write(report_text)
            
    def log_confusion_matrix(self, cm, target_names):
        """Generates and saves a high-res heatmap of the confusion matrix."""
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Final Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.cm_image_file, dpi=300)
        plt.close()
        
    def generate_plots(self):
        """Reads the CSV at the end of training and plots the performance curves."""
        if not os.path.exists(self.metrics_file):
            return
            
        df = pd.read_csv(self.metrics_file)
        
        plt.figure(figsize=(10, 5))
        
        # We will track Clean F1 and ASR if they exist in the CSV
        if 'Clean_F1' in df.columns:
            plt.plot(df['Round'], df['Clean_F1'], marker='o', label='Clean Macro-F1', color='#1f77b4')
        if 'ASR' in df.columns:
            plt.plot(df['Round'], df['ASR'], marker='x', label='Attack Success Rate', color='#d62728')
            
        plt.title('Federated Learning Attack/Defense Simulation')
        plt.xlabel('Round')
        plt.ylabel('Score (%)')
        plt.ylim(0, 105) # Lock y-axis from 0 to 100%
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.curve_image_file, dpi=300)
        plt.close()
        print(f"[Logger] All visualizations and reports saved successfully.")
        
    def get_path(self):
        return self.exp_dir