import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

@dataclass
class SmoothingParams:
    window_size: int
    min_duration: int
    hysteresis: float
    
    def to_dict(self) -> dict:
        return {
            'window_size': self.window_size,
            'min_duration': self.min_duration,
            'hysteresis': self.hysteresis
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SmoothingParams':
        return cls(
            window_size=d['window_size'],
            min_duration=d['min_duration'],
            hysteresis=d['hysteresis']
        )

class PredictionSmoother:
    def __init__(self):
        self.default_param_grid = {
            'window_size': [3, 5, 7, 9, 11],
            'min_duration': [1, 2, 3, 4, 5, 10],
            'hysteresis': [0.5, 0.55, 0.6, 0.65, 0.7]
        }

    def smooth_predictions(self, predictions: List[int], params: SmoothingParams) -> List[int]:
        if not predictions:
            return []
        
        predictions = [int(pred) for pred in predictions]
        window_size = max(3, params.window_size if params.window_size % 2 == 1 else params.window_size + 1)
        half_window = window_size // 2
        
        smoothed = []
        for i in range(len(predictions)):
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            window = predictions[start:end]
            active_ratio = sum(window) / len(window)
            threshold = params.hysteresis if smoothed and smoothed[-1] else 1 - params.hysteresis
            smoothed.append(1 if active_ratio >= threshold else 0)
        
        return smoothed

    def calculate_metrics(self, predictions: List[int], targets: List[int]) -> Dict[str, float]:
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length")
        
        pred_arr = np.array(predictions)
        target_arr = np.array(targets)
        
        accuracy = np.mean(pred_arr == target_arr)
        true_pos = np.sum((pred_arr == 1) & (target_arr == 1))
        false_pos = np.sum((pred_arr == 1) & (target_arr == 0))
        false_neg = np.sum((pred_arr == 0) & (target_arr == 1))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }

    def optimize_parameters(self, predictions: List[int], targets: List[int],
                            param_grid: Optional[Dict] = None, metric: str = 'f1_score') -> Tuple[SmoothingParams, Dict[str, float]]:
        if param_grid is None:
            param_grid = self.default_param_grid
        
        best_params, best_metrics, best_score = None, None, -float('inf')
        param_combinations = product(
            param_grid['window_size'],
            param_grid['min_duration'],
            param_grid['hysteresis']
        )
        
        for window_size, min_duration, hysteresis in param_combinations:
            params = SmoothingParams(window_size, min_duration, hysteresis)
            smoothed = self.smooth_predictions(predictions, params)
            metrics = self.calculate_metrics(smoothed, targets)
            
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_params = params
                best_metrics = metrics
        
        return best_params, best_metrics

def plot_predictions(raw_predictions: List[int], smoothed_predictions: List[int], target_sequence: List[int], filepath: str):
    plt.figure(figsize=(15, 6))
    x = range(len(raw_predictions))
    
    plt.step(x, raw_predictions, where='post', label='Raw Predictions', alpha=0.7, linestyle='dotted', color='blue')
    plt.step(x, smoothed_predictions, where='post', label='Smoothed Predictions', alpha=0.7, linestyle='solid', color='orange')
    plt.step(x, target_sequence, where='post', label='Target Sequence', alpha=0.7, linestyle='dashed', color='green')
    
    plt.ylim(-0.1, 1.1)
    plt.xlabel('Frame')
    plt.ylabel('Prediction')
    plt.title('Comparison of Raw and Smoothed Predictions')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

if __name__ == "__main__":
    predictions_df = pd.read_csv('predictionsXGB.csv')  # Use predictionsXGB.csv as input
    target_df = pd.read_csv('target.csv')

    merged_df = pd.merge(predictions_df, target_df, on='frame', suffixes=('_pred', '_target')).sort_values('frame')

    raw_predictions = merged_df['value_pred'].astype(int).tolist()
    target_sequence = merged_df['value_target'].astype(int).tolist()

    smoother = PredictionSmoother()
    best_params, best_metrics = smoother.optimize_parameters(
        raw_predictions, target_sequence, metric='f1_score'
    )

    smoothed_predictions = smoother.smooth_predictions(raw_predictions, best_params)

    print("\nBest parameters found:")
    print(json.dumps(best_params.to_dict(), indent=2))
    print("\nMetrics with best parameters:")
    print(json.dumps(best_metrics, indent=2))

    result_df = pd.DataFrame({
        'frame': merged_df['frame'],
        'value': smoothed_predictions
    })

    result_df.to_csv('smoothed_predictions.csv', index=False)

    plot_predictions(raw_predictions, smoothed_predictions, target_sequence, 'predictions_comparison_with_distinctions.png')
    print("Results saved to 'smoothed_predictions.csv' and 'predictions_comparison_with_distinctions.png'.")
