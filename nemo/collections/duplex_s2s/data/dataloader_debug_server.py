import threading
import time
from collections import defaultdict

import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# Thread-safe storage for metrics
metrics_lock = threading.Lock()
metrics_data = []


@app.route('/metrics', methods=['POST'])
def receive_metrics():
    """Endpoint to receive metrics from dataloader workers"""
    try:
        data = request.get_json()

        # Extract identification information

        record = {
            "worker_id": data.get('worker_id', 'unknown'),
            "node_id": data.get('node_id', 'unknown'),
            "gpu_id": data.get('gpu_id', 'unknown'),
            "timestamp": data.get('timestamp', time.time()),
            **data.get('metrics', {}),
        }

        # Store data with thread safety
        with metrics_lock:
            metrics_data.append(record)

        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Endpoint to retrieve all collected metrics"""
    with metrics_lock:
        return jsonify(metrics_data)


@app.route('/metrics/aggregate', methods=['GET'])
def get_aggregated_metrics():
    """Endpoint to get aggregated metrics with specified aggregation function"""
    aggregation_type = request.args.get('type', 'mean')

    with metrics_lock:
        aggregated = {}
        for worker_key, worker_metrics in metrics_data.items():
            aggregated[worker_key] = {}
            for metric_name, metric_values in worker_metrics.items():
                values = [entry['value'] for entry in metric_values if isinstance(entry['value'], (int, float))]

                if not values:
                    continue

                if aggregation_type == 'mean':
                    aggregated[worker_key][metric_name] = np.mean(values)
                elif aggregation_type == 'sum':
                    aggregated[worker_key][metric_name] = np.sum(values)
                elif aggregation_type == 'min':
                    aggregated[worker_key][metric_name] = np.min(values)
                elif aggregation_type == 'max':
                    aggregated[worker_key][metric_name] = np.max(values)
                elif aggregation_type == 'count':
                    aggregated[worker_key][metric_name] = len(values)
                elif aggregation_type == 'std':
                    aggregated[worker_key][metric_name] = np.std(values)

    return jsonify(aggregated)


@app.route('/metrics/summary', methods=['GET'])
def get_metrics_summary():
    """Get a summary of metrics across all workers"""
    with metrics_lock:
        summary = {}
        metric_values = defaultdict(list)

        for worker_metrics in metrics_data.values():
            for metric_name, metric_entries in worker_metrics.items():
                for entry in metric_entries:
                    if isinstance(entry['value'], (int, float)):
                        metric_values[metric_name].append(entry['value'])

        for metric_name, values in metric_values.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'count': len(values),
                }

    return jsonify(summary)


@app.route('/metrics/clear', methods=['POST'])
def clear_metrics():
    """Clear all stored metrics"""
    with metrics_lock:
        metrics_data.clear()

    return jsonify({"status": "success", "message": "All metrics cleared"})


if __name__ == '__main__':
    # Run the server on all available interfaces
    app.run(host='0.0.0.0', port=5123, threaded=True)
