import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import re
import os
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from evidently.core.report import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import DriftedColumnsCount

print("=" * 60)
print("Step 1: Setting up MLflow experiment")
print("=" * 60)

# Configure MLflow experiment
mlflow.set_experiment("Iris_RandomForest_Drift_Detection")

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

print("\nStep 2: Loading and preparing data")
print("-" * 60)

# Load data and prep
iris_data = load_iris(as_frame=True)
df = iris_data.frame
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Number of features: {X_train.shape[1]}")

print("\nStep 3: Training baseline model")
print("-" * 60)

# Start MLflow run for baseline model
with mlflow.start_run(run_name="baseline_model") as run:
    # Train
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    # Predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate quality metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Save model as artifact
    mlflow.sklearn.log_model(model, "model")
    
    print(f"\nBaseline model quality metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nModel saved to MLflow run: {run.info.run_id}")
    
    # Save baseline data for comparison
    baseline_run_id = run.info.run_id

print("\n" + "=" * 60)
print("Step 4: Simulating feature drift")
print("=" * 60)

# Create drifted data with random intensity and features
X_drifted = X_test.copy()

# Randomly choose drift intensity: 'strong' (will trigger alert) or 'weak' (won't trigger alert)
drift_intensity = np.random.choice(['strong', 'weak'], p=[0.5, 0.5])

if drift_intensity == 'strong':
    # Strong drift: apply to multiple features with high intensity (will trigger >30% alert)
    # Randomly select 2-3 features out of 4 to drift
    num_features_to_drift = np.random.randint(2, 4)
    features_to_drift = np.random.choice(X_drifted.columns, size=num_features_to_drift, replace=False)
    drift_loc = np.random.uniform(1.5, 2.5)  # Higher drift magnitude
    drift_scale = np.random.uniform(0.2, 0.4)
    print(f"Applying STRONG drift (may trigger alert) to {num_features_to_drift} feature(s)")
else:
    # Weak drift: apply to fewer features with lower intensity (won't trigger alert)
    num_features_to_drift = np.random.randint(1, 2)
    features_to_drift = np.random.choice(X_drifted.columns, size=num_features_to_drift, replace=False)
    drift_loc = np.random.uniform(0.3, 0.8)  # Lower drift magnitude
    drift_scale = np.random.uniform(0.1, 0.2)
    print(f"Applying WEAK drift (should not trigger alert) to {num_features_to_drift} feature(s)")

# Apply drift to selected features
drift_info = {}
for feature in features_to_drift:
    old_mean = X_drifted[feature].mean()
    X_drifted[feature] += np.random.normal(loc=drift_loc, scale=drift_scale, size=len(X_drifted))
    new_mean = X_drifted[feature].mean()
    drift_info[feature] = {
        'old_mean': old_mean,
        'new_mean': new_mean,
        'change': new_mean - old_mean
    }
    print(f"  {feature}:")
    print(f"    Mean before: {old_mean:.4f}")
    print(f"    Mean after:  {new_mean:.4f}")
    print(f"    Mean change: {new_mean - old_mean:.4f}")

print("\n" + "=" * 60)
print("Step 5: Drift detection with Evidently")
print("=" * 60)

# Define custom alert threshold (30% as specified in readme)
DRIFT_ALERT_THRESHOLD = 0.3  # 30% threshold as specified in readme

print("Generating drift report...")
print(f"Drift alert threshold: {DRIFT_ALERT_THRESHOLD*100:.0f}% ({DRIFT_ALERT_THRESHOLD})")

# Create report with Evidently's DataDriftPreset
report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(current_data=X_drifted, reference_data=X_train)

# Save HTML report locally with timestamp to ensure fresh generation
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
drift_report_path = f"drift_report_{timestamp}.html"

# Remove old drift_report.html if exists to ensure fresh file
old_report_path = "drift_report.html"
if os.path.exists(old_report_path):
    # Check if it's a symlink before removing
    if os.path.islink(old_report_path):
        os.unlink(old_report_path)
    else:
        os.remove(old_report_path)
    print(f"Removed old report file: {old_report_path}")

# Save new report
snapshot.save_html(drift_report_path)
print(f"Drift report saved to: {drift_report_path}")

# Create symlink to standard name for easy access (Unix/Linux/Mac only)
if os.name != 'nt':  # Not Windows
    try:
        os.symlink(drift_report_path, old_report_path)
        print(f"Created symlink: {old_report_path} -> {drift_report_path}")
    except Exception as e:
        print(f"Could not create symlink (using timestamped file directly): {e}")

# Get metrics from report
report_json = snapshot.dict()
drift_metrics = {}
drift_by_feature = {}

# Extract drift information from report (new API structure)
if 'metrics' in report_json:
    # Get total number of features from the dataset
    n_features = len(X_drifted.columns)
    drift_metrics['n_features'] = n_features
    
    # Extract DriftedColumnsCount metric
    for metric in report_json['metrics']:
        metric_id = str(metric.get('metric_id', ''))
        value = metric.get('value', {})
        
        # Get drift count and share
        if 'DriftedColumnsCount' in metric_id:
            if isinstance(value, dict):
                drift_metrics['n_drifted_features'] = int(value.get('count', 0))
                drift_metrics['drift_share'] = float(value.get('share', 0.0))
        
        # Get drift for each feature
        elif 'ValueDrift' in metric_id:
            # Extract column name from metric_id like "ValueDrift(column=sepal length (cm))"
            match = re.search(r'column=([^)]+)', metric_id)
            if match:
                feature_name = match.group(1)
                # Value is p-value, lower means more drift
                p_value = float(value) if not isinstance(value, dict) else 0.0
                drift_detected = p_value < 0.05  # Standard threshold for drift detection
                drift_by_feature[feature_name] = {
                    'drift_score': p_value,
                    'drift_detected': drift_detected
                }

# Display results
print(f"\nDrift detection results:")
print(f"  Total features: {drift_metrics.get('n_features', 0)}")
print(f"  Drifted features: {drift_metrics.get('n_drifted_features', 0)}")
if drift_metrics.get('n_features', 0) > 0:
    drift_share = drift_metrics.get('drift_share', 0.0)
    print(f"  Drift percentage: {drift_share*100:.2f}%")
    
    # Alert logic: alert if more than 30% of features exhibit significant drift
    # Note: DRIFT_ALERT_THRESHOLD is defined above in Step 5
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    if drift_share > DRIFT_ALERT_THRESHOLD:
        print(f"\n{RED}{'=' * 60}{RESET}")
        print(f"{RED}ALERT: SIGNIFICANT DRIFT DETECTED!{RESET}")
        print(f"{RED}{'=' * 60}{RESET}")
        print(f"{YELLOW}WARNING: Drift share ({drift_share*100:.2f}%) exceeds alert threshold ({DRIFT_ALERT_THRESHOLD*100:.0f}%){RESET}")
        print(f"{YELLOW}WARNING: Action required: Model performance may be degraded{RESET}")
        print(f"{YELLOW}WARNING: Recommended: Trigger retraining or investigate data pipeline{RESET}")
        print(f"{RED}{'=' * 60}{RESET}")
    else:
        print(f"\n{GREEN}OK: Drift share ({drift_share*100:.2f}%) is within acceptable range (< {DRIFT_ALERT_THRESHOLD*100:.0f}%){RESET}")

if drift_by_feature:
    print(f"\nFeature details:")
    for feature_name, feature_info in drift_by_feature.items():
        drift_score = feature_info.get('drift_score', 0)
        drift_detected = feature_info.get('drift_detected', False)
        status = "DRIFT DETECTED" if drift_detected else "No drift"
        print(f"  {feature_name}: {status} (p-value: {drift_score:.4f})")

print("\n" + "=" * 60)
print("Step 6: Logging drift report to MLflow")
print("=" * 60)

# Create new MLflow run for drift detection
with mlflow.start_run(run_name="drift_detection") as drift_run:
    # Log drift report as artifact
    mlflow.log_artifact(drift_report_path, "drift_reports")
    
    # Log drift metrics
    # Note: DRIFT_ALERT_THRESHOLD is defined above in Step 5 (0.3 = 30%)
    if drift_metrics:
        mlflow.log_metric("n_features", drift_metrics.get('n_features', 0))
        mlflow.log_metric("n_drifted_features", drift_metrics.get('n_drifted_features', 0))
        drift_share = drift_metrics.get('drift_share', 0.0)
        mlflow.log_metric("drift_share", drift_share)
        
        # Log alert status
        alert_triggered = drift_share > DRIFT_ALERT_THRESHOLD
        mlflow.log_metric("alert_triggered", 1 if alert_triggered else 0)
        mlflow.log_param("drift_alert_threshold", DRIFT_ALERT_THRESHOLD)
    else:
        drift_share = 0.0
        alert_triggered = False
    
    # Add tags
    mlflow.set_tag("run_type", "drift_detection")
    mlflow.set_tag("baseline_run_id", baseline_run_id)
    mlflow.set_tag("drift_detected", str(drift_metrics.get('n_drifted_features', 0) > 0))
    mlflow.set_tag("alert_triggered", str(alert_triggered))
    mlflow.set_tag("simulated_drift_intensity", drift_intensity)
    mlflow.set_tag("num_features_drifted", str(num_features_to_drift))
    mlflow.set_tag("features_with_drift", ", ".join(features_to_drift))
    if alert_triggered:
        mlflow.set_tag("alert_reason", f"drift_share_{drift_share:.2f}_exceeds_30_percent")
    
    print(f"Drift report logged to MLflow run: {drift_run.info.run_id}")
    print(f"Artifacts available in MLflow UI")

print("\n" + "=" * 60)
print("Successfully completed!")
print("=" * 60)
print("\nNext steps:")
print("1. Run 'mlflow ui' to view results in the web interface")
print("2. Find the experiment 'Iris_RandomForest_Drift_Detection'")
print("3. View drift report in Artifacts -> drift_reports")
print("=" * 60)

