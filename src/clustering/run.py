import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.clustering.load_data import load_data
from src.clustering.transform import transform_to_fact
from src.clustering.feature_engineering import create_features
from src.ml.unsupervised.train_risk import train_model
from src.clustering.evaluation import evaluate, evaluate_optimal_k_for_data

# 1. Load
spark, df = load_data("data/input/data_demo_1.csv")
features = create_features(transform_to_fact(df))

print("🔍 Evaluating optimal number of clusters...")
feature_cols = ["gpa", "std_score", "failed_subjects", "excellent_subjects"]
suggested_k = evaluate_optimal_k_for_data(features, min_k=2, max_k=8, feature_cols=feature_cols, auto_select=True)

# Train model with suggested K
print(f"🤖 Training model with K={suggested_k}...")
model, data = train_model(features, k=suggested_k)

# Evaluate final model
print("📊 Evaluating final model...")
clusters = evaluate(model, data)

print("✅ Pipeline hoàn thành!")
clusters.show(10)