import pandas as pd
import mlflow
from src.ingestion.data_generator import generate_cloud_costs
from src.models.prophet_model import train_prophet
from src.models.sarima_model import train_sarima
from src.models.baseline_model import train_baseline
from src.detection.isolation_forest import train_isolation_forest

# ── Configuration MLflow ──────────────────────────────────
mlflow.set_tracking_uri('http://localhost:5000')

# ── Paramètres ───────────────────────────────────────────
SERVICES = ['EC2', 'S3', 'RDS', 'Lambda', 'CloudFront', 'EKS', 'Redshift']
DATA_PATH = 'data/raw/cloud_costs.csv'

def main():
    # ── 1. Chargement des données ─────────────────────────
    print("=" * 50)
    print("📂 Chargement des données...")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    print(f"✅ {len(df)} lignes chargées — {df.is_anomaly.sum()} anomalies réelles")

    # ── 2. Benchmark des modèles de prévision ─────────────
    print("\n" + "=" * 50)
    print("🤖 Entraînement des modèles de prévision...")
    mlflow.set_experiment('finops-forecasting')

    sarima_results  = {}
    prophet_results = {}
    baseline_results = {}

    for service in SERVICES:
        print(f"\n--- {service} ---")
        sarima_results[service]   = train_sarima(df, service)
        prophet_results[service]  = train_prophet(df, service)
        baseline_results[service] = train_baseline(df, service)

    # ── 3. Résumé du benchmark ────────────────────────────
    print("\n" + "=" * 50)
    print("📊 Résumé du benchmark :")
    print(f"{'Service':<12} {'Baseline':>10} {'SARIMA':>10} {'Prophet':>10}")
    print("-" * 45)
    for service in SERVICES:
        b = baseline_results[service]['mape']
        s = sarima_results[service]['mape']
        p = prophet_results[service]['mape']
        print(f"{service:<12} {b:>9.1f}% {s:>9.1f}% {p:>9.1f}%")

    # Meilleur modèle
    mape_sarima  = sum(sarima_results[s]['mape']  for s in SERVICES) / len(SERVICES)
    mape_prophet = sum(prophet_results[s]['mape'] for s in SERVICES) / len(SERVICES)
    best_model   = "Prophet" if mape_prophet < mape_sarima else "SARIMA"

    print(f"\nMAPE moyen SARIMA  : {mape_sarima:.1f}%")
    print(f"MAPE moyen Prophet : {mape_prophet:.1f}%")
    print(f"✅ Meilleur modèle : {best_model}")

    # ── 4. Détection d'anomalies ──────────────────────────
    print("\n" + "=" * 50)
    print("🔍 Détection d'anomalies (Isolation Forest)...")
    mlflow.set_experiment('finops-anomaly-detection')

    clf, scaler, pivot = train_isolation_forest(df)
    n_anomalies = (pivot['if_label'] == -1).sum()
    print(f"✅ {n_anomalies} jours anormaux détectés sur {len(pivot)} jours")

    print("\n" + "=" * 50)
    print("🚀 Projet prêt !")
    print("→ API     : http://localhost:8000/docs")
    print("→ MLflow  : http://localhost:5000")
    print("→ Grafana : http://localhost:3000")

if __name__ == '__main__':
    main()
