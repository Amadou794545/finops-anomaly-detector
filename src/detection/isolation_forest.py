from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd, numpy as np, mlflow


def train_isolation_forest(df: pd.DataFrame, contamination=0.05):
    """
    contamination : proportion d'anomalies attendues (~5% pour des coûts Cloud)
    """
    # Pivoter : 1 ligne = 1 jour, colonnes = coûts par service
    pivot = df.pivot_table(
        index='date', columns='service', values='cost_usd', aggfunc='sum'
    ).fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(pivot)

    with mlflow.start_run(run_name='isolation_forest'):
        mlflow.set_experiment('finops-anomaly-detection')  # Pour la détection
        clf = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42
        )
        clf.fit(X)

        # -1 = anomalie, 1 = normal
        labels = clf.predict(X)
        scores = clf.score_samples(X)  # Plus négatif = plus anormal

        n_anomalies = (labels == -1).sum()
        mlflow.log_metric('n_anomalies_detected', int(n_anomalies))
        mlflow.log_param('contamination', contamination)

    pivot['if_label'] = labels
    pivot['if_score'] = scores
    print(f'Isolation Forest détecte {n_anomalies} anomalies sur {len(pivot)} jours ({n_anomalies/len(pivot)*100:.1f}%)')
    return clf, scaler, pivot
