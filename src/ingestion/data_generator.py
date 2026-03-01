import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SERVICES = ['EC2', 'S3', 'RDS', 'Lambda', 'CloudFront', 'EKS', 'Redshift']
REGIONS  = ['eu-west-1', 'us-east-1', 'ap-southeast-1']

def generate_cloud_costs(
    start='2022-01-01', end='2024-12-31',
    n_anomalies=40, seed=42
) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range(start, end, freq='D')
    records = []

    # Coûts de base par service ($/jour)
    base_costs = {
        'EC2': 450, 'S3': 80, 'RDS': 200,
        'Lambda': 30, 'CloudFront': 60,
        'EKS': 320, 'Redshift': 150
    }

    for service, base in base_costs.items():
        costs = []
        for i, date in enumerate(dates):
            # Tendance croissante (croissance cloud réaliste)
            trend = base * (1 + 0.0003 * i)
            # Saisonnalité hebdomadaire (moins cher le weekend)
            weekly = 1.0 - 0.15 * (date.weekday() >= 5)
            # Saisonnalité mensuelle (fin de mois = déploiements)
            monthly = 1.0 + 0.1 * np.sin(2*np.pi*date.day/30)
            # Bruit gaussien
            noise = np.random.normal(1.0, 0.05)
            cost = trend * weekly * monthly * noise
            costs.append(max(cost, 0))

        # Injecter des anomalies aléatoires
        anomaly_mask = np.zeros(len(dates), dtype=bool)
        anomaly_idx = np.random.choice(
            len(dates), size=n_anomalies//len(base_costs), replace=False
        )
        for idx in anomaly_idx:
            # Spike x2 à x5
            multiplier = np.random.uniform(2.0, 5.0)
            costs[idx] *= multiplier
            anomaly_mask[idx] = True

        for i, (date, cost) in enumerate(zip(dates, costs)):
            records.append({
                'date': date, 'service': service,
                'cost_usd': round(cost, 2),
                'region': np.random.choice(REGIONS),
                'is_anomaly': anomaly_mask[i]
            })

    return pd.DataFrame(records)

if __name__ == '__main__':
    df = generate_cloud_costs()
    df.to_csv('data/raw/cloud_costs.csv', index=False)
    print(f'Généré {len(df)} lignes, {df.is_anomaly.sum()} anomalies')
