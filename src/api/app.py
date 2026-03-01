from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import date
import pandas as pd, pickle, os

app = FastAPI(
    title='FinOps Anomaly Detector api',
    description='Prévision des coûts Cloud et détection d anomalies',
    version='1.0.0'
)


class ForecastRequest(BaseModel):
    service: str  # Ex: 'EC2', 'S3', 'RDS'
    horizon: int = 7  # Nb de jours à prévoir


class AnomalyRequest(BaseModel):
    date: str
    service: str
    cost_usd: float


@app.get('/')
def root():
    return {'status': 'ok', 'project': 'FinOps Anomaly Detector'}


@app.get('/health')
def health():
    return {'status': 'healthy'}


# /services : retourne la liste des services cloud disponibles
@app.get('/services')
def list_services():
    return ['EC2', 'S3', 'RDS', 'Lambda', 'CloudFront', 'EKS', 'Redshift']


# /forecast : reçoit un service et un horizon, retourne les prévisions de coûts pour les prochains jours
@app.post('/forecast')
def get_forecast(req: ForecastRequest):
    # Charger le modèle Prophet sauvegardé
    model_path = f'models/{req.service}_prophet.pkl'
    if not os.path.exists(model_path):
        raise HTTPException(404, f'Modèle non trouvé: {req.service}')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    future = model.make_future_dataframe(periods=req.horizon)
    forecast = model.predict(future).tail(req.horizon)
    return {
        'service': req.service,
        'horizon': req.horizon,
        'forecasts': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        .to_dict('records')
    }


# /detect : reçoit une observation et retourne si c'est une anomalie ou pas
@app.post('/detect')
def detect_anomaly(req: AnomalyRequest):
    # Logique de détection simplifiée pour l'api
    # En prod : charger le modèle Isolation Forest
    return {
        'date': req.date,
        'service': req.service,
        'cost_usd': req.cost_usd,
        'is_anomaly': False,  # Remplacer par la vraie inférence
        'score': 0.0
    }


# /cost : retourne les coûts journaliers totaux des 90 derniers jours
@app.get('/costs')
def get_costs():
    df = pd.read_csv('data/raw/cloud_costs.csv', parse_dates=['date'])
    daily = (df.groupby('date')['cost_usd']
             .sum()
             .reset_index()
             .tail(90))  # 90 derniers jours
    return daily.to_dict('records')


@app.get('/forecast/{service}')  # à ajouter
def get_service_forecast(service: str, horizon: int = 7):
    # Charger le modèle Prophet sauvegardé
    model_path = f'models/{service}_prophet.pkl'
    if not os.path.exists(model_path):
        raise HTTPException(404, f'Modèle non trouvé: {service}')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future).tail(horizon)
    return {
        'service': service,
        'horizon': horizon,
        'forecasts': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        .to_dict('records')
    }
