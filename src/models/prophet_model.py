from prophet import Prophet
import pandas as pd, mlflow


def train_prophet(df: pd.DataFrame, service: str) -> dict:
    # Prophet attend les colonnes 'ds' et 'y'
    ts = (df[df.service == service]
          .groupby('date')['cost_usd'].sum()
          .reset_index()
          .rename(columns={'date': 'ds', 'cost_usd': 'y'}))

    # Split train/test (80/20)
    split = int(len(ts) * 0.8)
    train, test = ts[:split], ts[split:]

    with mlflow.start_run(run_name=f'prophet_{service}'):
        mlflow.set_experiment('finops-forecasting')  # Pour les modèles de prévision

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05  # régularisation
        )
        model.fit(train)

        # Prédictions
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)
        pred = forecast.tail(len(test))['yhat'].values

        # Métriques
        mape = (abs(pred - test.y.values) / test.y.values).mean() * 100
        #MAPE c est le pourcentage moyen d ecart entre la realité et la prédiction par jour
        rmse = ((pred - test.y.values) ** 2).mean() ** 0.5


        mlflow.log_params({'service': service, 'changepoint_prior': 0.05})
        mlflow.log_metrics({'mape': mape, 'rmse': rmse})

        print(f'{service} — PROPHET MAPE: {mape:.1f}%, RMSE: {rmse:.0f}$')
        return {'model': model, 'mape': mape, 'rmse': rmse,
                'predictions': pred, 'test': test}
