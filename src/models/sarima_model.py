from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')
import mlflow


def train_sarima(df, service):
    ts = (df[df.service == service]
          .groupby('date')['cost_usd'].sum())
    split = int(len(ts) * 0.8)
    train, test = ts[:split], ts[split:]
    with mlflow.start_run(run_name=f'sarima_{service}'):
        mlflow.set_experiment('finops-forecasting')  # Pour les modèles de prévision

        # SARIMA(1,1,1)(1,1,1,7) — saisonnalité hebdomadaire
        model = SARIMAX(train,
                        order=(1, 1, 1), seasonal_order=(1, 1, 1, 7),
                        enforce_stationarity=False)
        fit = model.fit(disp=False)

        pred = fit.forecast(steps=len(test))
        mape = (abs(pred.values - test.values) / test.values).mean() * 100
        rmse = ((pred.values - test.values) ** 2).mean() ** 0.5

        # tout le code d'entraînement ici
        mlflow.log_params({'service': service, 'order': '1,1,1'})
        mlflow.log_metrics({'mape': mape, 'rmse': rmse})

        print(f'{service} — SARIMA MAPE: {mape:.1f}%, RMSE: {rmse:.0f}$')
        return {'model': fit, 'mape': mape, 'rmse': rmse,
            'test': test, 'predictions': pred.values}
