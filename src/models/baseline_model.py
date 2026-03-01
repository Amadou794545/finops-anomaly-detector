import pandas as pd
import mlflow


def train_baseline(df, service, window=7):
    ts = (df[df.service == service]
          .groupby('date')['cost_usd'].sum())

    split = int(len(ts) * 0.8)
    train, test = ts[:split], ts[split:]

    # Moyenne mobile sur 7 jours
    pred = train.rolling(window=window).mean().iloc[-1]
    predictions = [pred] * len(test)

    mape = (abs(pd.Series(predictions).values - test.values) / test.values).mean() * 100
    rmse = ((pd.Series(predictions).values - test.values) ** 2).mean() ** 0.5

    with mlflow.start_run(run_name=f'baseline_{service}'):
        mlflow.log_params({'service': service, 'window': window})
        mlflow.log_metrics({'mape': mape, 'rmse': rmse})

    print(f'{service} — Baseline MAPE: {mape:.1f}%, RMSE: {rmse:.0f}$')
    return {'mape': mape, 'rmse': rmse, 'predictions': predictions}
