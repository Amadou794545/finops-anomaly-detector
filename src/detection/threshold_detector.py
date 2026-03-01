import pandas as pd, numpy as np


def detect_with_forecast(df_real, forecast, ci_multiplier=2.5):
    """
    Détecte les anomalies en comparant coûts réels vs bandes de prévision.
    ci_multiplier : sensibilité (plus élevé = moins de faux positifs)
    """
    # Prophet fournit yhat_lower et yhat_upper (intervalle 80%)
    # On élargit l'intervalle avec ci_multiplier
    residuals = df_real['cost_usd'] - forecast['yhat']
    std_resid = residuals.std()

    lower = forecast['yhat'] - ci_multiplier * std_resid
    upper = forecast['yhat'] + ci_multiplier * std_resid

    anomalies = (df_real['cost_usd'] < lower) | (df_real['cost_usd'] > upper)

    return pd.DataFrame({
        'date': df_real['date'],
        'cost_usd': df_real['cost_usd'],
        'forecast': forecast['yhat'],
        'lower_bound': lower,
        'upper_bound': upper,
        'is_anomaly': anomalies,
        'anomaly_score': abs(residuals) / std_resid
    })
