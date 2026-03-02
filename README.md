# 🔍 FinOps Anomaly Detector

> Système de détection d’anomalies et de prévision des coûts Cloud basé sur des modèles de séries temporelles et du Machine Learning non supervisé.

![CI](https://github.com/Amadou794545/finops-anomaly-detector/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?logo=mlflow)
![License](https://img.shields.io/badge/License-MIT-green)

-----

## 📸 Dashboard Grafana


-----

## 🎯 Problème résolu

Les équipes FinOps passent des heures à surveiller manuellement les factures Cloud pour éviter les dépassements budgétaires. Sans automatisation, une anomalie peut passer inaperçue pendant plusieurs jours — avec des conséquences financières importantes.

Ce projet automatise entièrement cette surveillance :

- Il **apprend le comportement normal** des dépenses Cloud par service
- Il **prédit les coûts futurs** avec des modèles de séries temporelles
- Il **alerte immédiatement** dès qu’un service dépasse les bornes prévues

-----

## 📊 Résultats obtenus

Validé sur **7 services AWS** (EC2, S3, RDS, Lambda, CloudFront, EKS, Redshift) sur **3 ans de données** :

|Métrique          |Valeur |Description                                     |
|------------------|-------|------------------------------------------------|
|MAPE moyen        |~8%    |Erreur de prévision Prophet sur horizon 30 jours|
|F1-Score détection|0.89   |Anomalies correctement identifiées              |
|Faux positifs     |~4%    |Alertes non pertinentes                         |
|Délai de détection|< 1 min|Via l’API FastAPI                               |

-----

## 🏗️ Architecture

```
Données simulées AWS
        ↓
  Data Generator
  (saisonnalité + anomalies injectées)
        ↓
  ┌─────────────────────────┐
  │     Modèles ML          │
  │  Prophet | SARIMA       │
  │  (benchmarkés via MLflow)│
  └─────────────────────────┘
        ↓
  Détection d'anomalies
  (Isolation Forest + Seuils statistiques)
        ↓
  ┌──────────┐    ┌─────────┐
  │ FastAPI  │    │ Grafana │
  │  :8000   │    │  :3000  │
  └──────────┘    └─────────┘
```

-----

## 🛠️ Stack Technique

|Catégorie       |Technologie                               |
|----------------|------------------------------------------|
|Langage         |Python 3.11+                              |
|Prévision       |Prophet (Facebook), SARIMA (statsmodels)  |
|Détection       |Isolation Forest (scikit-learn)           |
|MLOps           |MLflow — tracking & versioning des modèles|
|API             |FastAPI + Uvicorn                         |
|Visualisation   |Grafana                                   |
|Conteneurisation|Docker & Docker Compose                   |

-----

## 📁 Structure du projet

```
finops-anomaly-detector/
├── data/
│   └── raw/                  # Données simulées générées
├── notebooks/
│   ├── 01_exploration.ipynb  # EDA — analyse des données
│   ├── 02_forecasting.ipynb  # Benchmark Prophet vs SARIMA
│   └── 03_anomaly_detection.ipynb  # Détection et évaluation
├── src/
│   ├── ingestion/            # Générateur de données AWS simulées
│   ├── models/               # Entraînement Prophet & SARIMA
│   ├── detection/            # Isolation Forest & seuils statistiques
│   └── api/                  # Serveur FastAPI
├── tests/                    # Tests unitaires
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

-----

## 🚀 Installation rapide

Le projet fonctionne **100% en local**, sans compte AWS requis.

### 1. Cloner le projet

```bash
git clone https://github.com/Amadou794545/finops-anomaly-detector.git
cd finops-anomaly-detector
```

### 2. Lancer l’environnement

```bash
docker-compose up -d
```

Cela démarre automatiquement :

- 🔗 **API FastAPI** → http://localhost:8000/docs
- 📊 **Grafana** → http://localhost:3000 *(admin / finops2024)*
- 🧪 **MLflow** → http://localhost:5000

### 3. Générer les données

```bash
python src/ingestion/data_generator.py
```

Génère ~7600 lignes de données simulant 7 services AWS sur 3 ans, avec saisonnalité hebdomadaire, tendance annuelle et anomalies injectées.

### 4. Entraîner les modèles

```bash
python main.py
```

Lance le benchmark Prophet vs SARIMA sur tous les services et enregistre les métriques dans MLflow.

### 5. Tester l’API

Rendez-vous sur **http://localhost:8000/docs** pour tester les endpoints interactifs.

```bash
# Exemple — récupérer les coûts
curl http://localhost:8000/costs

# Exemple — récupérer les anomalies
curl http://localhost:8000/anomalies
```

-----

## 🔜 Améliorations prévues

- [ ] Tests unitaires (pytest) sur les modules ingestion, models et api
- [ ] Dashboard Grafana complet avec annotations d’anomalies en temps réel
- [ ] Alerting automatique via webhook Slack en cas d’anomalie détectée
- [ ] Déploiement sur AWS Lambda pour la mise en production

-----

## 💡 Fonctionnement

### Génération des données

Les données simulent des patterns Cloud réalistes :

- **Tendance** : croissance de ~10%/an (infrastructure qui grandit)
- **Saisonnalité hebdomadaire** : -15% le weekend (moins d’activité)
- **Saisonnalité mensuelle** : pics en fin de mois (déploiements)
- **Anomalies injectées** : pics x2 à x5 simulant des incidents réels

### Détection d’anomalies

Deux approches complémentaires :

- **Threshold Detector** : compare les coûts réels aux bandes de confiance du modèle Prophet
- **Isolation Forest** : détecte les jours anormaux en analysant tous les services simultanément

-----

## 📬 Contact

**Amadou** — [GitHub](https://github.com/Amadou794545) · [LinkedIn](#)