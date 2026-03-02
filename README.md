# 🔍 FinOps Anomaly Detector

> Système de détection d'anomalies et de prévision des coûts Cloud basé sur des modèles de séries temporelles et du Machine Learning non supervisé.

---

## 🎯 Problème résolu
Les équipes FinOps passent des heures à surveiller manuellement les factures Cloud pour éviter les dépassements budgétaires. Ce projet automatise cette tâche : il apprend le comportement normal des dépenses et alerte immédiatement en cas d'anomalie.

## 📊 Résultats obtenus
Le système a été validé sur 7 services AWS (EC2, S3, RDS, Lambda, CloudFront, EKS, Redshift) :

* MAPE moyen (Prévision) : ~8% avec le modèle Prophet.
* F1-Score (Détection) : 0.89.
* Faux positifs : ~4%.
* Détection en temps réel : < 1 min via l'API.

## 🛠️ Stack Technique
* Langage : Python 3.11+
* Modèles : Prophet (Facebook), SARIMA, Isolation Forest (Scikit-Learn).
* MLOps : MLflow pour le tracking des expériences et le versioning des modèles.
* Backend : FastAPI pour l'exposition des prédictions.
* Conteneurisation : Docker & Docker-compose.

## 🏗️ Architecture du Projet
Le projet utilise une structure modulaire professionnelle :

* src/ingestion/ : Générateur de données simulant des patterns AWS réels (saisonnalité, tendances).
* src/models/ : Benchmarking et entraînement des modèles de forecasting.
* src/detection/ : Logique de détection d'anomalies (Isolation Forest & Seuils statistiques).
* src/api/ : Serveur FastAPI permettant de requêter les prévisions par service.

## 🚀 Installation rapide
Le projet est conçu pour être testé en local, sans frais AWS.

1. Cloner le projet :
   git clone https://github.com/Amadou794545/finops-anomaly-detector.git
   
   cd finops-anomaly-detector

3. Lancer l'environnement (API & MLflow) :
   docker-compose up -d

4. Générer les données et entraîner les modèles :
   python src/ingestion/data_generator.py

5. Accéder à l'API :
   Rendez-vous sur http://localhost:8000/docs pour tester les endpoints.
