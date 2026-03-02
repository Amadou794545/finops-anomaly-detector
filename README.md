# 🔍 FinOps Anomaly Detector

> Système de détection d'anomalies et de prévision des coûts Cloud basé sur des modèles de séries temporelles et du Machine Learning non supervisé.

---

## 🎯 Objectif du Projet
[cite_start]L'objectif est de résoudre un problème concret rencontré par les équipes Cloud : la peur de la "facture surprise"[cite: 4, 128]. [cite_start]Ce système automatise la surveillance des coûts en apprenant les patterns normaux et en alertant dès qu'une dérive est détectée[cite: 224, 226].

## 📊 Résultats & Métriques
[cite_start]Le projet a été validé sur 7 services AWS (EC2, S3, RDS, Lambda, CloudFront, EKS, Redshift) avec les performances suivantes[cite: 76, 222, 481]:

* [cite_start]**MAPE moyen (Prévision)** : ~6-8% avec le modèle Prophet [cite: 222, 483]
* [cite_start]**F1-Score (Détection)** : 0.89 [cite: 484]
* [cite_start]**Faux Positifs** : 4.1% [cite: 485]
* [cite_start]**Fréquence d'analyse** : Journalière avec détection < 1 min via API [cite: 83, 486]

## 🛠️ Stack Technique
* [cite_start]**Langage** : Python 3.11+ [cite: 36, 457]
* [cite_start]**Séries Temporelles** : Prophet (Facebook) & SARIMA [cite: 151, 184]
* [cite_start]**Machine Learning** : Isolation Forest (Scikit-Learn) [cite: 251, 252]
* [cite_start]**MLOps** : MLflow pour le tracking des expériences [cite: 293, 301]
* [cite_start]**API** : FastAPI pour l'exposition des modèles en temps réel [cite: 307, 313]
* [cite_start]**Infrastructure** : Docker & Docker-Compose [cite: 367, 398]

## 🏗️ Architecture du Code
[cite_start]Le projet respecte une structure professionnelle prête pour la production[cite: 7, 8, 9]:

* [cite_start]`src/ingestion/` : Générateur de données simulant la saisonnalité et les tendances AWS[cite: 72, 95, 97].
* [cite_start]`src/models/` : Scripts d'entraînement Prophet et SARIMA[cite: 151, 184].
* [cite_start]`src/detection/` : Logique de détection par Isolation Forest et seuils statistiques[cite: 227, 251].
* [cite_start]`src/api/` : API REST permettant de requêter les prévisions par service[cite: 308].
* [cite_start]`tests/` : Tests unitaires garantissant la qualité du code (Pytest)[cite: 411, 414].

## 🚀 Installation et Lancement
[cite_start]Le projet est conçu pour être lancé en local sans avoir besoin d'un compte AWS[cite: 34].

1. **Cloner le projet** :
   git clone [https://github.com/TON_USERNAME/finops-anomaly-detector](https://github.com/Amadou794545/finops-anomaly-detector.git)
   cd finops-anomaly-detector

2. **Installer les dépendances** :
   pip install -r requirements.txt

3. **Lancer l'environnement MLOps & API** :
   docker-compose up -d
   python src/ingestion/data_generator.py

4. **Accéder à l'API** :
   Rendez-vous sur http://localhost:8000/docs pour tester les prédictions en direct.

## 🧪 Industrialisation (CI/CD)

[cite_start]Une pipeline GitHub Actions est intégrée pour exécuter automatiquement les tests unitaires et vérifier le style du code à chaque modification (flake8)[cite: 443, 461, 464].

