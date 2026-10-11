# Tâches restantes – industrial-vlm-inspection

## État actuel du projet

Le projet dispose déjà d'une base fonctionnelle :
- architecture générale du système d'inspection VLM,
- scripts de chargement du modèle Qwen2.5-VL,
- pipeline d'inférence,
- interface Gradio,
- script de fine-tuning LoRA.

Cependant, plusieurs éléments restent à finaliser avant que le projet soit prêt pour un usage fiable en production ou en démonstration industrielle.

## Tâches restantes à traiter

### 1. Finaliser la structure des modules Python
- [ ] Corriger la nomenclature du module de données (`src/data_source` vs `src/dataset.py`) pour éviter les imports cassés.
- [ ] Vérifier que les imports `from .dataset import ...` fonctionnent dans tout le projet.
- [ ] Harmoniser les exports dans `src/__init__.py` et la structure du package.

### 2. Préparer le jeu de données complet
- [ ] Vérifier la présence des données brutes dans `data/raw/`.
- [ ] Préparer les images traitées dans `data/processed/`.
- [ ] Créer un jeu d'annotations de référence au format JSON exploitable par `IndustrialDataset`.
- [ ] Définir les classes de défauts et leur taxonomie (rayure, fissure, corrosion, etc.).
- [ ] Valider la cohérence des chemins `image_path` et la structure des échantillons.

### 3. Rendre le pipeline d'entraînement exécutable
- [ ] Tester le script `src/train.py` sur un petit échantillon de données.
- [ ] Vérifier les paramètres de configuration LoRA et du modèle.
- [ ] Contrôler la compatibilité GPU / CPU selon l'environnement de déploiement.
- [ ] Vérifier les dépendances spécifiques à Transformers, PEFT, TRL et Qwen2.5-VL.
- [ ] Définir une stratégie de sauvegarde des checkpoints (`checkpoints/vlm-inspection`).

### 4. Valider le pipeline d'inférence
- [ ] Tester l'analyse d'une image de pièce avec le mode standard.
- [ ] Tester le mode quantifié 4-bit et 8-bit selon la machine cible.
- [ ] Vérifier le parsing du verdict et des défauts retournés par le modèle.
- [ ] Valider le comportement du grounding (bounding boxes) sur un cas réel.
- [ ] Mesurer les temps d'inférence et la stabilité du modèle.

### 5. Mettre en place l'évaluation qualité
- [ ] Définir des métriques de performance : précision, rappel, F1, taux de détection par type de défaut.
- [ ] Créer un petit jeu de test de validation avec étiquettes connues.
- [ ] Comparer les résultats avant / après fine-tuning.
- [ ] Documenter les seuils de qualité pour un verdict conforme / non conforme.

### 6. Finaliser l'interface utilisateur
- [ ] Vérifier le comportement de l'interface Gradio sur un vrai dataset.
- [ ] Tester le chatbot opérateur avec plusieurs questions.
- [ ] Vérifier la robustesse si aucune image n'est fournie.
- [ ] Ajouter des messages d'erreur plus explicites et des états de chargement.

### 7. Vérifier le déploiement applicatif
- [ ] Contrôler le `Dockerfile` de déploiement.
- [ ] Vérifier le port, le host et les paramètres de lancement de l'application.
- [ ] Tester le conteneur sur une machine sans GPU ou avec GPU CUDA.
- [ ] Documenter les commandes de lancement pour l'usine / environnement de production.

### 8. Documentation et usage
- [ ] Compléter le `README.md` avec les commandes de setup et d'utilisation.
- [ ] Ajouter une section "Pré-requis" (Python, CUDA, dépendances, volumes de données).
- [ ] Ajouter des exemples d'invocation pour le training, l'inférence et le service Gradio.
- [ ] Documenter la structure des fichiers de données et des annotations.

### 9. Robustesse opérationnelle
- [ ] Ajouter un mécanisme de logs plus structuré pour les déploiements industriels.
- [ ] Vérifier la gestion des erreurs sur fichiers manquants, images corrompues ou modèles absents.
- [ ] Prévoir une stratégie de réentraînement et de versionnement des checkpoints.
- [ ] Évaluer le besoin d'un backend API plus stable que Gradio pour production.

### 10. Prochaine étape recommandée
- [ ] Valider le projet sur un sous-ensemble réaliste de données.
- [ ] Produire une première version fonctionnelle d'évaluation.
- [ ] Définir la feuille de route pour une version v1 utilisable en environnement industriel.

## Priorité recommandée

1. Correction de la structure des modules et imports
2. Préparation et validation du dataset
3. Test du fine-tuning sur un petit jeu de données
4. Validation de l'inférence et du verdict
5. Finalisation de l'interface et du déploiement
6. Documentation et mise en conformité opérationnelle

## Résumé

Le projet est bien avancé au niveau de l'architecture et du prototype technique, mais il reste encore les tâches critiques du pipeline de données, de la validation du modèle et de la préparation du déploiement. La priorité immédiate est de rendre le code exécutable et cohérent, puis d'obtenir une première démonstration fiable sur données réelles.
