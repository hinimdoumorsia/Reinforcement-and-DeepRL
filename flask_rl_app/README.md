# 🧠 Reinforcement-and-DeepRL

## 🚀 Description
Ce projet est une **application Flask interactive** dédiée à l’apprentissage par renforcement dans un environnement **GridWorld**.  
Elle permet d’expérimenter et de visualiser le comportement de **plusieurs agents** selon différents algorithmes, tout en offrant une **interface intuitive** pour configurer les paramètres d’entraînement.

---

## 📂 Structure du projet

flask_rl_app/
├── agents/
│   ├── random_agent.py
│   ├── policy_iteration.py
│   ├── value_iteration.py
│   ├── montecarlo_agent.py
│   └── qlearning_agent.py
├── trainers/
│   └── trainer.py
├── static/
│   ├── style.css
│   └── scripts.js
├── templates/
│   └── index.html
├── app.py
├── README.md
└── requirements.txt




---

## 🤖 Agents inclus

| Agent | Description |
|-------|------------|
| 🔀 **Random Agent** | Se déplace de façon aléatoire dans le GridWorld, utilisé comme baseline. |
| 🧭 **Policy Iteration Agent** | Apprend une politique optimale via évaluation et amélioration successives. |
| 📊 **Value Iteration Agent** | Calcule la fonction de valeur optimale jusqu’à convergence. |
| 🎲 **Monte Carlo Agent** | Estime les valeurs à partir d’épisodes complets. |
| ⚡ **Q-Learning Agent** | Apprentissage hors-policy par mise à jour incrémentale des Q-valeurs. |

Tous les agents fonctionnent dans le même environnement **GridWorld**, permettant de comparer leurs performances et trajectoires.

---

## 🌐 Interface utilisateur

L’application Flask offre une **interface intuitive** où l’utilisateur peut :  
- Sélectionner l’**agent** à utiliser.  
- Définir des **paramètres personnalisés** :  
  - Nombre d’épisodes  
  - Taux d’apprentissage  
  - Facteur de discount (gamma)  
  - Nombre d’obstacles  
  - Position du **goal**  
  - Valeurs de **reward/punition**  
- Visualiser l’évolution et les **tableaux de convergence**.

---

## ⚙️ Installation

### 1️⃣ Cloner le dépôt
```bash
git clone https://github.com/hinimdoumorsia/Reinforcement-and-DeepRL.git
cd Reinforcement-and-DeepRL/flask_rl_app


