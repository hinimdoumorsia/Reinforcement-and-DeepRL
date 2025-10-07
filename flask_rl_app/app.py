import sys
import os
from flask import Flask, render_template, request, jsonify
import threading
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "envs"))
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

from grid_env import GridWorld
from RandomAgent import RandomAgent
from MonteCarloAgent import MonteCarloAgent
from QLearningAgent import QLearningAgent
from PolicyIteration import PolicyIterationAgent
from ValueIteration import ValueIterationAgent

app = Flask(__name__)

AGENT_CLASSES = {
    "RandomAgent": RandomAgent,
    "MonteCarloAgent": MonteCarloAgent,
    "QLearningAgent": QLearningAgent,
    "PolicyIterationAgent": PolicyIterationAgent,
    "ValueIterationAgent": ValueIterationAgent
}

# Variables globales
current_data = {
    "positions": [],
    "errors": [],
    "goals": [],
    "obstacles": [],
    "size": 6,
    "training_complete": False,
    "current_episode": 0,
    "total_episodes": 100,
    "agent_name": ""
}

simulation_env = None
simulation_agent = None
training_active = False

def convert_to_serializable(obj):
    if isinstance(obj, (np.int32, np.int64, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html", agents=AGENT_CLASSES.keys())

@app.route("/start_training", methods=["POST"])
def start_training():
    global current_data, simulation_env, simulation_agent, training_active
    
    if training_active:
        return jsonify({"status": "error", "message": "Training already in progress"})
    
    data = request.json
    
    # Réinitialiser les données
    current_data = {
        "positions": [],
        "errors": [],
        "goals": [],
        "obstacles": [],
        "size": data.get("grid_size", 6),
        "training_complete": False,
        "current_episode": 0,
        "total_episodes": data.get("episodes", 100),
        "agent_name": data.get("agent", "RandomAgent")
    }
    
    # Configurer l'environnement
    size = data.get("grid_size", 6)
    goals = parse_positions(data.get("goal_positions", ""), data.get("num_goals", 1), size)
    obstacles = parse_positions(data.get("obstacle_positions", ""), data.get("num_obstacles", 3), size)
    
    if not goals:
        goals = [tuple(np.random.randint(0, size, 2)) for _ in range(data.get("num_goals", 1))]
    if not obstacles:
        obstacles = [tuple(np.random.randint(0, size, 2)) for _ in range(data.get("num_obstacles", 3))]
    
    current_data["goals"] = goals
    current_data["obstacles"] = obstacles
    
    # Créer l'environnement
    simulation_env = GridWorld(
        size=size,
        goal_positions=goals,
        obstacles=obstacles,
        start_pos=(0, 0),
        max_steps=100
    )
    
    # Créer l'agent
    agent_name = data.get("agent", "RandomAgent")
    agent_class = AGENT_CLASSES[agent_name]
    
    try:
        if agent_name == "QLearningAgent":
            simulation_agent = agent_class(
                simulation_env,
                alpha=float(data.get("alpha", 0.1)),
                gamma=float(data.get("gamma", 0.9)),
                epsilon=float(data.get("epsilon", 0.1))
            )
        elif agent_name in ["MonteCarloAgent", "PolicyIterationAgent", "ValueIterationAgent"]:
            simulation_agent = agent_class(simulation_env, gamma=float(data.get("gamma", 0.9)))
        else:
            simulation_agent = agent_class(simulation_env)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Agent creation failed: {str(e)}"})
    
    # Démarrer l'entraînement
    training_active = True
    thread = threading.Thread(
        target=run_training,
        args=(data.get("episodes", 100), agent_name)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "training_started"})

def parse_positions(position_str, num_positions, grid_size):
    if not position_str:
        return []
    
    positions = []
    for pos in position_str.split(','):
        if '-' in pos:
            try:
                x, y = map(int, pos.strip().split('-'))
                if 0 <= x < grid_size and 0 <= y < grid_size:
                    positions.append((x, y))
            except ValueError:
                continue
    
    return positions if len(positions) == num_positions else []

@app.route("/data")
def get_data():
    return jsonify(convert_to_serializable(current_data))

@app.route("/reset", methods=["POST"])
def reset_simulation():
    global current_data, training_active
    current_data["positions"] = []
    current_data["errors"] = []
    current_data["training_complete"] = False
    current_data["current_episode"] = 0
    training_active = False
    return jsonify({"status": "reset"})

def run_training(episodes, agent_name):
    global current_data, training_active
    
    try:
        print(f"🚀 Starting {agent_name} training for {episodes} episodes...")
        
        # Entraînement
        policy, errors = simulation_agent.train(episodes=episodes)
        
        # Mettre à jour les données
        current_data["errors"] = errors
        current_data["training_complete"] = True
        current_data["current_episode"] = episodes
        
        print(f"✅ {agent_name} training completed!")
        
        # Sauvegarde
        try:
            simulation_agent.save_table()
            print(f"💾 {agent_name} tables saved")
        except Exception as e:
            print(f"⚠️ Save failed: {e}")
        
        # Simulation
        run_simulation()
        
    except Exception as e:
        print(f"❌ Training error for {agent_name}: {e}")
        import traceback
        traceback.print_exc()
        current_data["training_complete"] = True
    finally:
        training_active = False

def run_simulation():
    global current_data
    
    if not simulation_env or not simulation_agent:
        return
    
    print("🎮 Starting simulation...")
    
    state = simulation_env.reset()
    done = False
    current_data["positions"] = []
    
    max_steps = 50
    for step in range(max_steps):
        action = simulation_agent.choose_action(state)
        next_state, reward, done, _ = simulation_env.step(action)
        current_data["positions"].append(list(next_state))
        state = next_state
        
        if done:
            print(f"🎯 Goal reached in {step + 1} steps!")
            break
        
        time.sleep(0.3)
    
    print("🏁 Simulation completed")

if __name__ == "__main__":
    app.run(debug=True, threaded=True)