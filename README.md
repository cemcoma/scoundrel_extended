# Scoundrel: Card Game & RL Agent

A Python implementation of the roguelike card game [Scoundrel](http://stfj.net/art/2011/Scoundrel.pdf) (originally designed by Zach Gage and Kurt Bieg). This repository features both a fully playable Command-Line Interface (CLI) version and a custom Reinforcement Learning (RL) environment to train an AI agent using Temporal Difference (TD) Q-Learning.

## 🚀 Features & WIP Roadmap

**Current Implementations:**
* **CLI Version:** Play the base Scoundrel ruleset directly in your favourite terminal!
* **RL Agent:** A model-free Q-Learning agent utilizing Linear Function Approximation, action masking, and an engineered 28-dimensional state space. 

**Extended Ruleset (WIP):**
1. **Player Classes:** Buffs to different value cards (e.g., reusability, inflicting more damage, lifesteal, etc.).
2. **Weapon Types:** Distinct weapon mechanics and durability logic.
3. **Extended Dungeons:** Infinite modes and expanded decks beyond the standard 44-card draw pile.
4. **Jokers:** Special card interactions and wildcards.

## 📦 Dependencies

The base CLI game runs on standard Python. 

The Reinforcement Learning environment and weight visualizations require the following data science libraries:

* `python >= 3.8`
* `numpy` (for state matrix operations and Q-value calculations)
* `gymnasium` (for the custom RL environment API)
* `pandas`, `seaborn`, `matplotlib` (for weight matrix heatmaps and visualization)

**Installation:**
```bash
pip install numpy gymnasium pandas seaborn matplotlib
```

## 🎮 Usage
* Play the Game (CLI)

Just run the CLI file to play on your favorite command line interpreter:
```bash
python scoundrel_cli.py
```
* Train & Simulate the AI

1. Train the Agent:
Run the environment script to initialize the Gymnasium environment, train the Q-Learning agent via ϵ-greedy exploration, and output the trained_scoundrel_weights.npy file along with an SVG heatmap.

```bash
python scoundrel_ai_rl.py
```

2. Watch the Simulation:
Run the simulation script to load the trained weights and watch the AI navigate a game sequentially with 0 exploration. If the AI wins, it saves the initial deck state to a text file.

```bash
python simulate.py
```