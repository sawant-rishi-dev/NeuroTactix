# 🧠 Psychological Tic-Tac-Toe: NeuroTactix

!![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production-success?style=for-the-badge)

A state-of-the-art AI experiment that combines **Monte Carlo Tree Search (MCTS)** and **Deep Reinforcement Learning (DQN)** to create an unbeatable Tic-Tac-Toe opponent.

This project features a **Psychological Engine** that taunts the player based on win probability and a **Service-Side Debug Dashboard** for real-time neural telemetry.

---

## 🚀 Key Features

### 🤖 Dual-Core AI
* **God Mode (MCTS):** Uses Monte Carlo Tree Search with 2000+ simulations per move. Mathematically unbeatable (can only win or draw).
* **Adaptive DQN:** A Dense Neural Network that learns patterns from game history without Convolutional overhead.

### 🧠 Psychological Engine
* **Win Probability Gauge:** A dynamic UI bar that shifts from Green (Winning) to Red (Losing) based on the AI's confidence.
* **Trash Talk System:** Context-aware taunts based on board state (e.g., *"Checkmate in 3 moves"* or *"That was a suboptimal error"*).

### 📡 Service-Side Telemetry (New!)
* **Hacker Dashboard:** A secondary window providing real-time logs of the AI's "thought process."
* **Live Metrics:** View Visit Counts, Q-Values, Epsilon decay, and Reward propagation as they happen.

---

## 📂 Modular Architecture

The project has been refactored from a monolithic script into a scalable, domain-driven architecture:

text
├── main.py              # Entry point. Handles app lifecycle and Win/Loss logic.
├── config.py            # Global configuration, constants, and assets.
├── game_engine.py       # Pure game rules (State, Valid Moves, Winner Check).
├── ai_brain.py          # The Intelligence (MCTS & DQN Neural Network classes).
└── ui_components.py     # Custom UI widgets (Gauges, Debug Dashboard).




1. The Turing Test (Play vs. AI)
Mode Selection:
<img width="919" height="169" alt="image" src="https://github.com/user-attachments/assets/ca9d9c06-e4f9-4684-8635-04d557704beb" />
God Mode: Select to play against the MCTS engine. Prepare for a draw or loss.

Hard/Medium: Select to play against the trained Neural Network (strategic_agent_v1.h5).

OR Train your own model using 
<img width="552" height="74" alt="image" src="https://github.com/user-attachments/assets/90d17aef-4894-409a-b1ad-270787932274" />

then play little with the parameters 
<img width="960" height="1033" alt="image" src="https://github.com/user-attachments/assets/61d66d42-9237-4b85-b90c-43607d9df25f" />
<img width="688" height="742" alt="image" src="https://github.com/user-attachments/assets/31e2785c-dd25-40ac-9862-f8271057fc6e" />

Feedback: Watch the Win Probability Gauge at the top. Green means the AI is confident; Red means you might have a chance.

Taunts: Pay attention to the text below the bar for insights into the AI's confidence level.

2. The Engineer's View (Debugging & Telemetry)
Launch: Click the "Open Debug Dashboard" button (or press Ctrl+D).

Monitor: A secondary window will open showing real-time logs.
<img width="686" height="811" alt="image" src="https://github.com/user-attachments/assets/65de8aa2-cff6-49c7-b145-23cd4a41d74d" />

During Gameplay: View the AI's decision logic (e.g., "MCTS simulations: 300").
<img width="940" height="1022" alt="image" src="https://github.com/user-attachments/assets/5a0125fc-5568-47e5-8d90-9124f0fa1477" />
During Training: Watch the Training Monitor for live updates on Reward, Loss, and Epsilon values.

3. The Battle Royale (Model Arena)
Objective: Scientifically prove which AI model is superior.

Setup: Go to the Model Arena tab.

Select Combatants: Choose multiple models from the list. Check "Include MCTS God Mode" to benchmark against perfection.

Execute: Set "Games per Matchup" to 50 and click Start Tournament. The system will auto-play and report win rates.
