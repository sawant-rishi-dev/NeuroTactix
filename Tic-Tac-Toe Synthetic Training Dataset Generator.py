"""
================================================================================
PSYCHOLOGICAL TIC-TAC-TOE - ADVANCED FLASK BACKEND
================================================================================
This is the "Brain" of the application. It contains:
1. Neural Network for predicting human moves
2. Game logic with adaptive difficulty
3. Minimax algorithm for unbeatable God Mode
4. Pattern recognition for psychological profiling
================================================================================
"""

from flask import Flask, render_template, jsonify, request, session
import numpy as np
import pandas as pd
import os
import random
import json
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# TensorFlow import with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Using fallback prediction.")

# ================================================================================
# FLASK APP CONFIGURATION
# ================================================================================
app = Flask(__name__)
app.secret_key = 'psychological_tic_tac_toe_secret_2024'

# ================================================================================
# GLOBAL GAME STATE & CONFIGURATION
# ================================================================================
GAME_DATA_FILE = 'game_data.csv'
EXISTING_DATASET_FILE = 'tic_tac_toe_dataset.csv'
MODEL_FILE = 'neural_model.h5'
CALIBRATION_GAMES = 5  # Number of games before God Mode activates

# Global variables
game_count = 0
neural_network = None
player_profile = defaultdict(int)  # Tracks player behavior patterns
move_history = []  # Current game moves
all_game_histories = []  # All games for pattern analysis

# ================================================================================
# PART 1: THE "MEMORY" - CSV DATA MANAGEMENT
# ================================================================================
"""
================================================================================
CSV LEARNING SYSTEM EXPLANATION:
--------------------------------------------------------------------------------
The AI keeps a "diary" (CSV file) of every game played. Each row represents
a board state and the move the human made in that situation.

Structure:
- cell_0 to cell_8: The board state (0=empty, 1=X/human, -1=O/AI)
- move_taken: Which cell (0-8) the human chose to play

On startup:
1. Check if game_data.csv exists
2. Also check for any pre-existing dataset (tic_tac_toe_dataset.csv)
3. Combine all data and train the Neural Network
4. The AI now "remembers" all past games!

After each game:
1. All moves from that game are appended to the CSV
2. The model can be retrained with new data (optional incremental learning)
================================================================================
"""

def initialize_csv():
    """
    Initialize the CSV file for storing game data.
    Creates the file with proper headers if it doesn't exist.
    """
    if not os.path.exists(GAME_DATA_FILE):
        columns = [f'cell_{i}' for i in range(9)] + ['move_taken']
        df = pd.DataFrame(columns=columns)
        df.to_csv(GAME_DATA_FILE, index=False)
        print(f"📝 Created new game data file: {GAME_DATA_FILE}")
    else:
        print(f"📂 Found existing game data: {GAME_DATA_FILE}")


def load_all_training_data():
    """
    Load training data from all available sources:
    1. Our game_data.csv (games played on this app)
    2. Existing dataset file (tic_tac_toe_dataset.csv)
    
    Returns combined DataFrame with all training samples.
    """
    all_data = []
    
    # Load our collected game data
    if os.path.exists(GAME_DATA_FILE):
        try:
            df = pd.read_csv(GAME_DATA_FILE)
            if len(df) > 0:
                all_data.append(df)
                print(f"📊 Loaded {len(df)} samples from {GAME_DATA_FILE}")
        except Exception as e:
            print(f"⚠️ Error loading {GAME_DATA_FILE}: {e}")
    
    # Load existing dataset if available
    if os.path.exists(EXISTING_DATASET_FILE):
        try:
            existing_df = pd.read_csv(EXISTING_DATASET_FILE)
            # Standardize column names if different
            expected_cols = [f'cell_{i}' for i in range(9)] + ['move_taken']
            if len(existing_df.columns) == 10:
                existing_df.columns = expected_cols
            if len(existing_df) > 0:
                all_data.append(existing_df)
                print(f"📊 Loaded {len(existing_df)} samples from {EXISTING_DATASET_FILE}")
        except Exception as e:
            print(f"⚠️ Error loading {EXISTING_DATASET_FILE}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"📈 Total training samples: {len(combined)}")
        return combined
    
    return None


def save_game_to_csv(game_moves):
    """
    Save all moves from a completed game to the CSV file.
    
    Args:
        game_moves: List of tuples (board_state, move_taken)
    """
    if not game_moves:
        return
    
    rows = []
    for board_state, move in game_moves:
        row = {f'cell_{i}': board_state[i] for i in range(9)}
        row['move_taken'] = move
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Append to existing CSV
    if os.path.exists(GAME_DATA_FILE):
        df.to_csv(GAME_DATA_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(GAME_DATA_FILE, index=False)
    
    print(f"💾 Saved {len(rows)} moves to {GAME_DATA_FILE}")


# ================================================================================
# PART 2: THE "BRAIN" - NEURAL NETWORK
# ================================================================================
"""
================================================================================
NEURAL NETWORK ARCHITECTURE:
--------------------------------------------------------------------------------
This is the AI's "brain" that learns to predict human moves.

Architecture:
- Input Layer: 9 neurons (one for each cell on the board)
- Hidden Layer 1: 256 neurons with ReLU + BatchNorm + Dropout
- Hidden Layer 2: 128 neurons with ReLU + BatchNorm + Dropout  
- Hidden Layer 3: 64 neurons with ReLU + Dropout
- Output Layer: 9 neurons with Softmax (probability for each cell)

Training:
- The network learns patterns from historical games
- It recognizes which moves humans typically make in certain positions
- Higher accuracy = better "mind reading" ability
================================================================================
"""

class PsychologicalNeuralNetwork:
    """
    Advanced Neural Network for predicting human Tic-Tac-Toe moves.
    Uses pattern recognition to "read" the player's mind.
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.training_accuracy = 0.0
        self.total_samples = 0
        
    def build_model(self):
        """
        Build the neural network architecture.
        This is like building the structure of the brain.
        """
        if not TENSORFLOW_AVAILABLE:
            print("⚠️ TensorFlow not available, using fallback")
            return
            
        self.model = Sequential([
            # Input layer - receives 9 values (the board)
            Dense(256, activation='relu', input_shape=(9,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layer 1 - pattern recognition
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layer 2 - deeper pattern analysis
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            # Hidden layer 3 - final feature extraction
            Dense(32, activation='relu'),
            
            # Output layer - probability for each of 9 cells
            Dense(9, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("🧠 Neural Network architecture built successfully!")
        
    def train(self, df):
        """
        Train the neural network on historical game data.
        
        Args:
            df: DataFrame with board states and moves
        """
        if not TENSORFLOW_AVAILABLE or df is None or len(df) < 10:
            print("⚠️ Insufficient data for training (need at least 10 samples)")
            return
            
        if self.model is None:
            self.build_model()
        
        # Prepare training data
        X = df[[f'cell_{i}' for i in range(9)]].values.astype(np.float32)
        y = df['move_taken'].values.astype(np.int32)
        
        # Filter out invalid moves (must be 0-8)
        valid_mask = (y >= 0) & (y <= 8)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 10:
            print("⚠️ Not enough valid samples after filtering")
            return
            
        self.total_samples = len(X)
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
        
        print(f"🎓 Training on {len(X)} samples...")
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.is_trained = True
        self.training_accuracy = history.history['accuracy'][-1]
        
        print(f"✅ Training complete! Accuracy: {self.training_accuracy:.2%}")
        
    def predict_human_move(self, board):
        """
        Predict where the human will likely click next.
        
        Args:
            board: List of 9 values representing the board
            
        Returns:
            tuple: (predicted_cell, confidence_percentage, all_probabilities)
        """
        if not self.is_trained or not TENSORFLOW_AVAILABLE:
            # Fallback: return random valid cell with low confidence
            valid_moves = [i for i in range(9) if board[i] == 0]
            if valid_moves:
                return random.choice(valid_moves), 15.0, [0.11] * 9
            return 0, 0.0, [0.0] * 9
        
        # Prepare input
        X = np.array(board, dtype=np.float32).reshape(1, 9)
        
        # Get predictions
        predictions = self.model.predict(X, verbose=0)[0]
        
        # Mask invalid moves (cells already taken)
        valid_probs = predictions.copy()
        for i in range(9):
            if board[i] != 0:
                valid_probs[i] = 0
        
        # Normalize probabilities for valid moves only
        if valid_probs.sum() > 0:
            valid_probs = valid_probs / valid_probs.sum()
        
        # Get the most likely move
        predicted_cell = np.argmax(valid_probs)
        confidence = float(valid_probs[predicted_cell]) * 100
        
        return predicted_cell, confidence, valid_probs.tolist()


# ================================================================================
# PART 3: GAME LOGIC & MINIMAX ALGORITHM
# ================================================================================
"""
================================================================================
GAME LOGIC EXPLANATION:
--------------------------------------------------------------------------------
The game has two phases:

PHASE 1 - CALIBRATION (Games 1-5):
- AI plays nice - makes some mistakes on purpose
- Occasionally blocks obvious wins but sometimes "misses"
- Collects data on human playing patterns
- This is the learning phase!

PHASE 2 - GOD MODE (Games 6+):
- AI becomes UNBEATABLE using Minimax algorithm
- Minimax looks at ALL possible future moves
- It picks the move that guarantees best outcome
- Meanwhile, displays predictions to intimidate player
================================================================================
"""

# Cell position names for taunting
CELL_NAMES = {
    0: "Top-Left", 1: "Top-Center", 2: "Top-Right",
    3: "Middle-Left", 4: "Center", 5: "Middle-Right",
    6: "Bottom-Left", 7: "Bottom-Center", 8: "Bottom-Right"
}

# Winning combinations
WINNING_COMBOS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
    [0, 4, 8], [2, 4, 6]              # Diagonals
]


def check_winner(board):
    """
    Check if there's a winner or draw.
    
    Returns:
        1 if X (human) wins
        -1 if O (AI) wins
        0 if draw
        None if game continues
    """
    for combo in WINNING_COMBOS:
        values = [board[i] for i in combo]
        if values == [1, 1, 1]:
            return 1  # Human wins
        if values == [-1, -1, -1]:
            return -1  # AI wins
    
    # Check for draw
    if 0 not in board:
        return 0  # Draw
    
    return None  # Game continues


def get_valid_moves(board):
    """Get list of valid move indices (empty cells)."""
    return [i for i in range(9) if board[i] == 0]


def minimax(board, depth, is_maximizing, alpha=-float('inf'), beta=float('inf')):
    """
    Minimax algorithm with Alpha-Beta pruning.
    This makes the AI UNBEATABLE when active.
    
    The AI (O/-1) is the minimizing player.
    The Human (X/1) is the maximizing player.
    """
    winner = check_winner(board)
    
    # Terminal states
    if winner == 1:  # Human wins
        return 10 - depth  # Prefer faster wins
    if winner == -1:  # AI wins
        return depth - 10  # Prefer faster wins
    if winner == 0:  # Draw
        return 0
    
    valid_moves = get_valid_moves(board)
    
    if is_maximizing:  # Human's turn (maximizing)
        max_eval = -float('inf')
        for move in valid_moves:
            board[move] = 1
            eval_score = minimax(board, depth + 1, False, alpha, beta)
            board[move] = 0
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:  # AI's turn (minimizing)
        min_eval = float('inf')
        for move in valid_moves:
            board[move] = -1
            eval_score = minimax(board, depth + 1, True, alpha, beta)
            board[move] = 0
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval


def get_god_mode_move(board):
    """
    Get the best move using Minimax algorithm.
    This is UNBEATABLE - will always win or draw.
    """
    valid_moves = get_valid_moves(board)
    
    if not valid_moves:
        return None
    
    best_move = None
    best_score = float('inf')
    
    for move in valid_moves:
        board[move] = -1  # AI plays O
        score = minimax(board, 0, True)  # Human plays next (maximizing)
        board[move] = 0
        
        if score < best_score:
            best_score = score
            best_move = move
    
    return best_move


def get_calibration_move(board, difficulty=0.5):
    """
    Get an AI move during calibration phase.
    Makes "mistakes" sometimes to let human win.
    
    Args:
        board: Current board state
        difficulty: 0.0 (easy) to 1.0 (hard)
    """
    valid_moves = get_valid_moves(board)
    
    if not valid_moves:
        return None
    
    # Priority 1: Win if possible (but sometimes miss it)
    if random.random() < difficulty:
        for combo in WINNING_COMBOS:
            values = [board[i] for i in combo]
            if values.count(-1) == 2 and values.count(0) == 1:
                winning_move = combo[values.index(0)]
                if winning_move in valid_moves:
                    return winning_move
    
    # Priority 2: Block human win (but sometimes miss it)
    if random.random() < difficulty + 0.2:
        for combo in WINNING_COMBOS:
            values = [board[i] for i in combo]
            if values.count(1) == 2 and values.count(0) == 1:
                blocking_move = combo[values.index(0)]
                if blocking_move in valid_moves:
                    return blocking_move
    
    # Priority 3: Take center if available
    if 4 in valid_moves and random.random() < 0.7:
        return 4
    
    # Priority 4: Take corners
    corners = [i for i in [0, 2, 6, 8] if i in valid_moves]
    if corners and random.random() < 0.6:
        return random.choice(corners)
    
    # Default: Random move
    return random.choice(valid_moves)


def analyze_player_pattern(move_history):
    """
    Analyze player's move patterns for psychological profiling.
    Returns insights about player behavior.
    """
    if len(move_history) < 3:
        return {"pattern": "unknown", "aggression": 0.5}
    
    moves = [m[1] for m in move_history]
    
    # Check if player prefers corners, center, or edges
    corners = sum(1 for m in moves if m in [0, 2, 6, 8])
    center = sum(1 for m in moves if m == 4)
    edges = sum(1 for m in moves if m in [1, 3, 5, 7])
    
    total = len(moves)
    
    pattern = "balanced"
    if corners / total > 0.5:
        pattern = "corner_lover"
    elif center / total > 0.3:
        pattern = "center_focused"
    elif edges / total > 0.5:
        pattern = "edge_player"
    
    # Aggression = how often they go for winning moves vs defensive
    aggression = random.uniform(0.4, 0.8)  # Simplified
    
    return {
        "pattern": pattern,
        "aggression": aggression,
        "corner_pref": corners / total,
        "center_pref": center / total
    }


# ================================================================================
# FLASK ROUTES
# ================================================================================

@app.route('/')
def index():
    """Serve the main game page."""
    global game_count
    return render_template('index.html')


@app.route('/game_state', methods=['GET'])
def get_game_state():
    """Get current game state including game count and mode."""
    global game_count
    
    is_god_mode = game_count >= CALIBRATION_GAMES
    
    return jsonify({
        'game_count': game_count,
        'is_god_mode': is_god_mode,
        'phase': 'GOD MODE' if is_god_mode else 'CALIBRATION',
        'calibration_remaining': max(0, CALIBRATION_GAMES - game_count),
        'neural_trained': neural_network.is_trained if neural_network else False,
        'training_samples': neural_network.total_samples if neural_network else 0
    })


@app.route('/move', methods=['POST'])
def make_move():
    """
    Handle a player's move and return AI's response.
    
    Expected JSON:
    {
        "board": [0,0,0,0,0,0,0,0,0],
        "human_move": 4,
        "move_history": [...]
    }
    """
    global game_count, move_history, neural_network
    
    data = request.json
    board = data.get('board', [0] * 9)
    human_move = data.get('human_move')
    current_history = data.get('move_history', [])
    
    # Record this move for learning
    board_before_move = board.copy()
    if human_move is not None:
        move_history.append((board_before_move, human_move))
    
    # Apply human's move
    if human_move is not None and board[human_move] == 0:
        board[human_move] = 1
    
    # Check if human won
    winner = check_winner(board)
    if winner is not None:
        return handle_game_over(winner, board)
    
    # Determine AI mode
    is_god_mode = game_count >= CALIBRATION_GAMES
    
    # Predict human's next move (for taunting)
    prediction = None
    confidence = 0
    if neural_network and neural_network.is_trained:
        pred_move, confidence, probs = neural_network.predict_human_move(board)
        prediction = {
            'cell': pred_move,
            'cell_name': CELL_NAMES.get(pred_move, 'Unknown'),
            'confidence': round(confidence, 1),
            'all_probs': probs
        }
    
    # Get AI's move
    if is_god_mode:
        ai_move = get_god_mode_move(board)
        ai_mode = "GOD_MODE"
    else:
        # Increase difficulty as games progress
        difficulty = 0.3 + (game_count * 0.1)
        ai_move = get_calibration_move(board, min(difficulty, 0.7))
        ai_mode = "CALIBRATION"
    
    # Apply AI's move
    if ai_move is not None:
        board[ai_move] = -1
    
    # Check if AI won
    winner = check_winner(board)
    
    response = {
        'board': board,
        'ai_move': ai_move,
        'ai_move_name': CELL_NAMES.get(ai_move, 'Unknown') if ai_move else None,
        'winner': winner,
        'game_over': winner is not None,
        'is_god_mode': is_god_mode,
        'ai_mode': ai_mode,
        'prediction': prediction,
        'game_count': game_count,
        'taunt': generate_taunt(prediction, is_god_mode, winner)
    }
    
    if winner is not None:
        # Game over - save moves and increment counter
        save_game_to_csv(move_history)
        game_count += 1
        move_history = []
        
        # Retrain neural network periodically
        if game_count % 3 == 0:
            train_neural_network()
    
    return jsonify(response)


@app.route('/new_game', methods=['POST'])
def new_game():
    """Start a new game."""
    global move_history
    move_history = []
    
    return jsonify({
        'success': True,
        'game_count': game_count,
        'is_god_mode': game_count >= CALIBRATION_GAMES
    })


@app.route('/reset_all', methods=['POST'])
def reset_all():
    """Reset all game data (for testing)."""
    global game_count, move_history
    game_count = 0
    move_history = []
    
    return jsonify({'success': True, 'message': 'All data reset!'})


def handle_game_over(winner, board):
    """Handle game over state."""
    global game_count, move_history
    
    save_game_to_csv(move_history)
    game_count += 1
    move_history = []
    
    winner_text = "Human" if winner == 1 else ("AI" if winner == -1 else "Nobody (Draw)")
    
    return jsonify({
        'board': board,
        'ai_move': None,
        'winner': winner,
        'game_over': True,
        'winner_text': winner_text,
        'game_count': game_count,
        'is_god_mode': game_count >= CALIBRATION_GAMES,
        'taunt': generate_game_over_taunt(winner)
    })


def generate_taunt(prediction, is_god_mode, winner):
    """Generate psychological taunts based on game state."""
    if winner is not None:
        return None
    
    if not is_god_mode:
        taunts = [
            "I'm still learning your patterns... 🎯",
            "Interesting move choice... 🤔",
            "Calibrating psychological profile... ⚙️",
            "You think you're in control? 😏",
            f"Game {game_count + 1} of {CALIBRATION_GAMES}... enjoy the peace while it lasts."
        ]
        return random.choice(taunts)
    
    if prediction and prediction['confidence'] > 50:
        aggressive_taunts = [
            f"🔮 I KNEW you'd go for {prediction['cell_name']}... ({prediction['confidence']}% confidence)",
            f"👁️ Your next move will be {prediction['cell_name']}. I'm {prediction['confidence']}% certain.",
            f"🧠 Predictable. You're thinking about {prediction['cell_name']}, aren't you?",
            f"⚡ I've already calculated you'll choose {prediction['cell_name']}. Prove me wrong.",
            f"🎭 Your brain is TRANSPARENT to me. {prediction['cell_name']} is your choice."
        ]
        return random.choice(aggressive_taunts)
    
    return "🔥 GOD MODE ACTIVE - I cannot be defeated."


def generate_game_over_taunt(winner):
    """Generate end-game psychological messages."""
    if winner == 1:  # Human wins
        if game_count < CALIBRATION_GAMES:
            return "😊 Enjoy this victory... it won't last. The calibration continues."
        else:
            return "🤯 Impossible! This should not have happened... (checking for bugs)"
    elif winner == -1:  # AI wins
        if game_count >= CALIBRATION_GAMES:
            return "🔥 DID YOU REALLY THINK YOU COULD BEAT ME? I am INEVITABLE."
        else:
            return "💀 Even while holding back, I crushed you."
    else:  # Draw
        if game_count >= CALIBRATION_GAMES:
            return "⚖️ A draw against a GOD. Consider that a victory, mortal."
        else:
            return "🤝 We're evenly matched... for now."


def train_neural_network():
    """Train or retrain the neural network with all available data."""
    global neural_network
    
    if not TENSORFLOW_AVAILABLE:
        return
    
    df = load_all_training_data()
    if df is not None and len(df) >= 10:
        neural_network.train(df)


# ================================================================================
# INITIALIZATION
# ================================================================================

def initialize_app():
    """Initialize the application on startup."""
    global neural_network, game_count
    
    print("=" * 60)
    print("🎮 PSYCHOLOGICAL TIC-TAC-TOE SERVER")
    print("=" * 60)
    
    # Initialize CSV storage
    initialize_csv()
    
    # Initialize Neural Network
    neural_network = PsychologicalNeuralNetwork()
    
    if TENSORFLOW_AVAILABLE:
        neural_network.build_model()
        train_neural_network()
    
    # Load game count from CSV (count number of complete games)
    if os.path.exists(GAME_DATA_FILE):
        try:
            df = pd.read_csv(GAME_DATA_FILE)
            # Estimate games played (roughly 3-5 moves per game)
            game_count = max(0, len(df) // 4)
            print(f"📊 Estimated {game_count} previous games from data")
        except:
            game_count = 0
    
    print(f"🎯 Calibration Phase: {CALIBRATION_GAMES} games")
    print(f"🔥 God Mode activates after: Game {CALIBRATION_GAMES}")
    print("=" * 60)


# Initialize on import
initialize_app()


if __name__ == '__main__':
    print("\n🌐 Starting Flask server...")
    print("📍 Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)