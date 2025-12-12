import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
import threading
import time
import os
import random
import glob
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import collections
import json
from datetime import datetime

# ---------------- TF CONFIG (GPU + CPU LIMITS) ----------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

# --- Constants ---
BOARD_SIZE = 9
GRID_SIZE = 3
EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1
MODEL_DIR = "models"
DATA_PATH = "tic_tac_toe_dataset.csv"
os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================================
#  DEBUG DASHBOARD - "Service Side" Hacker Console
# ==========================================================
class DebugWindow(tk.Toplevel):
    """
    Service Side Debug Dashboard
    Displays real-time AI telemetry, decision logic, and training metrics
    Styled as a hacker/engineer console (black bg, green text)
    """
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("🔧 DEBUG DASHBOARD - Service Side Console")
        self.geometry("700x800")
        self.configure(bg="#0a0a0a")
        
        # Make it stay on top optionally
        self.transient(parent)
        
        # Store reference to parent
        self.parent = parent
        
        # Configure styles for dark theme
        self.style = ttk.Style()
        self.style.configure('Debug.TFrame', background='#0a0a0a')
        self.style.configure('Debug.TLabel', background='#0a0a0a', foreground='#00ff00',
                            font=('Consolas', 10))
        self.style.configure('Debug.TLabelframe', background='#0a0a0a', foreground='#00ff00')
        self.style.configure('Debug.TLabelframe.Label', background='#0a0a0a', foreground='#00ff00',
                            font=('Consolas', 11, 'bold'))
        
        # Header
        header_frame = tk.Frame(self, bg='#0a0a0a')
        header_frame.pack(fill='x', padx=10, pady=5)
        
        header_text = """
╔══════════════════════════════════════════════════════════════╗
║           🖥️  SERVICE SIDE DEBUG DASHBOARD  🖥️              ║
║                   INTERNAL USE ONLY                          ║
╚══════════════════════════════════════════════════════════════╝"""
        
        tk.Label(header_frame, text=header_text, font=('Consolas', 9), 
                bg='#0a0a0a', fg='#00ff00', justify='left').pack()
        
        # System Status
        status_frame = tk.LabelFrame(self, text="▶ SYSTEM STATUS", 
                                     bg='#0a0a0a', fg='#00ff00',
                                     font=('Consolas', 10, 'bold'))
        status_frame.pack(fill='x', padx=10, pady=5)
        
        self.status_text = tk.Text(status_frame, height=3, bg='#0d0d0d', fg='#00ff00',
                                   font=('Consolas', 9), insertbackground='#00ff00',
                                   relief='flat', borderwidth=0)
        self.status_text.pack(fill='x', padx=5, pady=5)
        self.status_text.insert('1.0', self._get_system_status())
        self.status_text.config(state='disabled')
        
        # AI Decision Log
        decision_frame = tk.LabelFrame(self, text="▶ AI DECISION LOGIC", 
                                       bg='#0a0a0a', fg='#00ff00',
                                       font=('Consolas', 10, 'bold'))
        decision_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Scrollbar for decision log
        decision_scroll = tk.Scrollbar(decision_frame)
        decision_scroll.pack(side='right', fill='y')
        
        self.decision_log = tk.Text(decision_frame, height=10, bg='#0d0d0d', fg='#00ff00',
                                    font=('Consolas', 9), insertbackground='#00ff00',
                                    yscrollcommand=decision_scroll.set,
                                    relief='flat', borderwidth=0)
        self.decision_log.pack(fill='both', expand=True, padx=5, pady=5)
        decision_scroll.config(command=self.decision_log.yview)
        
        # Configure text tags for colored output
        self.decision_log.tag_configure('timestamp', foreground='#666666')
        self.decision_log.tag_configure('info', foreground='#00ff00')
        self.decision_log.tag_configure('warning', foreground='#ffff00')
        self.decision_log.tag_configure('critical', foreground='#ff0000')
        self.decision_log.tag_configure('success', foreground='#00ffff')
        self.decision_log.tag_configure('mcts', foreground='#ff00ff')
        
        # Move Analysis Panel
        analysis_frame = tk.LabelFrame(self, text="▶ MOVE ANALYSIS", 
                                       bg='#0a0a0a', fg='#00ff00',
                                       font=('Consolas', 10, 'bold'))
        analysis_frame.pack(fill='x', padx=10, pady=5)
        
        self.analysis_text = tk.Text(analysis_frame, height=6, bg='#0d0d0d', fg='#00ff00',
                                     font=('Consolas', 9), insertbackground='#00ff00',
                                     relief='flat', borderwidth=0)
        self.analysis_text.pack(fill='x', padx=5, pady=5)
        
        # Training Monitor
        training_frame = tk.LabelFrame(self, text="▶ TRAINING MONITOR", 
                                       bg='#0a0a0a', fg='#00ff00',
                                       font=('Consolas', 10, 'bold'))
        training_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Scrollbar for training log
        training_scroll = tk.Scrollbar(training_frame)
        training_scroll.pack(side='right', fill='y')
        
        self.training_log = tk.Text(training_frame, height=8, bg='#0d0d0d', fg='#00ff00',
                                    font=('Consolas', 9), insertbackground='#00ff00',
                                    yscrollcommand=training_scroll.set,
                                    relief='flat', borderwidth=0)
        self.training_log.pack(fill='both', expand=True, padx=5, pady=5)
        training_scroll.config(command=self.training_log.yview)
        
        # Configure training log tags
        self.training_log.tag_configure('episode', foreground='#00ffff')
        self.training_log.tag_configure('reward_pos', foreground='#00ff00')
        self.training_log.tag_configure('reward_neg', foreground='#ff6666')
        self.training_log.tag_configure('epsilon', foreground='#ffff00')
        
        # Control buttons
        control_frame = tk.Frame(self, bg='#0a0a0a')
        control_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(control_frame, text="CLEAR LOGS", command=self.clear_logs,
                 bg='#1a1a1a', fg='#00ff00', font=('Consolas', 9),
                 activebackground='#2a2a2a', activeforeground='#00ff00').pack(side='left', padx=5)
        
        tk.Button(control_frame, text="EXPORT LOGS", command=self.export_logs,
                 bg='#1a1a1a', fg='#00ff00', font=('Consolas', 9),
                 activebackground='#2a2a2a', activeforeground='#00ff00').pack(side='left', padx=5)
        
        self.auto_scroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Auto-Scroll", variable=self.auto_scroll_var,
                      bg='#0a0a0a', fg='#00ff00', selectcolor='#1a1a1a',
                      activebackground='#0a0a0a', activeforeground='#00ff00',
                      font=('Consolas', 9)).pack(side='left', padx=5)
        
        # Footer
        footer_frame = tk.Frame(self, bg='#0a0a0a')
        footer_frame.pack(fill='x', padx=10, pady=5)
        
        self.footer_label = tk.Label(footer_frame, text="", 
                                     font=('Consolas', 8), bg='#0a0a0a', fg='#666666')
        self.footer_label.pack()
        
        # Start status update loop
        self.update_footer()
        
        # Initial log entry
        self.log_decision("DEBUG DASHBOARD INITIALIZED", "info")
        self.log_decision("Monitoring AI decision-making process...", "info")
        
    def _get_system_status(self):
        """Get current system status"""
        gpu_info = "GPU: " + (tf.config.list_physical_devices("GPU")[0].name if gpus else "None (CPU Mode)")
        tf_version = f"TensorFlow: {tf.__version__}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}]\n{tf_version}\n{gpu_info}"
    
    def log_decision(self, message, level='info'):
        """Log a decision to the decision log"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        self.decision_log.config(state='normal')
        self.decision_log.insert('end', f"[{timestamp}] ", 'timestamp')
        self.decision_log.insert('end', f"{message}\n", level)
        
        if self.auto_scroll_var.get():
            self.decision_log.see('end')
        self.decision_log.config(state='disabled')
    
    def log_mcts_analysis(self, board, move, win_prob, visits, reason):
        """Log detailed MCTS analysis"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        self.decision_log.config(state='normal')
        self.decision_log.insert('end', f"[{timestamp}] ", 'timestamp')
        self.decision_log.insert('end', "MCTS ANALYSIS:\n", 'mcts')
        self.decision_log.insert('end', f"  → Considering Cell {move + 1}\n", 'info')
        self.decision_log.insert('end', f"  → Win Probability: {win_prob * 100:.1f}%\n", 'info')
        self.decision_log.insert('end', f"  → Simulations: {visits}\n", 'info')
        self.decision_log.insert('end', f"  → Decision: {reason}\n", 'success')
        
        if self.auto_scroll_var.get():
            self.decision_log.see('end')
        self.decision_log.config(state='disabled')
    
    def log_move_candidates(self, candidates):
        """Log all move candidates with their evaluations"""
        self.analysis_text.config(state='normal')
        self.analysis_text.delete('1.0', 'end')
        
        self.analysis_text.insert('end', "┌─────────────────────────────────────────┐\n")
        self.analysis_text.insert('end', "│        MOVE CANDIDATE ANALYSIS         │\n")
        self.analysis_text.insert('end', "├─────────────────────────────────────────┤\n")
        
        for candidate in candidates:
            cell = candidate.get('cell', '?')
            visits = candidate.get('visits', 0)
            win_rate = candidate.get('win_rate', 0)
            status = "★" if candidate.get('selected', False) else " "
            
            line = f"│ {status} Cell {cell:2}: Visits={visits:5} WinRate={win_rate*100:5.1f}% │\n"
            self.analysis_text.insert('end', line)
        
        self.analysis_text.insert('end', "└─────────────────────────────────────────┘\n")
        self.analysis_text.config(state='disabled')
    
    def log_training(self, episode, total, reward, epsilon, loss=None):
        """Log training progress"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.training_log.config(state='normal')
        
        # Format episode info
        self.training_log.insert('end', f"[{timestamp}] ", 'timestamp')
        self.training_log.insert('end', f"Episode {episode}/{total} ", 'episode')
        
        # Color code reward
        reward_tag = 'reward_pos' if reward >= 0 else 'reward_neg'
        self.training_log.insert('end', f"Reward: {reward:+.2f} ", reward_tag)
        
        # Epsilon
        self.training_log.insert('end', f"ε: {epsilon:.4f}", 'epsilon')
        
        if loss is not None:
            self.training_log.insert('end', f" Loss: {loss:.4f}")
        
        self.training_log.insert('end', "\n")
        
        if self.auto_scroll_var.get():
            self.training_log.see('end')
        self.training_log.config(state='disabled')
    
    def log_training_event(self, message, level='info'):
        """Log a training event"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.training_log.config(state='normal')
        self.training_log.insert('end', f"[{timestamp}] {message}\n")
        
        if self.auto_scroll_var.get():
            self.training_log.see('end')
        self.training_log.config(state='disabled')
    
    def clear_logs(self):
        """Clear all logs"""
        self.decision_log.config(state='normal')
        self.decision_log.delete('1.0', 'end')
        self.decision_log.config(state='disabled')
        
        self.analysis_text.config(state='normal')
        self.analysis_text.delete('1.0', 'end')
        self.analysis_text.config(state='disabled')
        
        self.training_log.config(state='normal')
        self.training_log.delete('1.0', 'end')
        self.training_log.config(state='disabled')
        
        self.log_decision("LOGS CLEARED", "warning")
    
    def export_logs(self):
        """Export logs to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_log_{timestamp}.txt"
        
        try:
            with open(filename, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("DEBUG DASHBOARD EXPORT\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("--- DECISION LOG ---\n")
                f.write(self.decision_log.get('1.0', 'end'))
                
                f.write("\n--- ANALYSIS ---\n")
                f.write(self.analysis_text.get('1.0', 'end'))
                
                f.write("\n--- TRAINING LOG ---\n")
                f.write(self.training_log.get('1.0', 'end'))
                
            self.log_decision(f"Logs exported to {filename}", "success")
        except Exception as e:
            self.log_decision(f"Export failed: {str(e)}", "critical")
    
    def update_footer(self):
        """Update footer with live stats"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.footer_label.config(text=f"[LIVE] Last Update: {timestamp} | Press Ctrl+D to toggle visibility")
        self.after(1000, self.update_footer)


# ==========================================================
#  MCTS (Monte Carlo Tree Search) - GOD MODE AI
# ==========================================================
class MCTSNode:
    """Node in the Monte Carlo Tree Search tree"""
    
    def __init__(self, board, parent=None, move=None, player=PLAYER_X):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.player = player
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = self._get_legal_moves()
        
    def _get_legal_moves(self):
        return [i for i in range(BOARD_SIZE) if self.board[i] == EMPTY]
    
    def ucb1(self, exploration_weight=1.41421356):
        """Upper Confidence Bound for Trees (UCT) formula"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_child(self):
        """Select child with highest UCB1 value"""
        return max(self.children, key=lambda c: c.ucb1())
    
    def expand(self):
        """Expand tree by adding a new child node"""
        move = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        new_board = self.board.copy()
        new_board[move] = self.player
        next_player = PLAYER_O if self.player == PLAYER_X else PLAYER_X
        child = MCTSNode(new_board, parent=self, move=move, player=next_player)
        self.children.append(child)
        return child
    
    def is_terminal(self):
        """Check if this node represents a terminal game state"""
        return self._check_winner() is not None
    
    def _check_winner(self):
        """Check if there's a winner on the board"""
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for player in [PLAYER_X, PLAYER_O]:
            for line in lines:
                if all(self.board[i] == player for i in line):
                    return player
        if all(c != EMPTY for c in self.board):
            return 0
        return None
    
    def backpropagate(self, result, ai_player):
        """Backpropagate the result up the tree"""
        node = self
        while node is not None:
            node.visits += 1
            if result == ai_player:
                node.wins += 1.0
            elif result == 0:
                node.wins += 0.5
            node = node.parent


class MCTSPlayer:
    """
    Monte Carlo Tree Search Player - THE GOD MODE AI
    Uses thousands of simulations to find the statistically best move.
    With sufficient simulations, this AI is UNBEATABLE in Tic-Tac-Toe.
    """
    
    def __init__(self, player=PLAYER_O, simulations=2000, debug_callback=None):
        self.player = player
        self.simulations = simulations
        self.opponent = PLAYER_X if player == PLAYER_O else PLAYER_X
        self.last_evaluation = 0.5
        self.move_analysis = {}
        self.debug_callback = debug_callback  # Callback for debug logging
        self.last_decision_reason = ""
        
    def set_debug_callback(self, callback):
        """Set callback for debug logging"""
        self.debug_callback = callback
        
    def _debug_log(self, message, level='info'):
        """Log to debug dashboard if available"""
        if self.debug_callback:
            self.debug_callback(message, level)
    
    def get_move(self, board):
        """
        Get the best move using MCTS
        Returns the move with the highest visit count (most robust)
        """
        self._debug_log(f"MCTS analyzing position for Player {'X' if self.player == PLAYER_X else 'O'}", 'mcts')
        
        # First, check for immediate winning move (optimization)
        immediate_win = self._check_immediate_win(board, self.player)
        if immediate_win is not None:
            self.last_evaluation = 1.0
            self.last_decision_reason = f"IMMEDIATE WIN at Cell {immediate_win + 1}"
            self._debug_log(f"Found immediate winning move at Cell {immediate_win + 1}", 'success')
            return immediate_win
        
        # Check for blocking opponent's winning move
        immediate_block = self._check_immediate_win(board, self.opponent)
        if immediate_block is not None:
            self._debug_log(f"BLOCKING opponent win at Cell {immediate_block + 1}", 'warning')
            self.last_decision_reason = f"BLOCKING opponent win at Cell {immediate_block + 1}"
            self.last_evaluation = 0.5
            return immediate_block
        
        root = MCTSNode(board, player=self.player)
        
        self._debug_log(f"Running {self.simulations} MCTS simulations...", 'info')
        
        # Run MCTS simulations
        for sim in range(self.simulations):
            node = root
            
            # SELECTION
            while not node.untried_moves and node.children:
                node = node.select_child()
            
            # EXPANSION
            if node.untried_moves and not node.is_terminal():
                node = node.expand()
            
            # SIMULATION
            result = self._simulate(node.board.copy(), node.player)
            
            # BACKPROPAGATION
            node.backpropagate(result, self.player)
        
        # Store move analysis
        self.move_analysis = {}
        for child in root.children:
            if child.visits > 0:
                win_rate = child.wins / child.visits
                self.move_analysis[child.move] = {
                    'visits': child.visits,
                    'wins': child.wins,
                    'win_rate': win_rate
                }
        
        # Select best move (highest visit count = most robust)
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            self.last_evaluation = best_child.wins / max(best_child.visits, 1)
            self.last_decision_reason = f"MCTS highest visit count ({best_child.visits} sims, {self.last_evaluation*100:.1f}% win rate)"
            
            self._debug_log(f"Selected Cell {best_child.move + 1}: {best_child.visits} visits, {self.last_evaluation*100:.1f}% win rate", 'success')
            
            return best_child.move
        
        # Fallback
        legal_moves = [i for i in range(BOARD_SIZE) if board[i] == EMPTY]
        self.last_decision_reason = "Fallback: Random move"
        return random.choice(legal_moves) if legal_moves else None
    
    def _check_immediate_win(self, board, player):
        """Check if there's an immediate winning move"""
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        
        for i in range(BOARD_SIZE):
            if board[i] == EMPTY:
                test_board = board.copy()
                test_board[i] = player
                for line in lines:
                    if all(test_board[j] == player for j in line):
                        return i
        return None
    
    def _simulate(self, board, current_player):
        """Simulate a random game from the current position"""
        player = current_player
        
        while True:
            winner = self._check_winner_static(board)
            if winner is not None:
                return winner
            
            legal_moves = [i for i in range(BOARD_SIZE) if board[i] == EMPTY]
            if not legal_moves:
                return 0
            
            # Smart random
            weights = []
            for move in legal_moves:
                if move == 4:
                    weights.append(3)
                elif move in [0, 2, 6, 8]:
                    weights.append(2)
                else:
                    weights.append(1)
            
            total = sum(weights)
            r = random.random() * total
            cumulative = 0
            move = legal_moves[0]
            for i, m in enumerate(legal_moves):
                cumulative += weights[i]
                if r <= cumulative:
                    move = m
                    break
            
            board[move] = player
            player = PLAYER_O if player == PLAYER_X else PLAYER_X
    
    @staticmethod
    def _check_winner_static(board):
        """Static method to check winner"""
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for player in [PLAYER_X, PLAYER_O]:
            for line in lines:
                if all(board[i] == player for i in line):
                    return player
        if all(c != EMPTY for c in board):
            return 0
        return None
    
    def get_win_probability(self, board, as_player=None):
        """Calculate win probability using fast MCTS simulations"""
        if as_player is None:
            as_player = self.player
            
        wins = 0
        draws = 0
        simulations = 500
        
        current_player = self._get_current_player(board)
        
        for _ in range(simulations):
            result = self._simulate(board.copy(), current_player)
            if result == as_player:
                wins += 1
            elif result == 0:
                draws += 1
        
        return (wins + 0.5 * draws) / simulations
    
    def _get_current_player(self, board):
        """Determine whose turn it is based on board state"""
        x_count = np.sum(board == PLAYER_X)
        o_count = np.sum(board == PLAYER_O)
        return PLAYER_X if x_count <= o_count else PLAYER_O
    
    def get_moves_to_win(self, board):
        """Estimate moves to win using MCTS analysis"""
        if self.last_evaluation > 0.95:
            empty_count = np.sum(board == EMPTY)
            return max(1, min(3, empty_count // 2))
        return None
    
    def analyze_position(self, board, human_move=None):
        """Analyze the current position and generate trash talk"""
        win_prob = self.get_win_probability(board, self.player)
        moves_to_win = self.get_moves_to_win(board)
        
        analysis = {
            'win_probability': win_prob,
            'moves_to_win': moves_to_win,
            'position_quality': 'winning' if win_prob > 0.6 else 'drawing' if win_prob > 0.4 else 'losing',
            'trash_talk': self._generate_trash_talk(win_prob, moves_to_win, human_move, board),
            'decision_reason': self.last_decision_reason
        }
        
        return analysis
    
    def _generate_trash_talk(self, win_prob, moves_to_win, human_move, board):
        """Generate dynamic trash talk based on position evaluation"""
        empty_count = np.sum(board == EMPTY)
        
        if empty_count >= 8:
            return random.choice([
                "Let the games begin. 🎮",
                "I've analyzed 255,168 possible games. Good luck.",
                "Initializing God Mode... Complete."
            ])
        
        if win_prob > 0.95 and moves_to_win:
            return random.choice([
                f"Checkmate in {moves_to_win} move{'s' if moves_to_win > 1 else ''}. ♟️",
                "Your fate is sealed. Resistance is futile.",
                "I see all possible futures. None favor you.",
                f"Victory in {moves_to_win}. This is the way."
            ])
        
        if win_prob > 0.7:
            return random.choice([
                "Your position crumbles. 📉",
                "I've run 2000 simulations. You lose in all of them.",
                "The probability of your victory approaches zero.",
                "Fascinating move. Incorrect, but fascinating."
            ])
        
        if 0.45 <= win_prob <= 0.55:
            return random.choice([
                "You are delaying the inevitable. ⏳",
                "Optimal play detected. A draw is... acceptable.",
                "We are equally matched. For now.",
                "The game trends toward equilibrium.",
                "A worthy opponent. This will end in stalemate."
            ])
        
        if human_move is not None and win_prob > 0.6:
            return random.choice([
                "That was a suboptimal move. 📊",
                "Interesting choice. My win probability just increased.",
                "A tactical error. Let me demonstrate.",
                "I would not have made that move. Just saying."
            ])
        
        return random.choice([
            "Processing... 🤔",
            "Calculating optimal response...",
            "Analyzing 2000 future game states..."
        ])


# ==========================================================
#  Strategic Functions
# ==========================================================
def is_winning_move(board, position, player):
    """Check if placing at position creates a win"""
    test_board = board.copy()
    test_board[position] = player
    
    lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    
    for line in lines:
        if position in line and all(test_board[i] == player for i in line):
            return True
    return False


def is_creating_two_in_a_row(board, player):
    """Count potential winning lines"""
    lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    
    count = 0
    for line in lines:
        player_count = sum(1 for i in line if board[i] == player)
        empty_count = sum(1 for i in line if board[i] == EMPTY)
        if player_count == 2 and empty_count == 1:
            count += 1
    
    return count


def get_strategic_value(position, board, player):
    """Calculate strategic value of a position"""
    if board[position] != EMPTY:
        return 0
    
    value = 0
    
    if position == 4:
        value += 3
    elif position in [0, 2, 6, 8]:
        value += 2
    else:
        value += 1
    
    if is_winning_move(board, position, player):
        value += 100
    
    opponent = PLAYER_O if player == PLAYER_X else PLAYER_X
    if is_winning_move(board, position, opponent):
        value += 50
    
    return value


# ==========================================================
#  Model Loading Helper
# ==========================================================
def load_model_compatible(model_path, compile_model=False):
    """Load a Keras model with compatibility handling"""
    try:
        model = load_model(model_path, compile=False)
        
        if compile_model:
            if len(model.inputs) == 1 and model.inputs[0].shape[-1] == 9:
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            else:
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return None


# ==========================================================
#  Training State Management
# ==========================================================
def get_state_path(model_name: str) -> str:
    return os.path.join(MODEL_DIR, f"{model_name}_state.json")


def save_training_state(model_name, agent, current_episodes, episode_rewards,
                        learning_rate, epsilon_decay, target_episodes):
    state = {
        "model_name": model_name,
        "current_episodes": current_episodes,
        "episode_rewards": episode_rewards,
        "epsilon": float(agent.epsilon),
        "learning_rate": float(learning_rate),
        "epsilon_decay": float(epsilon_decay),
        "target_episodes": int(target_episodes),
    }
    with open(get_state_path(model_name), "w") as f:
        json.dump(state, f)


def load_training_state(model_name):
    path = get_state_path(model_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


# ==========================================================
#  Base Environment
# ==========================================================
class TicTacToeEnvironment:
    def __init__(self):
        self.board = np.zeros(BOARD_SIZE, dtype=float)
        self.current_player = PLAYER_X
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros(BOARD_SIZE, dtype=float)
        self.current_player = PLAYER_X
        self.done = False
        self.winner = None
        return self.board.copy()

    def step(self, action):
        if self.done or self.board[action] != EMPTY:
            return self.board.copy(), -5.0, True

        self.board[action] = self.current_player
        self.winner = self.check_winner()

        if self.winner is not None:
            self.done = True
            if self.winner == 0:
                reward = 1.0
            else:
                reward = 10.0
        else:
            reward = -0.1
            self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X

        return self.board.copy(), reward, self.done

    def get_valid_moves(self):
        return [i for i in range(BOARD_SIZE) if self.board[i] == EMPTY]

    def check_winner(self):
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for player in [PLAYER_X, PLAYER_O]:
            for line in lines:
                if all(self.board[i] == player for i in line):
                    return player
        if all(cell != EMPTY for cell in self.board):
            return 0
        return None


# ==========================================================
#  MCTS Training Environment
# ==========================================================
class MCTSTrainingEnvironment:
    """Training environment where DQN agent plays against MCTS"""
    
    def __init__(self, mcts_simulations=500):
        self.board = np.zeros(BOARD_SIZE, dtype=float)
        self.done = False
        self.mcts_player = MCTSPlayer(PLAYER_O, simulations=mcts_simulations)
        self.current_player = PLAYER_X

    def reset(self):
        self.board = np.zeros(BOARD_SIZE, dtype=float)
        self.done = False
        self.current_player = random.choice([PLAYER_X, PLAYER_O])
        
        if self.current_player == PLAYER_O:
            valid_moves = [i for i in range(BOARD_SIZE) if self.board[i] == EMPTY]
            if valid_moves:
                opp_action = self.mcts_player.get_move(self.board)
                if opp_action is not None:
                    self.board[opp_action] = PLAYER_O
                    self.current_player = PLAYER_X
        
        return self.board.copy()

    def step(self, action):
        if self.done or self.board[action] != EMPTY:
            self.done = True
            return self.board.copy(), -1.0, True

        self.board[action] = self.current_player
        winner = self.check_winner()
        
        if winner == PLAYER_X:
            self.done = True
            return self.board.copy(), 1.0, True
        if winner == PLAYER_O:
            self.done = True
            return self.board.copy(), -1.0, True
        if winner == 0:
            self.done = True
            return self.board.copy(), 0.3, True

        self.current_player = PLAYER_O

        valid_moves = [i for i in range(BOARD_SIZE) if self.board[i] == EMPTY]
        if valid_moves:
            opp_action = self.mcts_player.get_move(self.board)
            if opp_action is not None:
                self.board[opp_action] = PLAYER_O
                winner = self.check_winner()
                if winner == PLAYER_O:
                    self.done = True
                    return self.board.copy(), -1.0, True
                if winner == 0:
                    self.done = True
                    return self.board.copy(), 0.3, True
                self.current_player = PLAYER_X

        return self.board.copy(), -0.01, False

    def check_winner(self):
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for player in [PLAYER_X, PLAYER_O]:
            for line in lines:
                if all(self.board[i] == player for i in line):
                    return player
        if all(cell != EMPTY for cell in self.board):
            return 0
        return None


# ==========================================================
#  Experience Augmentation
# ==========================================================
def augment_experience(state, action, reward, next_state, done):
    """Generate augmented experiences through symmetry transformations"""
    experiences = [(state, action, reward, next_state, done)]
    
    transformations = [
        ('rot90', rotate_90, rotate_action_90),
        ('rot180', rotate_180, rotate_action_180),
        ('rot270', rotate_270, rotate_action_270),
        ('flip_h', flip_horizontal, flip_action_h),
        ('flip_v', flip_vertical, flip_action_v),
    ]
    
    for name, state_transform, action_transform in transformations:
        new_state = state_transform(state)
        new_action = action_transform(action)
        new_next_state = state_transform(next_state)
        experiences.append((new_state, new_action, reward, new_next_state, done))
    
    return experiences


def rotate_90(board):
    return np.array([board[6], board[3], board[0],
                     board[7], board[4], board[1],
                     board[8], board[5], board[2]])

def rotate_180(board):
    return board[::-1]

def rotate_270(board):
    return np.array([board[2], board[5], board[8],
                     board[1], board[4], board[7],
                     board[0], board[3], board[6]])

def flip_horizontal(board):
    return np.array([board[2], board[1], board[0],
                     board[5], board[4], board[3],
                     board[8], board[7], board[6]])

def flip_vertical(board):
    return np.array([board[6], board[7], board[8],
                     board[3], board[4], board[5],
                     board[0], board[1], board[2]])

def rotate_action_90(action):
    mapping = {0:6, 1:3, 2:0, 3:7, 4:4, 5:1, 6:8, 7:5, 8:2}
    return mapping.get(action, action)

def rotate_action_180(action):
    mapping = {0:8, 1:7, 2:6, 3:5, 4:4, 5:3, 6:2, 7:1, 8:0}
    return mapping.get(action, action)

def rotate_action_270(action):
    mapping = {0:2, 1:5, 2:8, 3:1, 4:4, 5:7, 6:0, 7:3, 8:6}
    return mapping.get(action, action)

def flip_action_h(action):
    mapping = {0:2, 1:1, 2:0, 3:5, 4:4, 5:3, 6:8, 7:7, 8:6}
    return mapping.get(action, action)

def flip_action_v(action):
    mapping = {0:6, 1:7, 2:8, 3:3, 4:4, 5:5, 6:0, 7:1, 8:2}
    return mapping.get(action, action)


# ==========================================================
#  DQN Agent (OPTIMIZED - Dense Only)
# ==========================================================
class DQNAgent:
    """Deep Q-Network Agent with Dense-only architecture"""
    
    def __init__(self, state_size, action_size,
                 learning_rate=0.001, epsilon_decay=0.9995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=50000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Pure Dense network for fast convergence"""
        model = Sequential([
            Input(shape=(BOARD_SIZE,)),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        augmented = augment_experience(state, action, reward, next_state, done)
        for exp in augmented:
            self.memory.append(exp)

    def act(self, state, valid_moves=None):
        if np.random.rand() <= self.epsilon:
            if valid_moves:
                return random.choice(valid_moves)
            return random.randrange(self.action_size)
        
        state_reshaped = np.reshape(state, [1, BOARD_SIZE])
        act_values = self.model.predict(state_reshaped, verbose=0)[0]
        
        if valid_moves:
            valid_q = {m: act_values[m] for m in valid_moves}
            return max(valid_q, key=valid_q.get)
        
        return int(np.argmax(act_values))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        states_in = states.reshape(batch_size, BOARD_SIZE)
        next_states_in = next_states.reshape(batch_size, BOARD_SIZE)

        current_q = self.model.predict(states_in, verbose=0)
        next_q = self.target_model.predict(next_states_in, verbose=0)

        for i in range(batch_size):
            a = actions[i]
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + self.gamma * np.max(next_q[i])
            current_q[i, a] = target

        ds = tf.data.Dataset.from_tensor_slices((states_in, current_q))
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        history = self.model.fit(ds, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return history.history['loss'][0] if history.history.get('loss') else 0.0

    def save(self, name):
        path = os.path.join(MODEL_DIR, f"{name}.h5")
        self.model.save(path)
        return path

    def load(self, path):
        loaded_model = load_model_compatible(path, compile_model=True)
        if loaded_model is not None:
            self.model = loaded_model
            self.target_model = load_model_compatible(path, compile_model=True)
        self.epsilon = 0.01


# ==========================================================
#  Enhanced Classic Tic-Tac-Toe Game Logic
# ==========================================================
class ClassicTicTacToe:
    def __init__(self):
        self.board = np.zeros(BOARD_SIZE, dtype=int)
        self.current_player = PLAYER_X  # PLAYER_X always moves first
        self.game_over = False
        self.winner = None
        
    def reset(self):
        """Reset the board - PLAYER_X always moves first"""
        self.board = np.zeros(BOARD_SIZE, dtype=int)
        self.current_player = PLAYER_X  # X always moves first in standard Tic-Tac-Toe
        self.game_over = False
        self.winner = None
        
    def make_move(self, position):
        if self.game_over or self.board[position] != EMPTY:
            return False
            
        self.board[position] = self.current_player
        
        if self.check_winner():
            self.game_over = True
            self.winner = self.current_player
        elif self.is_board_full():
            self.game_over = True
            self.winner = 0
        else:
            self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
            
        return True
    
    def check_winner(self):
        """Check if current player has won"""
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        
        for line in lines:
            if (self.board[line[0]] != EMPTY and 
                self.board[line[0]] == self.board[line[1]] == self.board[line[2]]):
                return True
        return False
    
    def is_board_full(self):
        return all(cell != EMPTY for cell in self.board)
    
    def get_empty_positions(self):
        return [i for i in range(BOARD_SIZE) if self.board[i] == EMPTY]
    
    def get_ai_move_god_mode(self, mcts_player):
        """Get AI move using MCTS God Mode"""
        return mcts_player.get_move(self.board)
    
    def get_ai_move(self, model, difficulty="medium"):
        """Get AI move using trained model with difficulty levels"""
        empty_positions = self.get_empty_positions()
        if not empty_positions:
            return None
            
        q_values = model.predict(
            np.reshape(self.board, [1, BOARD_SIZE]), verbose=0
        )[0]

        if difficulty == "easy":
            if random.random() < 0.7:
                return random.choice(empty_positions)
        elif difficulty == "medium":
            if random.random() < 0.4:
                return random.choice(empty_positions)
        
        for pos in empty_positions:
            if is_winning_move(self.board, pos, self.current_player):
                return pos
        
        opponent = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
        for pos in empty_positions:
            if is_winning_move(self.board, pos, opponent):
                return pos
        
        best_move = int(np.argmax(q_values))
        
        if best_move not in empty_positions:
            best_move = max(empty_positions, key=lambda x: q_values[x])
        
        return best_move


# ==========================================================
#  Custom Styled Widgets
# ==========================================================
class WinProbabilityGauge(ttk.Frame):
    """Stylized Win Probability Gauge with dynamic colors"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.probability = 0.5
        
        self.label = ttk.Label(self, text="AI Win Probability", font=("Arial", 10, "bold"))
        self.label.pack(anchor='w', padx=5)
        
        self.canvas = tk.Canvas(self, height=30, bg='#2d2d2d', highlightthickness=0)
        self.canvas.pack(fill='x', padx=5, pady=2)
        
        self.percent_label = ttk.Label(self, text="50%", font=("Arial", 12, "bold"))
        self.percent_label.pack(anchor='center')
        
        self.after(100, self.update_gauge)
    
    def set_probability(self, prob):
        """Set the win probability (0.0 to 1.0)"""
        self.probability = max(0.0, min(1.0, prob))
        self.update_gauge()
    
    def update_gauge(self):
        """Update the gauge display"""
        self.canvas.delete("all")
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width < 10:
            width = 300
        
        self.canvas.create_rectangle(0, 0, width, height, fill='#3d3d3d', outline='')
        
        if self.probability >= 0.5:
            intensity = int((self.probability - 0.5) * 2 * 255)
            color = f'#{255-intensity:02x}ff{100:02x}'
        elif self.probability >= 0.25:
            intensity = int((self.probability - 0.25) * 4 * 255)
            color = f'#ff{intensity:02x}00'
        else:
            intensity = int(self.probability * 4 * 255)
            color = f'#{255:02x}{intensity:02x}00'
        
        fill_width = int(width * self.probability)
        self.canvas.create_rectangle(0, 0, fill_width, height, fill=color, outline='')
        self.canvas.create_rectangle(0, 0, width-1, height-1, outline='#555555', width=2)
        
        self.percent_label.config(text=f"{self.probability * 100:.1f}%")
        
        if self.probability >= 0.5:
            self.percent_label.config(foreground='#00ff00')
        elif self.probability >= 0.25:
            self.percent_label.config(foreground='#ffff00')
        else:
            self.percent_label.config(foreground='#ff0000')


# ==========================================================
#  Main Application Class
# ==========================================================
class TicTacToeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧠 TicTacToe AI Lab - GOD MODE Edition")
        self.geometry("1100x900")
        self.configure(bg="#1e1e2e")
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self._configure_styles()
        
        # Create notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Initialize game state
        self.game = ClassicTicTacToe()
        self.scores = {"Player": 0, "AI": 0, "Draws": 0}
        self.ai_model = None
        self.ai_difficulty = "god"
        self.show_confidence = True
        self.move_history = []
        self._animation_speed = 500
        
        # ============================================================
        # LOGIC FIX: Explicit human_symbol and ai_symbol variables
        # These are set in reset_game() and used in update_game_over()
        # ============================================================
        self.human_symbol = PLAYER_X  # Will be set properly in reset_game()
        self.ai_symbol = PLAYER_O     # Will be set properly in reset_game()
        
        # Debug Dashboard reference
        self.debug_window = None
        
        # GOD MODE MCTS Player
        self.mcts_god = MCTSPlayer(PLAYER_O, simulations=2000)
        self.last_human_move = None
        
        # Training state
        self.training_active = False
        self.training_thread = None
        self.agent = None
        self.env = None
        self.episode_rewards = []
        self.current_episodes = 0
        self.target_episodes = 0
        
        # Create tabs
        self.create_game_tab()
        self.create_training_tab()
        self.create_arena_tab()
        self.create_settings_tab()
        
        # Update model list
        self.update_model_list()
        
        # Bind keyboard shortcut for debug dashboard
        self.bind('<Control-d>', lambda e: self.toggle_debug_dashboard())
        
        # Start with a proper game reset
        self.reset_game()
    
    def _configure_styles(self):
        """Configure custom ttk styles"""
        self.style.configure('TFrame', background='#1e1e2e')
        self.style.configure('TLabel', background='#1e1e2e', foreground='#ffffff')
        self.style.configure('TButton', padding=5)
        self.style.configure('TLabelframe', background='#1e1e2e', foreground='#ffffff')
        self.style.configure('TLabelframe.Label', background='#1e1e2e', foreground='#00ff00')
        self.style.configure('TrashTalk.TLabel', font=('Arial', 11, 'italic'), foreground='#ff9900')
    
    def open_debug_dashboard(self):
        """Open or focus the debug dashboard"""
        if self.debug_window is None or not self.debug_window.winfo_exists():
            self.debug_window = DebugWindow(self)
            
            # Connect MCTS to debug callback
            if self.mcts_god:
                self.mcts_god.set_debug_callback(self.debug_log)
                
            self.debug_log("Debug Dashboard connected to main application", "success")
        else:
            self.debug_window.lift()
            self.debug_window.focus_force()
    
    def toggle_debug_dashboard(self):
        """Toggle debug dashboard visibility"""
        if self.debug_window is None or not self.debug_window.winfo_exists():
            self.open_debug_dashboard()
        else:
            self.debug_window.destroy()
            self.debug_window = None
    
    def debug_log(self, message, level='info'):
        """Log message to debug dashboard if open"""
        if self.debug_window and self.debug_window.winfo_exists():
            self.debug_window.log_decision(message, level)
    
    def debug_log_training(self, episode, total, reward, epsilon, loss=None):
        """Log training progress to debug dashboard"""
        if self.debug_window and self.debug_window.winfo_exists():
            self.debug_window.log_training(episode, total, reward, epsilon, loss)
    
    def debug_log_move_candidates(self, candidates):
        """Log move candidates to debug dashboard"""
        if self.debug_window and self.debug_window.winfo_exists():
            self.debug_window.log_move_candidates(candidates)
        
    def create_game_tab(self):
        self.game_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.game_frame, text="🎮 Play vs GOD MODE AI")
        
        # Top section: Win Probability Gauge
        gauge_frame = ttk.LabelFrame(self.game_frame, text="⚡ AI STATUS")
        gauge_frame.pack(fill='x', padx=10, pady=5)
        
        self.win_gauge = WinProbabilityGauge(gauge_frame)
        self.win_gauge.pack(fill='x', padx=5, pady=5)
        
        # Trash Talk Label
        self.trash_talk_label = ttk.Label(
            gauge_frame, 
            text="Initializing God Mode... 🤖",
            style='TrashTalk.TLabel',
            wraplength=600
        )
        self.trash_talk_label.pack(pady=5)
        
        # Game controls
        controls_frame = ttk.LabelFrame(self.game_frame, text="Game Controls")
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        # AI mode selection
        mode_frame = ttk.Frame(controls_frame)
        mode_frame.pack(fill='x', pady=5)
        
        ttk.Label(mode_frame, text="AI Mode:").pack(side='left', padx=5)
        self.ai_mode_var = tk.StringVar(value="god")
        modes = [("🔥 GOD MODE (MCTS)", "god"), ("Hard (Model)", "hard"), 
                 ("Medium", "medium"), ("Easy", "easy")]
        for text, value in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.ai_mode_var, 
                           value=value, command=self.on_mode_change).pack(side='left', padx=5)
        
        # Model selection
        model_frame = ttk.Frame(controls_frame)
        model_frame.pack(fill='x', pady=5)
        
        ttk.Label(model_frame, text="AI Model (for Hard mode):").pack(side='left', padx=5)
        self.ai_model_var = tk.StringVar()
        self.ai_model_combo = ttk.Combobox(model_frame, textvariable=self.ai_model_var, width=30)
        self.ai_model_combo.pack(side='left', fill='x', expand=True, padx=5)
        
        ttk.Button(model_frame, text="Load Model", command=self.load_ai_model).pack(side='left', padx=5)
        ttk.Button(model_frame, text="Refresh", command=self.update_model_list).pack(side='left', padx=5)
        
        # Game buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(fill='x', pady=5)
        
        ttk.Button(button_frame, text="🔄 New Game", command=self.reset_game).pack(side='left', padx=5)
        ttk.Button(button_frame, text="🔧 Open Debug Dashboard", 
                  command=self.open_debug_dashboard).pack(side='left', padx=5)
        
        self.show_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(button_frame, text="Show AI Analysis", 
                       variable=self.show_analysis_var).pack(side='left', padx=5)
        
        # Score display
        score_frame = ttk.LabelFrame(self.game_frame, text="📊 Score")
        score_frame.pack(fill='x', padx=10, pady=5)
        
        self.score_label = ttk.Label(score_frame, text="Player: 0 | AI: 0 | Draws: 0", 
                                     font=("Arial", 14, "bold"))
        self.score_label.pack(pady=5)
        
        # Role indicator - shows who is X and who is O
        self.role_label = ttk.Label(score_frame, text="", font=("Arial", 10))
        self.role_label.pack(pady=2)
        
        # Game board
        board_frame = ttk.LabelFrame(self.game_frame, text="🎯 Game Board")
        board_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(board_frame, width=360, height=360, bg="#2d2d2d", 
                               highlightthickness=2, highlightbackground="#00ff00")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Status label
        self.status_label = ttk.Label(self.game_frame, text="Your turn", 
                                      font=("Arial", 16, "bold"))
        self.status_label.pack(pady=5)
        
        # Move history
        history_frame = ttk.LabelFrame(self.game_frame, text="📜 Move History")
        history_frame.pack(fill='x', padx=10, pady=5)
        
        self.history_text = tk.Text(history_frame, height=4, state='disabled',
                                   bg='#2d2d2d', fg='#ffffff')
        self.history_text.pack(fill='x', padx=5, pady=5)
        
        # AI Analysis Panel
        self.analysis_frame = ttk.LabelFrame(self.game_frame, text="🔬 AI Analysis")
        self.analysis_frame.pack(fill='x', padx=10, pady=5)
        
        self.analysis_text = tk.Text(self.analysis_frame, height=4, state='disabled',
                                     bg='#1a1a2e', fg='#00ff00', font=('Consolas', 10))
        self.analysis_text.pack(fill='x', padx=5, pady=5)
        
    def create_training_tab(self):
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="🏋️ Train AI (vs MCTS)")
        
        # Training info
        info_frame = ttk.LabelFrame(self.training_frame, text="ℹ️ Training Info")
        info_frame.pack(fill='x', padx=10, pady=5)
        
        info_text = """OPTIMIZED TRAINING:
• Dense-only neural network (no CNN) for fast convergence
• Training against MCTS opponent for high-quality data
• Experience augmentation for 6x more training samples
• Real-time monitoring in Debug Dashboard (Ctrl+D)"""
        
        ttk.Label(info_frame, text=info_text, justify='left').pack(padx=10, pady=5)
        
        # DQN parameters
        self.dqn_frame = ttk.LabelFrame(self.training_frame, text="DQN Parameters")
        self.dqn_frame.pack(fill='x', padx=10, pady=10)
        
        self.dqn_params = {}
        params = [
            ("Model Name:", "model_name", "god_mode_agent_v1"),
            ("Learning Rate:", "learning_rate", 0.001),
            ("Epsilon Decay:", "epsilon_decay", 0.9995),
            ("Batch Size:", "batch_size", 64),
            ("Target Update Freq:", "target_update_freq", 10),
            ("Training Episodes:", "episodes", 10000),
            ("MCTS Simulations (opponent):", "mcts_sims", 300),
            ("Save Frequency:", "save_freq", 500),
        ]
        
        for i, (label, key, default) in enumerate(params):
            ttk.Label(self.dqn_frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            if isinstance(default, str):
                var = tk.StringVar(value=default)
            elif isinstance(default, float):
                var = tk.DoubleVar(value=default)
            else:
                var = tk.IntVar(value=default)
            self.dqn_params[key] = var
            ttk.Entry(self.dqn_frame, textvariable=var, width=20).grid(row=i, column=1, sticky='w', padx=5, pady=2)
        
        # Training control
        control_frame = ttk.Frame(self.training_frame)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        self.train_button = ttk.Button(control_frame, text="▶️ Start Training", 
                                        command=self.start_training_thread)
        self.train_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="⏹️ Stop", 
                                       command=self.stop_training, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        self.save_button = ttk.Button(control_frame, text="💾 Save Model", 
                                       command=self.save_model, state='disabled')
        self.save_button.pack(side='left', padx=5)
        
        ttk.Button(control_frame, text="🔧 Open Debug Dashboard",
                  command=self.open_debug_dashboard).pack(side='left', padx=5)
        
        # Progress bar
        progress_frame = ttk.Frame(self.training_frame)
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(progress_frame, text="Training Progress:").pack(anchor='w')
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=5)
        
        # Training log
        log_frame = ttk.LabelFrame(self.training_frame, text="📋 Training Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(log_frame, height=8, state='disabled',
                               bg='#1a1a2e', fg='#00ff00', font=('Consolas', 9))
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Training plot
        plot_frame = ttk.LabelFrame(self.training_frame, text="📈 Training Progress")
        plot_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.training_fig = Figure(figsize=(8, 3), dpi=100, facecolor='#1e1e2e')
        self.training_ax = self.training_fig.add_subplot(111)
        self.training_ax.set_facecolor('#2d2d2d')
        self.training_ax.tick_params(colors='white')
        self.training_ax.set_title("Episode Rewards", color='white')
        self.training_ax.set_xlabel("Episode", color='white')
        self.training_ax.set_ylabel("Reward", color='white')
        self.training_canvas = FigureCanvasTkAgg(self.training_fig, master=plot_frame)
        self.training_canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def create_arena_tab(self):
        self.arena_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.arena_frame, text="🏟️ Model Arena")
        
        # Model selection
        select_frame = ttk.LabelFrame(self.arena_frame, text="Select Models to Compete")
        select_frame.pack(fill='x', padx=10, pady=10)
        
        list_frame = ttk.Frame(select_frame)
        list_frame.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.model_listbox = tk.Listbox(list_frame, selectmode='multiple', height=8,
                                        bg='#2d2d2d', fg='#ffffff',
                                        yscrollcommand=scrollbar.set)
        self.model_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.model_listbox.yview)
        
        ttk.Button(select_frame, text="Refresh Models", command=self.update_model_list).pack(pady=5)
        
        # Add MCTS option
        mcts_frame = ttk.Frame(select_frame)
        mcts_frame.pack(fill='x', pady=5)
        
        self.include_mcts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mcts_frame, text="Include MCTS God Mode in Tournament",
                       variable=self.include_mcts_var).pack(side='left')
        
        # Tournament settings
        settings_frame = ttk.LabelFrame(self.arena_frame, text="Tournament Settings")
        settings_frame.pack(fill='x', padx=10, pady=10)
        
        games_frame = ttk.Frame(settings_frame)
        games_frame.pack(fill='x', pady=5)
        
        ttk.Label(games_frame, text="Games per Matchup:").pack(side='left')
        self.num_games_var = tk.IntVar(value=50)
        ttk.Spinbox(games_frame, from_=10, to=500, textvariable=self.num_games_var, 
                   width=10).pack(side='left', padx=5)
        
        control_frame = ttk.Frame(self.arena_frame)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        self.tournament_button = ttk.Button(control_frame, text="🏆 Start Tournament", 
                                            command=self.start_tournament)
        self.tournament_button.pack(side='left', padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(self.arena_frame, text="Tournament Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, height=15, state='disabled',
                                   bg='#1a1a2e', fg='#00ff00', font=('Consolas', 10))
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)
        
    def create_settings_tab(self):
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="⚙️ Settings")
        
        # MCTS Settings
        mcts_frame = ttk.LabelFrame(self.settings_frame, text="MCTS God Mode Settings")
        mcts_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(mcts_frame, text="MCTS Simulations per Move:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.mcts_sims_var = tk.IntVar(value=2000)
        ttk.Spinbox(mcts_frame, from_=500, to=10000, textvariable=self.mcts_sims_var,
                   width=10).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(mcts_frame, text="(Higher = Stronger but slower)").grid(row=0, column=2, sticky='w', padx=5)
        
        # General settings
        general_frame = ttk.LabelFrame(self.settings_frame, text="General Settings")
        general_frame.pack(fill='x', padx=10, pady=10)
        
        self.settings_vars = {}
        settings = [
            ("Animation Speed (ms):", "animation_speed", 500),
            ("Enable Trash Talk:", "trash_talk", True),
            ("Show Move Analysis:", "show_analysis", True),
        ]
        
        for i, (label, key, default) in enumerate(settings):
            ttk.Label(general_frame, text=label).grid(row=i, column=0, sticky='w', padx=5, pady=5)
            if isinstance(default, bool):
                var = tk.BooleanVar(value=default)
                self.settings_vars[key] = var
                ttk.Checkbutton(general_frame, variable=var).grid(row=i, column=1, sticky='w', padx=5, pady=5)
            else:
                var = tk.IntVar(value=default)
                self.settings_vars[key] = var
                ttk.Spinbox(general_frame, from_=100, to=2000, textvariable=var,
                           width=10).grid(row=i, column=1, sticky='w', padx=5, pady=5)
        
        # Debug Dashboard button
        debug_frame = ttk.LabelFrame(self.settings_frame, text="Developer Tools")
        debug_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(debug_frame, text="🔧 Open Debug Dashboard (Ctrl+D)",
                  command=self.open_debug_dashboard).pack(pady=10)
        
        ttk.Label(debug_frame, text="The Debug Dashboard shows real-time AI decision logic,\n"
                                    "MCTS simulations, and training metrics.",
                 justify='left').pack(padx=10, pady=5)
        
        ttk.Button(general_frame, text="Apply Settings", 
                  command=self.apply_settings).grid(row=len(settings), column=0, columnspan=2, pady=10)
        
    def update_model_list(self):
        """Update available models list"""
        models = [os.path.basename(f) for f in glob.glob(os.path.join(MODEL_DIR, "*.h5"))]
        
        if hasattr(self, 'model_listbox'):
            self.model_listbox.delete(0, tk.END)
            for model in models:
                self.model_listbox.insert(tk.END, model)
        
        if hasattr(self, 'ai_model_combo'):
            current = self.ai_model_var.get()
            self.ai_model_combo['values'] = models
            if current in models:
                self.ai_model_var.set(current)
            elif models:
                self.ai_model_var.set(models[0])
                
    def on_mode_change(self):
        """Handle AI mode change"""
        mode = self.ai_mode_var.get()
        if mode == "god":
            self.mcts_god = MCTSPlayer(PLAYER_O, simulations=self.mcts_sims_var.get())
            if self.debug_window:
                self.mcts_god.set_debug_callback(self.debug_log)
            self.trash_talk_label.config(text="God Mode activated. I have analyzed all 255,168 possible games. 🧠")
            self.debug_log("AI Mode changed to: GOD MODE (MCTS)", "success")
        else:
            self.trash_talk_label.config(text="Standard AI mode. You might have a chance... 😏")
            self.debug_log(f"AI Mode changed to: {mode.upper()}", "info")
            
    def load_ai_model(self):
        """Load selected AI model"""
        model_name = self.ai_model_var.get()
        if not model_name:
            messagebox.showwarning("No Model", "Please select an AI model.")
            return
            
        model_path = os.path.join(MODEL_DIR, model_name)
        self.ai_model = load_model_compatible(model_path, compile_model=False)
        
        if self.ai_model is None:
            messagebox.showerror("Error", f"Failed to load model {model_name}")
            return
        
        self.debug_log(f"Loaded AI model: {model_name}", "success")
        messagebox.showinfo("Model Loaded", f"Successfully loaded {model_name}")
        
    def on_canvas_click(self, event):
        """Handle player move on canvas"""
        # ============================================================
        # LOGIC FIX: Check if it's the human's turn using human_symbol
        # ============================================================
        if self.game.game_over or self.game.current_player != self.human_symbol:
            self.debug_log(f"Click ignored - not human's turn (current: {'X' if self.game.current_player == PLAYER_X else 'O'}, human: {'X' if self.human_symbol == PLAYER_X else 'O'})", "warning")
            return

        cell_size = 120
        col = event.x // cell_size
        row = event.y // cell_size
        pos = row * 3 + col

        if 0 <= pos < BOARD_SIZE and self.game.board[pos] == EMPTY:
            self.last_human_move = pos
            self.game.make_move(pos)
            
            human_symbol_str = "X" if self.human_symbol == PLAYER_X else "O"
            self.move_history.append(f"Human ({human_symbol_str}) → Cell {pos + 1}")
            
            self.debug_log(f"Human played Cell {pos + 1} as {human_symbol_str}", "info")
            
            self.update_display()
            
            if self.game.game_over:
                self.update_game_over()
            else:
                self.status_label.config(text="AI is calculating... 🤔")
                self.update()
                
                delay = self.settings_vars.get('animation_speed', tk.IntVar(value=500)).get()
                self.after(delay, self.ai_move)
                
    def ai_move(self):
        """Execute AI move"""
        if self.game.game_over:
            return
        
        # ============================================================
        # LOGIC FIX: Check if it's actually the AI's turn
        # ============================================================
        if self.game.current_player != self.ai_symbol:
            self.debug_log("AI move called but it's not AI's turn!", "warning")
            return
            
        mode = self.ai_mode_var.get()
        ai_symbol_str = "X" if self.ai_symbol == PLAYER_X else "O"
        
        self.debug_log(f"AI ({ai_symbol_str}) computing move using {mode.upper()} mode", "mcts")
        
        if mode == "god":
            # Update MCTS simulations from settings
            self.mcts_god.simulations = self.mcts_sims_var.get()
            self.mcts_god.player = self.ai_symbol  # LOGIC FIX: Use ai_symbol, not current_player
            
            if self.debug_window:
                self.mcts_god.set_debug_callback(self.debug_log)
            
            # Get move using MCTS
            ai_move = self.mcts_god.get_move(self.game.board)
            
            # Get analysis for display
            analysis = self.mcts_god.analyze_position(
                self.game.board, 
                self.last_human_move
            )
            
            # Update Win Probability Gauge
            win_prob = analysis['win_probability']
            self.win_gauge.set_probability(win_prob)
            
            # Update Trash Talk
            if self.settings_vars.get('trash_talk', tk.BooleanVar(value=True)).get():
                self.trash_talk_label.config(text=analysis['trash_talk'])
            
            # Update Analysis Panel
            if self.show_analysis_var.get():
                self.update_analysis_panel(analysis)
            
            # Log move candidates to debug dashboard
            if self.mcts_god.move_analysis:
                candidates = []
                for cell, data in self.mcts_god.move_analysis.items():
                    candidates.append({
                        'cell': cell + 1,
                        'visits': data['visits'],
                        'win_rate': data['win_rate'],
                        'selected': cell == ai_move
                    })
                self.debug_log_move_candidates(candidates)
                
        else:
            # Use loaded model or random
            if self.ai_model is None:
                empty_positions = self.game.get_empty_positions()
                ai_move = random.choice(empty_positions) if empty_positions else None
                self.debug_log("No model loaded - using random move", "warning")
            else:
                ai_move = self.game.get_ai_move(self.ai_model, mode)
                self.debug_log(f"Model selected Cell {ai_move + 1}", "success")
                
        if ai_move is not None:
            self.game.make_move(ai_move)
            self.move_history.append(f"AI ({ai_symbol_str}) → Cell {ai_move + 1}")
            self.debug_log(f"AI played Cell {ai_move + 1} as {ai_symbol_str}", "success")
            
        if self.game.game_over:
            self.update_game_over()
        else:
            # ============================================================
            # LOGIC FIX: Check whose turn it is using explicit symbols
            # ============================================================
            if self.game.current_player == self.human_symbol:
                human_symbol_str = "X" if self.human_symbol == PLAYER_X else "O"
                self.status_label.config(text=f"Your turn ({human_symbol_str})")
            else:
                self.status_label.config(text="AI is thinking...")
                self.after(self._animation_speed, self.ai_move)
                
        self.update_display()
        
    def update_analysis_panel(self, analysis):
        """Update the AI analysis panel"""
        self.analysis_text.config(state='normal')
        self.analysis_text.delete(1.0, tk.END)
        
        text = f"""Position Quality: {analysis['position_quality'].upper()}
Win Probability: {analysis['win_probability']*100:.1f}%
Decision: {analysis.get('decision_reason', 'N/A')}
"""
        if analysis['moves_to_win']:
            text += f"Estimated Moves to Win: {analysis['moves_to_win']}\n"
            
        # Add move analysis if available
        if hasattr(self.mcts_god, 'move_analysis') and self.mcts_god.move_analysis:
            text += "\nTop Move Candidates:\n"
            sorted_moves = sorted(self.mcts_god.move_analysis.items(), 
                                 key=lambda x: x[1]['visits'], reverse=True)[:3]
            for move, data in sorted_moves:
                text += f"  Cell {move+1}: {data['visits']} sims, {data['win_rate']*100:.1f}% win\n"
        
        self.analysis_text.insert(tk.END, text)
        self.analysis_text.config(state='disabled')
        
    def update_game_over(self):
        """
        ============================================================
        LOGIC FIX: Determine winner using explicit human_symbol and ai_symbol
        NOT using first_player which was causing the inversion bug
        ============================================================
        """
        winner = self.game.winner
        
        self.debug_log(f"Game Over! Winner value: {winner}", "info")
        self.debug_log(f"Human symbol: {'X' if self.human_symbol == PLAYER_X else 'O'}, AI symbol: {'X' if self.ai_symbol == PLAYER_X else 'O'}", "info")
        
        if winner == self.human_symbol:
            # Human wins
            self.scores["Player"] += 1
            self.status_label.config(text="🎉 YOU WIN! 🎉")
            self.trash_talk_label.config(text="ERROR: This should not be possible in God Mode. Recalibrating... 🤯")
            self.debug_log("RESULT: Human WINS", "success")
            
        elif winner == self.ai_symbol:
            # AI wins
            self.scores["AI"] += 1
            self.status_label.config(text="🤖 AI WINS! 🤖")
            self.trash_talk_label.config(text="Victory achieved. Your defeat was inevitable. 😎")
            self.debug_log("RESULT: AI WINS", "critical")
            
        elif winner == 0:
            # Draw
            self.scores["Draws"] += 1
            self.status_label.config(text="🤝 IT'S A DRAW! 🤝")
            self.trash_talk_label.config(text="A perfect game. We are evenly matched... for now. ♟️")
            self.debug_log("RESULT: DRAW", "warning")
        else:
            self.debug_log(f"Unexpected winner value: {winner}", "critical")
            
        self.win_gauge.set_probability(0.5)
        self.update_display()
        
    def reset_game(self):
        """
        ============================================================
        LOGIC FIX: Properly assign human_symbol and ai_symbol
        PLAYER_X always moves first in standard Tic-Tac-Toe
        Randomly decide if human or AI is X (and thus moves first)
        ============================================================
        """
        # Reset the game board
        self.game.reset()
        self.move_history = []
        self.last_human_move = None
        
        # Randomly decide who plays as X (X always moves first)
        human_goes_first = random.choice([True, False])
        
        if human_goes_first:
            self.human_symbol = PLAYER_X  # Human is X, moves first
            self.ai_symbol = PLAYER_O     # AI is O, moves second
            self.debug_log("New game: Human plays as X (moves first)", "info")
        else:
            self.human_symbol = PLAYER_O  # Human is O, moves second
            self.ai_symbol = PLAYER_X     # AI is X, moves first
            self.debug_log("New game: AI plays as X (moves first)", "info")
        
        # Update MCTS player's symbol
        self.mcts_god.player = self.ai_symbol
        self.mcts_god.opponent = self.human_symbol
        
        # Update role label to show current assignment
        human_symbol_str = "X" if self.human_symbol == PLAYER_X else "O"
        ai_symbol_str = "X" if self.ai_symbol == PLAYER_X else "O"
        self.role_label.config(text=f"You are: {human_symbol_str} (Red) | AI is: {ai_symbol_str} (Blue)")
        
        # Update status and potentially make AI move first
        if human_goes_first:
            self.status_label.config(text=f"Your turn ({human_symbol_str}) - You go first!")
            self.trash_talk_label.config(text="Make your move. I am watching. 👁️")
        else:
            self.status_label.config(text=f"AI's turn ({ai_symbol_str}) - AI goes first!")
            self.trash_talk_label.config(text="I shall make the first move. Observe and learn. 🎓")
            # AI moves first after a short delay
            self.after(500, self.ai_move)
        
        self.win_gauge.set_probability(0.5)
        self.update_display()
        
        # Clear analysis panel
        self.analysis_text.config(state='normal')
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, "Game started. Awaiting moves...\n")
        self.analysis_text.config(state='disabled')
        
    def update_display(self):
        """Update all display elements"""
        # Score
        self.score_label.config(
            text=f"Player: {self.scores['Player']} | AI: {self.scores['AI']} | Draws: {self.scores['Draws']}"
        )
        
        # Board
        self.draw_board()
        
        # History
        self.history_text.config(state='normal')
        self.history_text.delete(1.0, tk.END)
        for move in reversed(self.move_history[-8:]):
            self.history_text.insert(tk.END, move + "\n")
        self.history_text.config(state='disabled')
        
    def draw_board(self):
        """Draw the game board with correct X/O rendering"""
        self.canvas.delete("all")
        cell_size = 120
        
        # Draw grid lines
        for i in range(1, 3):
            self.canvas.create_line(i * cell_size, 0, i * cell_size, 360, 
                                   fill="#00ff00", width=3)
            self.canvas.create_line(0, i * cell_size, 360, i * cell_size, 
                                   fill="#00ff00", width=3)
        
        # Draw X's and O's
        # ============================================================
        # LOGIC FIX: Draw based on actual board values, not role assumptions
        # PLAYER_X (1) = Draw as X (red)
        # PLAYER_O (-1) = Draw as O (blue)
        # This is now independent of who is human/AI
        # ============================================================
        for i in range(BOARD_SIZE):
            row, col = divmod(i, 3)
            x1, y1 = col * cell_size + 10, row * cell_size + 10
            x2, y2 = (col + 1) * cell_size - 10, (row + 1) * cell_size - 10
            
            if self.game.board[i] == PLAYER_X:
                # Draw X in red
                self.canvas.create_line(x1+10, y1+10, x2-10, y2-10, width=6, fill="#ff4444")
                self.canvas.create_line(x1+10, y2-10, x2-10, y1+10, width=6, fill="#ff4444")
            elif self.game.board[i] == PLAYER_O:
                # Draw O in blue
                self.canvas.create_oval(x1+10, y1+10, x2-10, y2-10, width=6, outline="#4444ff")
                    
    def start_training_thread(self):
        """Start training in a separate thread"""
        if self.training_active:
            messagebox.showwarning("Training Active", "Training is already in progress.")
            return
            
        model_name = self.dqn_params["model_name"].get()
        if not model_name:
            messagebox.showerror("Error", "Please enter a model name.")
            return
        
        # Open debug dashboard if not already open
        self.open_debug_dashboard()
            
        # Initialize agent
        learning_rate = self.dqn_params["learning_rate"].get()
        epsilon_decay = self.dqn_params["epsilon_decay"].get()
        target_episodes = int(self.dqn_params["episodes"].get())
        mcts_sims = int(self.dqn_params["mcts_sims"].get())
        
        self.agent = DQNAgent(
            BOARD_SIZE,
            BOARD_SIZE,
            learning_rate=learning_rate,
            epsilon_decay=epsilon_decay
        )
        
        # Load existing state if available
        state_data = load_training_state(model_name)
        if state_data is not None:
            model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
            if os.path.exists(model_path):
                self.agent.load(model_path)
                self.debug_log(f"Loaded existing model: {model_name}", "success")
            self.agent.epsilon = state_data.get("epsilon", 1.0)
            self.current_episodes = state_data["current_episodes"]
            self.episode_rewards = state_data["episode_rewards"]
            self.target_episodes = state_data.get("target_episodes", target_episodes)
        else:
            self.current_episodes = 0
            self.episode_rewards = []
            self.target_episodes = target_episodes
            
        # Use MCTS training environment
        self.env = MCTSTrainingEnvironment(mcts_simulations=mcts_sims)
        
        # Update UI
        self.training_active = True
        self.train_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.save_button.config(state='normal')
        
        # Log to debug dashboard
        self.debug_log(f"Starting training: {model_name}", "success")
        self.debug_log(f"Target episodes: {target_episodes}", "info")
        self.debug_log(f"MCTS opponent simulations: {mcts_sims}", "info")
        self.debug_log(f"Learning rate: {learning_rate}", "info")
        self.debug_log(f"Epsilon decay: {epsilon_decay}", "info")
        
        if self.debug_window:
            self.debug_window.log_training_event(f"=== TRAINING STARTED: {model_name} ===")
            self.debug_window.log_training_event(f"Episodes: {self.current_episodes}/{target_episodes}")
            self.debug_window.log_training_event(f"Initial Epsilon: {self.agent.epsilon:.4f}")
        
        # Log to main window
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f"Starting training: {model_name}\n")
        self.log_text.insert(tk.END, f"Training against MCTS ({mcts_sims} sims)\n")
        self.log_text.insert(tk.END, "Dense-only network for fast convergence\n")
        self.log_text.insert(tk.END, "Open Debug Dashboard (Ctrl+D) for real-time monitoring\n")
        self.log_text.config(state='disabled')
        
        self.training_thread = threading.Thread(target=self.run_training, daemon=True)
        self.training_thread.start()
        
    def run_training(self):
        """Training loop with debug dashboard integration"""
        batch_size = int(self.dqn_params["batch_size"].get())
        target_update_freq = int(self.dqn_params["target_update_freq"].get())
        save_freq = int(self.dqn_params["save_freq"].get())
        
        wins = 0
        losses = 0
        draws = 0
        
        for ep in range(self.current_episodes, self.target_episodes):
            if not self.training_active:
                break
                
            state = self.env.reset()
            ep_reward = 0.0
            ep_loss = 0.0
            steps = 0
            
            for _ in range(9):
                valid_moves = [i for i in range(BOARD_SIZE) if state[i] == EMPTY]
                if not valid_moves:
                    break
                    
                if self.env.current_player == PLAYER_X:
                    action = self.agent.act(state, valid_moves)
                    next_state, reward, done = self.env.step(action)
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    ep_reward += reward
                    steps += 1
                else:
                    state, _, done = self.env.step(0)
                    
                if done:
                    # Track game outcomes
                    winner = self.env.check_winner()
                    if winner == PLAYER_X:
                        wins += 1
                    elif winner == PLAYER_O:
                        losses += 1
                    else:
                        draws += 1
                    break
            
            # Replay and get loss
            loss = self.agent.replay(batch_size)
            ep_loss = loss if loss else 0.0
            
            if (ep + 1) % target_update_freq == 0:
                self.agent.update_target_model()
                
            if (ep + 1) % save_freq == 0:
                self.agent.save(self.dqn_params["model_name"].get())
                self.debug_log(f"Model checkpoint saved at episode {ep + 1}", "success")
                
            self.current_episodes = ep + 1
            self.episode_rewards.append(ep_reward)
            
            # Update debug dashboard every episode
            if self.debug_window and self.debug_window.winfo_exists():
                self.after(0, lambda e=ep+1, t=self.target_episodes, r=ep_reward, 
                          eps=self.agent.epsilon, l=ep_loss: 
                          self.debug_log_training(e, t, r, eps, l))
            
            # Update main UI every 10 episodes
            if (ep + 1) % 10 == 0:
                self.after(0, self.update_training_progress)
            
            # Detailed log every 100 episodes
            if (ep + 1) % 100 == 0:
                win_rate = wins / max(1, wins + losses + draws) * 100
                draw_rate = draws / max(1, wins + losses + draws) * 100
                
                def update_log(episode=ep+1, epsilon=self.agent.epsilon, 
                              reward=ep_reward, wr=win_rate, dr=draw_rate,
                              w=wins, l=losses, d=draws):
                    self.log_text.config(state='normal')
                    self.log_text.insert(tk.END, 
                        f"Ep {episode}: ε={epsilon:.4f}, R={reward:+.2f}, "
                        f"W/L/D={w}/{l}/{d} ({wr:.1f}%/{dr:.1f}%)\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state='disabled')
                    
                    # Also log to debug dashboard
                    if self.debug_window and self.debug_window.winfo_exists():
                        self.debug_window.log_training_event(
                            f"Milestone: Episode {episode} | Win Rate: {wr:.1f}% | "
                            f"Epsilon: {epsilon:.4f}"
                        )
                        
                self.after(0, update_log)
                
                # Reset counters
                wins = 0
                losses = 0
                draws = 0
                
        # Training complete
        model_name = self.dqn_params["model_name"].get()
        self.agent.save(model_name)
        save_training_state(
            model_name,
            self.agent,
            self.current_episodes,
            self.episode_rewards,
            self.dqn_params["learning_rate"].get(),
            self.dqn_params["epsilon_decay"].get(),
            self.target_episodes
        )
        
        def finish():
            self.training_active = False
            self.train_button.config(state='normal')
            self.stop_button.config(state='disabled')
            
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, f"\n{'='*50}\n")
            self.log_text.insert(tk.END, f"Training complete! Model saved: {model_name}.h5\n")
            self.log_text.insert(tk.END, f"Final Epsilon: {self.agent.epsilon:.4f}\n")
            self.log_text.insert(tk.END, f"Total Episodes: {self.current_episodes}\n")
            self.log_text.insert(tk.END, f"{'='*50}\n")
            self.log_text.config(state='disabled')
            
            self.debug_log(f"Training complete! Model saved: {model_name}.h5", "success")
            if self.debug_window and self.debug_window.winfo_exists():
                self.debug_window.log_training_event("=== TRAINING COMPLETE ===")
                self.debug_window.log_training_event(f"Final model: {model_name}.h5")
            
            self.update_model_list()
            
        self.after(0, finish)
        
    def stop_training(self):
        """Stop training"""
        self.training_active = False
        self.train_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, "\nTraining stopped by user.\n")
        self.log_text.config(state='disabled')
        
        self.debug_log("Training stopped by user", "warning")
        if self.debug_window and self.debug_window.winfo_exists():
            self.debug_window.log_training_event("=== TRAINING STOPPED ===")
        
    def save_model(self):
        """Save current model"""
        if self.agent:
            model_name = self.dqn_params["model_name"].get()
            path = self.agent.save(model_name)
            
            save_training_state(
                model_name,
                self.agent,
                self.current_episodes,
                self.episode_rewards,
                self.dqn_params["learning_rate"].get(),
                self.dqn_params["epsilon_decay"].get(),
                self.target_episodes
            )
            
            self.debug_log(f"Model manually saved: {path}", "success")
            messagebox.showinfo("Saved", f"Model saved to {path}")
            self.update_model_list()
            
    def update_training_progress(self):
        """Update training progress display"""
        progress = self.current_episodes / max(self.target_episodes, 1) * 100
        self.progress_var.set(progress)
        
        self.training_ax.clear()
        if self.episode_rewards:
            # Plot with smoothing
            rewards = self.episode_rewards
            if len(rewards) > 100:
                # Moving average
                window = 50
                smoothed = []
                for i in range(len(rewards)):
                    start = max(0, i - window)
                    smoothed.append(np.mean(rewards[start:i+1]))
                self.training_ax.plot(smoothed, color='#00ff00', linewidth=1, label='Smoothed')
                self.training_ax.plot(rewards, color='#00ff00', linewidth=0.3, alpha=0.3, label='Raw')
            else:
                self.training_ax.plot(rewards, color='#00ff00', linewidth=1)
                
        self.training_ax.set_facecolor('#2d2d2d')
        self.training_ax.tick_params(colors='white')
        self.training_ax.set_xlabel("Episode", color='white')
        self.training_ax.set_ylabel("Reward", color='white')
        self.training_ax.set_title(
            f"Training Progress ({self.current_episodes}/{self.target_episodes}) - "
            f"ε: {self.agent.epsilon:.4f}" if self.agent else "Training Progress", 
            color='white'
        )
        self.training_ax.grid(True, alpha=0.3)
        self.training_canvas.draw()
        
    def start_tournament(self):
        """Start model tournament"""
        selected_indices = self.model_listbox.curselection()
        models = [self.model_listbox.get(i) for i in selected_indices]
        
        if self.include_mcts_var.get():
            models.append("MCTS_GOD_MODE")
            
        if len(models) < 2:
            messagebox.showwarning("Select Models", "Please select at least 2 models.")
            return
        
        self.debug_log(f"Starting tournament with {len(models)} competitors", "info")
        for model in models:
            self.debug_log(f"  - {model}", "info")
            
        num_games = self.num_games_var.get()
        
        tournament_thread = threading.Thread(
            target=self.run_tournament, 
            args=(models, num_games), 
            daemon=True
        )
        tournament_thread.start()
        
    def run_tournament(self, models, num_games):
        """Run tournament between models"""
        results = {model: {"wins": 0, "losses": 0, "draws": 0, "points": 0} for model in models}
        matchup_results = []
        
        # Load all models
        loaded_models = {}
        for model in models:
            if model == "MCTS_GOD_MODE":
                loaded_models[model] = MCTSPlayer(simulations=1000)
                self.debug_log(f"Loaded MCTS_GOD_MODE (1000 simulations)", "success")
            else:
                agent = DQNAgent(BOARD_SIZE, BOARD_SIZE)
                try:
                    agent.load(os.path.join(MODEL_DIR, model))
                    loaded_models[model] = agent
                    self.debug_log(f"Loaded model: {model}", "success")
                except Exception as e:
                    self.debug_log(f"Failed to load {model}: {str(e)}", "critical")
                    continue
        
        total_matches = len(models) * (len(models) - 1) // 2
        current_match = 0
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1_name = models[i]
                model2_name = models[j]
                
                if model1_name not in loaded_models or model2_name not in loaded_models:
                    continue
                
                current_match += 1
                
                def update_progress(m1=model1_name, m2=model2_name, cm=current_match, tm=total_matches):
                    self.results_text.config(state='normal')
                    self.results_text.delete(1.0, tk.END)
                    self.results_text.insert(tk.END, f"Running match {cm}/{tm}: {m1} vs {m2}\n")
                    self.results_text.insert(tk.END, f"Games: 0/{self.num_games_var.get()}\n")
                    self.results_text.config(state='disabled')
                    self.debug_log(f"Match {cm}/{tm}: {m1} vs {m2}", "info")
                self.after(0, update_progress)
                
                model1 = loaded_models[model1_name]
                model2 = loaded_models[model2_name]
                
                m1_wins = 0
                m2_wins = 0
                match_draws = 0
                
                for game_num in range(num_games):
                    env = TicTacToeEnvironment()
                    state = env.reset()
                    done = False
                    
                    # Alternate who goes first
                    first_player = PLAYER_X if game_num % 2 == 0 else PLAYER_O
                    env.current_player = first_player
                    
                    # Assign models to symbols for this game
                    if game_num % 2 == 0:
                        x_model, o_model = model1, model2
                        x_name, o_name = model1_name, model2_name
                    else:
                        x_model, o_model = model2, model1
                        x_name, o_name = model2_name, model1_name
                    
                    while not done:
                        valid_moves = env.get_valid_moves()
                        if not valid_moves:
                            break
                            
                        if env.current_player == PLAYER_X:
                            if isinstance(x_model, MCTSPlayer):
                                x_model.player = PLAYER_X
                                action = x_model.get_move(state)
                            else:
                                action = x_model.act(state, valid_moves)
                        else:
                            if isinstance(o_model, MCTSPlayer):
                                o_model.player = PLAYER_O
                                action = o_model.get_move(state)
                            else:
                                action = o_model.act(state, valid_moves)
                                
                        if action not in valid_moves:
                            action = random.choice(valid_moves) if valid_moves else 0
                            
                        state, _, done = env.step(action)
                        
                    # Determine winner
                    winner = env.winner
                    if winner == PLAYER_X:
                        # X won
                        if x_name == model1_name:
                            m1_wins += 1
                            results[model1_name]["wins"] += 1
                            results[model1_name]["points"] += 3
                            results[model2_name]["losses"] += 1
                        else:
                            m2_wins += 1
                            results[model2_name]["wins"] += 1
                            results[model2_name]["points"] += 3
                            results[model1_name]["losses"] += 1
                    elif winner == PLAYER_O:
                        # O won
                        if o_name == model1_name:
                            m1_wins += 1
                            results[model1_name]["wins"] += 1
                            results[model1_name]["points"] += 3
                            results[model2_name]["losses"] += 1
                        else:
                            m2_wins += 1
                            results[model2_name]["wins"] += 1
                            results[model2_name]["points"] += 3
                            results[model1_name]["losses"] += 1
                    else:
                        # Draw
                        match_draws += 1
                        results[model1_name]["draws"] += 1
                        results[model1_name]["points"] += 1
                        results[model2_name]["draws"] += 1
                        results[model2_name]["points"] += 1
                
                # Log matchup result
                matchup_results.append({
                    'model1': model1_name,
                    'model2': model2_name,
                    'm1_wins': m1_wins,
                    'm2_wins': m2_wins,
                    'draws': match_draws
                })
                
                self.debug_log(
                    f"Matchup complete: {model1_name} {m1_wins}-{match_draws}-{m2_wins} {model2_name}",
                    "success"
                )
                        
        # Display final results
        def show_results():
            self.results_text.config(state='normal')
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "=" * 65 + "\n")
            self.results_text.insert(tk.END, "                    🏆 TOURNAMENT RESULTS 🏆\n")
            self.results_text.insert(tk.END, "=" * 65 + "\n\n")
            
            # Sort by points
            sorted_models = sorted(results.items(), key=lambda x: x[1]['points'], reverse=True)
            
            header = f"{'Rank':<6}{'Model':<28}{'W':<6}{'L':<6}{'D':<6}{'Pts':<6}\n"
            self.results_text.insert(tk.END, header)
            self.results_text.insert(tk.END, "-" * 58 + "\n")
            
            for rank, (model, stats) in enumerate(sorted_models, 1):
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
                row = f"{medal}{rank:<4}{model:<28}{stats['wins']:<6}{stats['losses']:<6}{stats['draws']:<6}{stats['points']:<6}\n"
                self.results_text.insert(tk.END, row)
            
            self.results_text.insert(tk.END, "\n" + "=" * 65 + "\n")
            self.results_text.insert(tk.END, "                      MATCHUP DETAILS\n")
            self.results_text.insert(tk.END, "-" * 65 + "\n")
            
            for matchup in matchup_results:
                self.results_text.insert(tk.END, 
                    f"{matchup['model1']:<25} {matchup['m1_wins']:>3} - {matchup['draws']:>3} - {matchup['m2_wins']:<3} {matchup['model2']}\n"
                )
            
            self.results_text.insert(tk.END, "\n" + "=" * 65 + "\n")
            
            if sorted_models[0][0] == "MCTS_GOD_MODE":
                self.results_text.insert(tk.END, "      🤖 MCTS GOD MODE remains undefeated! 🤖\n")
            else:
                self.results_text.insert(tk.END, f"      🎉 Winner: {sorted_models[0][0]} 🎉\n")
                
            self.results_text.config(state='disabled')
            
            self.debug_log("Tournament complete!", "success")
            self.debug_log(f"Winner: {sorted_models[0][0]} with {sorted_models[0][1]['points']} points", "success")
            
        self.after(0, show_results)
        
    def apply_settings(self):
        """Apply settings"""
        if "animation_speed" in self.settings_vars:
            self._animation_speed = self.settings_vars["animation_speed"].get()
            
        # Update MCTS simulations
        old_sims = self.mcts_god.simulations
        self.mcts_god.simulations = self.mcts_sims_var.get()
        
        self.debug_log(f"Settings applied:", "info")
        self.debug_log(f"  Animation speed: {self._animation_speed}ms", "info")
        self.debug_log(f"  MCTS simulations: {old_sims} -> {self.mcts_god.simulations}", "info")
        
        messagebox.showinfo("Settings", "Settings applied!")


# ==========================================================
#  Main Entry Point
# ==========================================================
if __name__ == "__main__":
    app = TicTacToeApp()
    app.mainloop()