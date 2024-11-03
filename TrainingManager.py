import json
import time
from pathlib import Path
from typing import Dict, Optional

from utils.logging import setup_run_directory

class TrainingManager:
    """Manages training runs, statistics and checkpoints"""

    def __init__(self):
        self.history: Dict[str, list] = {
            'loss':   [],
            'time':   [],
            'steps':  [],
            'tokens': []
        }
        self.current_run_dir: Optional[Path] = None
        self.total_steps = 0
        self.total_tokens = 0
        self.start_time = time.time()

    def record_stats(self, stats: dict):
        """Record training statistics"""
        for key, value in stats.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        self.total_steps += 1
        self.total_tokens += stats.get('batch_size', 0)

        # Record timestamps
        self.history['time'].append(time.time() - self.start_time)
        self.history['steps'].append(self.total_steps)
        self.history['tokens'].append(self.total_tokens)

        # Save history to run directory
        if self.current_run_dir:
            with open(self.current_run_dir / 'history.json', 'w') as f:
                json.dump(self.history, f)

    def start_new_run(self, model_name: str, config: dict):
        """Start a new training run"""
        self.current_run_dir = setup_run_directory(model_name)
        self.history.clear()
        self.total_steps = 0
        self.total_tokens = 0
        self.start_time = time.time()

        # Save initial config
        with open(self.current_run_dir / 'config.json', 'w') as f:
            json.dump(config, f)

    def load_run(self, run_dir: Path):
        """Load an existing training run"""
        self.current_run_dir = run_dir

        # Load history
        try:
            with open(run_dir / 'history.json', 'r') as f:
                self.history = json.load(f)

            # Reconstruct total counts
            self.total_steps = len(self.history['steps'])
            self.total_tokens = self.history['tokens'][-1] if self.history['tokens'] else 0

        except FileNotFoundError:
            print(f"No history found in {run_dir}, starting fresh")
            self.history.clear()
            self.total_steps = 0
            self.total_tokens = 0

        self.start_time = time.time() - (self.history['time'][-1] if self.history['time'] else 0)