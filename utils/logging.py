import os
from pathlib import Path
from datetime import datetime
import torch
import json

def setup_run_directory(model_name: str) -> Path:
    """Create and return a new run directory"""
    base_dir = Path("runs") / model_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Find next run number
    existing_runs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    next_run = len(existing_runs)

    # Create new run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_{next_run}_{timestamp}"
    run_dir.mkdir()

    return run_dir

def save_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int
):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step':                 step
    }

    checkpoint_path = run_dir / f"checkpoint_{step}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save latest checkpoint reference
    with open(run_dir / "latest.txt", "w") as f:
        f.write(f"checkpoint_{step}.pt")

def load_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """Load latest checkpoint and return the step number"""
    try:
        with open(run_dir / "latest.txt", "r") as f:
            latest = f.read().strip()

        checkpoint_path = run_dir / latest
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['step']

    except FileNotFoundError:
        return 0

def load_run_history(run_dir: Path) -> dict:
    """Load training history from a run directory"""
    try:
        with open(run_dir / "history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}