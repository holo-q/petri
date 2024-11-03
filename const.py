import importlib
import pkgutil

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cpu':
    print("Running on CPU, watch out bro")

# Global state for discovered modules and their Petri mappings
petri_modules = []
petri_mappings = []

def discover_petri_modules():
    """Discover and import all modules in models/ directory and collect their __petri__ mappings"""
    global petri_modules, petri_mappings

    # Import all modules in models/ directory
    for _, name, _ in pkgutil.iter_modules(['models']):
        mod = importlib.import_module(f'models.{name}')
        petri_modules.append(mod)

    # Get all __petri__ mappings from imported modules
    for mod in petri_modules:
        if hasattr(mod, '__petri__'):
            petri_mappings.append(mod.__petri__)
