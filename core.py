import datetime
import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import auto, Enum
from pathlib import Path
from typing import Any, Dict, Generic, Optional, override, T, Tuple, Type, TypeVar, Union

import dill
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

import const
from optimizers.cbd import CBD

logger = logging.getLogger(__name__)

# ----------------------------------------
# CONFIG
# ----------------------------------------

class ISerializable:
	"""Mixin class providing dill-based serialization methods"""

	def write(self, path):
		"""Write object to file using dill serialization"""
		import dill
		with open(path, 'wb') as f:
			dill.dump(self, f)

	def get(self, path):
		"""Read object from file using dill serialization"""
		import dill
		with open(path, 'rb') as f:
			loaded = dill.load(f)
			self.__dict__.update(loaded.__dict__)


@dataclass
class Config(ISerializable, Generic[T]):
	"""
	Base class for config around a class T
	This is useful as a configuration interfarce for other objects
	which allows us to selectively expose what we want, instead of
	pulling all of __dict__ into serialization.
	"""

	@abstractmethod
	def create(self, *args, **kwargs) -> T:
		"""Create a new physics instance with these parameters"""
		# Create instance using the class specified by subclass
		raise NotImplementedError

	def set(self, obj: T) -> None:
		"""Set parameters on an existing object instance"""
		# Default implementation using dataclass fields
		for field_name, value in asdict(self).items():
			if hasattr(obj, field_name):
				setattr(obj, field_name, value)
			else:
				print(f"Warning: Field '{field_name}' not found in object instance")

	def get(self, obj: T) -> None:
		"""Read parameters from an existing object instance"""
		# Default implementation using dataclass fields
		for field_name in self.__dict__:
			if hasattr(obj, field_name):
				setattr(self, field_name, getattr(obj, field_name))
			else:
				print(f"Warning: Field '{field_name}' not found in object instance")

@dataclass
class TissueConfig(Config[torch.nn.Module], ABC):
	"""Base class for tissue configurations"""
	pass

@dataclass
class PhysicsConfig(Config[torch.optim.Optimizer], ABC):
	"""Base class for physics configurations"""

	@override
	def create(self, params) -> torch.optim.Optimizer:
		# Create new instance of optimizer class with params and config dict
		return self.get_class()(params, **self.__dict__)

	@abstractmethod
	def get_class(self) -> Type[torch.optim.Optimizer]:
		"""Get the class of the corresponding optimizer implementation in pytorch"""
		raise NotImplementedError

	@override
	def set(self, physics: torch.optim.Optimizer) -> None:
		"""Set parameters on optimizer param_groups"""
		for group in physics.param_groups:
			for field_name, value in asdict(self).items():
				if field_name in group:
					group[field_name] = value
				else:
					print(f"Warning: Field '{field_name}' not found in optimizer param group")

	@override
	def get(self, physics: torch.optim.Optimizer) -> None:
		"""Read parameters from optimizer param_groups"""
		group = physics.param_groups[0]
		for field_name in self.__dict__:
			if field_name in group:
				setattr(self, field_name, group[field_name])
			else:
				print(f"Warning: Field '{field_name}' not found in optimizer param group")


# ----------------------------------------
# PHYSICS
# ----------------------------------------


class PhysicsType(Enum):
	"""Available physics types for training tissues"""
	ADAM = auto()  # Standard Adam physics with adaptive learning rates
	SGD = auto()  # Basic stochastic gradient descent
	CRONKLE_BISECTION = auto()  # Custom Cronkle physics using bisection search
	# ADAGRAD = auto()  # Adaptive gradient algorithm for sparse data
	# RMSPROP = auto()  # Root mean square propagation
	# ADADELTA = auto()  # Extension of Adagrad that adapts learning rates
	# ADAMW = auto()  # Adam with weight decay regularization
	# NADAM = auto()  # Adam with Nesterov momentum
	# RADAM = auto()  # Rectified Adam with warmup
	# LAMB = auto()  # Layer-wise Adaptive Moments for large batches
	# LION = auto()  # Evolved sign momentum physics
	# ADAFACTOR = auto()  # Memory efficient variant of Adam
	# MADGRAD = auto()  # Momentumized, Adaptive, Dual Averaged Gradient Method
	# SAM = auto()  # Sharpness-Aware Minimization


@dataclass
class SGDPhysics(PhysicsConfig):
	lr: float = 0.01
	momentum: float = 0.5

	@override
	def get_class(self) -> Type[torch.optim.Optimizer]:
		return torch.optim.SGD

@dataclass
class AdamPhysics(PhysicsConfig):
	lr: float = 0.001
	betas: Tuple[float, float] = (0.9, 0.999)

	@override
	def get_class(self) -> Type[torch.optim.Optimizer]:
		return torch.optim.Adam

@dataclass
class CBDPhysics(PhysicsConfig):
	lr: float = 0.937
	momentum: float = 0.9
	noise_scale: float = 0.007
	adam_control: float = 0.896

	@override
	def get_class(self) -> Type[torch.optim.Optimizer]:
		return CBD
	# @override
	# def create(self, params) -> CBD:
	#     return CBD(
	#         params,
	#         lr=self.lr,
	#         momentum=self.momentum,
	#         noise_scale=self.noise_scale
	#     )

# ----------------------------------------
# RUN
# ----------------------------------------
@dataclass
class TrainingStep(ISerializable):
	"""
	A single step in a training schedule, defining what to do during training.
	Used to create complex training regimens by composing multiple steps.
	"""
	# Training parameters
	batch_size: int  # Batch size during this step

	# Optional tissue & physics modifications
	physics_config: PhysicsConfig | None  # Change physics config
	tissue_config: TissueConfig | None  # Change tissue architecture
	freeze_layers: list[str] | None  # Names of layers to freeze
	unfreeze_layers: list[str] | None  # Names of layers to unfreeze

	def __init__(self):
		# Initialize empty scheduler
		self.tissue_config: TissueConfig = None
		self.physics_config: PhysicsConfig = None
		self.batch_size: int = 32
		self.schedule: list[TrainingStep] = []

		# Physics and tissue modifications
		self.physics_type: PhysicsType | None = None
		self.freeze_layers: list[str] | None = None
		self.unfreeze_layers: list[str] | None = None

	def set(self, petri):
		"""Apply this training step's configuration to the tissue and physics"""
		tissue = petri.tissue
		physics = petri.physics

		if self.tissue_config:
			self.tissue_config.set(tissue)

		if not isinstance(physics, self.physics_config.get_class()):
			physics = self.physics_config.create(tissue.parameters())
			petri.physics = physics
		if self.physics_config:
			self.physics_config.set(physics)

		# Freeze/unfreeze layers if specified
		if self.freeze_layers:
			for name, param in tissue.named_parameters():
				if any(layer in name for layer in self.freeze_layers):
					param.requires_grad = False

		if self.unfreeze_layers:
			for name, param in tissue.named_parameters():
				if any(layer in name for layer in self.unfreeze_layers):
					param.requires_grad = True

class RunStats(ISerializable):
	"""Training history that auto-expands arrays when step exceeds current size"""

	def __init__(self, initial_size: int = 0):
		# Basic training metrics
		self.loss = np.zeros(initial_size, dtype=np.float32)  # Training loss over time
		self.grad_norm = np.zeros(initial_size, dtype=np.float32)  # Gradient L2 norm
		self.param_norm = np.zeros(initial_size, dtype=np.float32)  # Parameter L2 norm
		self.learning_rate = np.zeros(initial_size, dtype=np.float32)  # Effective learning rate
		self.batch_time = np.zeros(initial_size, dtype=np.float32)  # Time per batch
		self.memory_used = np.zeros(initial_size, dtype=np.float32)  # GPU memory usage

		# Validation metrics computed periodically
		self.val_loss = np.zeros(initial_size, dtype=np.float32)
		self.val_accuracy = np.zeros(initial_size, dtype=np.float32)

		# Tissue-specific statistics
		self.activation_sparsity = np.zeros(initial_size, dtype=np.float32)
		self.weight_sparsity = np.zeros(initial_size, dtype=np.float32)
		self.layer_gradients = np.zeros((initial_size, 10), dtype=np.float32)  # Per-layer gradient norms
		self.extra: dict[str,np.ndarray] = {}

	def extend(self, size: int):
		"""Extend all arrays by size additional elements"""
		for key, val in self.__dict__.items():
			if isinstance(val, np.ndarray):
				if val.ndim == 1:
					# Extend array by size elements
					existing_size = len(val)
					self.__dict__[key] = np.pad(val, (0, size), mode='constant', constant_values=0)
				elif val.ndim == 2:
					# Extend rows by size elements
					existing_size = len(val)
					self.__dict__[key] = np.pad(val, ((0, size), (0, 0)), mode='constant', constant_values=0)

	def resize(self, size: int):
		"""Resize all arrays to given size, truncating from tail if necessary"""
		for key, val in self.__dict__.items():
			if isinstance(val, np.ndarray):
				if val.ndim == 1:
					self.__dict__[key] = np.resize(val, size)
				elif val.ndim == 2:
					self.__dict__[key] = np.resize(val, (size, val.shape[1]))

@dataclass
class PetriConfig(ISerializable):
	# initialization & scheduling settings
	batch_size: int
	num_steps: int

	def __init__(self):
		# Initialize empty scheduler
		self.tissue: TissueConfig = None
		self.physics: PhysicsConfig = None
		self.batch_size: int = 32
		self.num_steps: int = 5
		self.schedule: list[TrainingStep] = []

	def create_step(self) -> TrainingStep:
		"""Create a new training step with current scheduler configuration"""
		step = TrainingStep()
		step.tissue_config = self.tissue
		step.physics_config = self.physics
		step.batch_size = self.batch_size
		return step

@dataclass
class StepMetrics:
	"""Metrics computed during a single training step"""
	loss: float = 0
	loss_val: float = 0
	grad_norm: float = 0
	param_norm: float = 0
	learning_rate: float = 0
	batch_time: float = 0
	memory_used: float = 0

	def __init__(self, **kwargs):
		"""Initialize metrics with default values and any additional keyword arguments"""
		# Set default values
		self.loss = 0
		self.loss_val = 0
		self.grad_norm = 0
		self.param_norm = 0
		self.learning_rate = 0
		self.batch_time = 0
		self.memory_used = 0

		# Set any additional metrics passed as kwargs
		for key, value in kwargs.items():
			setattr(self, key, value)

	def __setattr__(self, name: str, value: Any) -> None:
		"""Allow setting new attributes dynamically"""
		self.__dict__[name] = value

@dataclass
class CheckpointState:
	i: int = 0
	tissue: nn.Module= None
	physics: optim.Optimizer = None

class Run:
	"""
	A tissue training session which contains its tissue state,
	various checkpoints along the way, its configuration,
	training history, statistics, etc.
	Stored in runs/<run_id>/ which contain <config.dill>, <ckpt_id.ckpt> and <ckpt_name.ckpt>
	History data is history.bin
	"""

	# CONSTRUCTORS
	# ----------------------------------------

	@staticmethod
	def LoadRun(run_id: str) -> "Run":
		"""Load an existing run from disk"""
		run = Run()
		run.run_id = run_id
		run.read()
		run.read_ckpt()
		run.read_stats()
		return run

	@staticmethod
	def NewRun(init_tissue: Optional[TissueConfig], init_physics: Optional[PhysicsConfig]) -> "Run":
		"""Create a new run with the given config"""
		run = Run()
		timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		run.run_id = f"run_{timestamp}"
		run.init_tissue = init_tissue
		run.init_physics = init_physics
		run.tissue = init_tissue.create() if init_tissue else None
		run.physics = init_physics.create(run.tissue.parameters()) if init_physics and run.tissue else None
		logger.info(f"Created new run {run.run_id}")
		return run



	def __init__(self) -> None:
		# State (serialized)
		self.run_id: str = ""
		self.i: int = 0
		self.init_tissue: Optional[TissueConfig] = None
		self.init_physics: Optional[PhysicsConfig] = None
		self.schedule: list[TrainingStep] | None = None
		self.schedule_config: PetriConfig = PetriConfig()
		self.stats = RunStats()
		self.last_step: StepMetrics | None = None

		# # State (live)
		# self.tissue: nn.Module | None = None  # Living tissue state (formerly model)
		# self.physics: optim.Optimizer | None = None  # Physics engine for weight updates

		# Metadata, all read together
		self.checkpoint_indices: list[int] = []
		self.checkpoint_names: list[str] = []
		self.physics_names: list[str] = []

	def create_petri(self):
		"""Create a Petri interface for the tissue if a mapping exists"""
		# Find matching petri mapping from const.petri_mappings based on tissue type
		for mapping in const.petri_mappings:
			if isinstance(self.init_tissue, mapping.config_class):
				return mapping.petri_class(self)

		return None

	def get_serialization_keys(self) -> list[str]:
		"""Get list of attributes to serialize"""
		return [
			'i',
			'init_tissue',
			'init_physics',
			'schedule',
			'schedule_config',
			'checkpoint_indices',
			'checkpoint_names',
			'physics_names'
		]

	# SERIALIZATION
	# ----------------------------------------

	@property
	def serialization_path(self) -> Path:
		"""Path to serialized run state. Only lightweight information."""
		return self.run_dir / "run.dill"

	def write(self) -> None:
		"""Write run configuration and state to disk"""
		state = {}
		for key in self.get_serialization_keys():
			state[key] = getattr(self, key)

		with open(self.serialization_path, 'wb') as f:
			dill.dump(state, f)

		logger.info(f"Wrote run {self.run_id}")

	def read(self) -> None:
		"""Read run configuration and state from disk"""
		with open(self.serialization_path, 'rb') as f:
			state = dill.load(f)
			for key, value in state.items():
				setattr(self, key, value)

		logger.info(f"Read run {self.run_id}")

	def write_ckpt(self, *,
				   tissue,
				   physics,
				   step: Optional[int] = None,
				   name: Optional[str] = None) -> None:
		"""Save a checkpoint of the tissue state, either by step number or name"""

		# Get checkpoint path and update indices/names
		ckpt_path = self.get_checkpoint_path(step if step is not None else name)
		if step is not None and step not in self.checkpoint_indices:
			self.checkpoint_indices.append(step)
		elif name is not None and name not in self.checkpoint_names:
			self.checkpoint_names.append(name)

		if ckpt_path is None:
			raise ValueError("Must specify either step or name for checkpoint")

		# Create checkpoint state
		checkpoint_state = CheckpointState(
			i=self.i,
			tissue=tissue,
			physics=physics
		)

		# Save checkpoint data directly using torch.save
		checkpoint_data = {
			'step': checkpoint_state.i,
			'tissue_state': tissue.state_dict(),
			'physics_state': physics.state_dict()
		}
		torch.save(checkpoint_data, ckpt_path)

		logger.info(f"Wrote checkpoint {self.run_id}")

	def read_ckpt(self, *,
				  step: Optional[int] = None,
				  name: Optional[str] = None) -> CheckpointState:
		"""Load a checkpoint of the tissue state, either by step number or name"""
		ckpt_path = self.get_checkpoint_path(step if step is not None else name)
		if ckpt_path is None:
			raise ValueError("Must specify either step or name for checkpoint")

		# Load checkpoint
		checkpoint = torch.load(ckpt_path)

		# Create checkpoint state and restore
		checkpoint_state = CheckpointState(
			i=checkpoint['step'],
			tissue=checkpoint['tissue'],
			physics=checkpoint['physics']
		)

		logger.info(f"Read checkpoint {self.run_id}")

		return checkpoint_state

	def write_stats(self) -> None:
		"""Write training history to disk"""
		with open(self.stats_path, 'wb') as f:
			dill.dump(self.stats, f)

		logger.info(f"Wrote stats {self.run_id}")

	def read_stats(self) -> None:
		"""Read training history from disk"""
		with open(self.stats_path, 'rb') as f:
			self.stats = dill.load(f)

		def read_checkpoints() -> None:
			"""Read available checkpoint files to populate checkpoint indices and names"""
			# Clear existing lists
			self.checkpoint_indices = []
			self.checkpoint_names = []

			# Get all checkpoint files in run directory
			run_dir = Path(f"runs/{self.run_id}")
			if not run_dir.exists():
				return

			# Check all .ckpt files
			for ckpt_file in run_dir.glob("*.ckpt"):
				# Parse step number from filename (e.g. "step_1000.ckpt")
				if ckpt_file.stem.startswith("step_"):
					try:
						step = int(ckpt_file.stem.split("_")[1])
						self.checkpoint_indices.append(step)
					except (IndexError, ValueError):
						pass
				# Add named checkpoints (e.g. "best.ckpt")
				else:
					self.checkpoint_names.append(ckpt_file.stem)

			# Sort indices numerically
			self.checkpoint_indices.sort()

		def read_physics() -> None:
			"""Read available physics configurations from disk"""
			# Clear existing list
			self.physics_names = []

			# Get all physics files in run directory
			run_dir = Path(f"runs/{self.run_id}")
			if not run_dir.exists():
				return

			# Check all .physics files
			for physics_file in run_dir.glob("*.physics"):
				self.physics_names.append(physics_file.stem)

		with open(self.stats_path, 'rb') as f:
			self.stats = dill.load(f)

		read_checkpoints()
		read_physics()

		logger.info(f"Read stats {self.run_id}")


	# PROPERTIES
	# ----------------------------------------

	@property
	def run_dir(self) -> Path:
		# Ensure runs directory exists
		p1 = Path("runs")
		p2 = Path("runs") / self.run_id
		p1.mkdir(exist_ok=True)
		p2.mkdir(exist_ok=True)
		return p2

	@property
	@property
	def stats_path(self) -> Path:
		"""Path to training history/metrics"""
		return self.run_dir / "stats.bin"

	def get_checkpoint_path(self, step_or_name: Union[int, str]) -> Path:
		"""Get path to checkpoint file for given step number or name"""
		if isinstance(step_or_name, int):
			return self.run_dir / f"{step_or_name}.ckpt"
		else:
			return self.run_dir / f"{step_or_name}.ckpt"

	# region IO functions



	# endregion

# ----------------------------------------
# PETRI
# ----------------------------------------


CT = TypeVar('CT', bound=nn.Module)
from gui_tensor import TensorModality, TensorView

class Petri(ABC, Generic[CT]):
	"""Base class for model training chamber"""

	def __init__(self, run:Run):
		self.run = run
		self.physics: Optional[torch.optim.Optimizer] = None

		# Training
		self.train_loader: Optional[DataLoader] = None
		self.val_loader: Optional[DataLoader] = None
		self._is_training = False
		self._restore_physics: optim.Optimizer = None  # Physics engine to restore after this step
		self._restore_tissue: nn.Module = None  # Tissue config to restore after this step
		self.tensorviews:dict[str, TensorView] = {}
		self.val_dataset: Optional[Dataset] = None
		self.train_dataset: Optional[Dataset] = None

	def view(self, name: str, tensor: torch.Tensor, modality: TensorModality = None):
		"""
		Add a new tensor view to be displayed in the GUI.

		Args:
			name: Unique identifier for this tensor view
			tensor: The tensor data to visualize (numpy array or torch tensor)
			modality: The modality/type of visualization to use
		"""
		import torch

		# Convert torch tensor to numpy if needed
		if isinstance(tensor, torch.Tensor):
			tensor = tensor.detach().cpu().numpy()

		view = TensorView(name, tensor, modality)
		self.tensorviews[name] = view

	def __del__(self):
		"""Clean up texture views"""
		self.tensorviews.clear()
		self.texture_pool = None

	@property
	def is_training(self) -> bool:
		"""Whether the tissue is currently training"""
		return self._is_training

	def init(self, tissue_conf=None, physics_conf=None) -> None:
		"""Initialize or reinitialize tissue (model) with optional config"""
		# Use provided config or initial config
		tissue_config = tissue_conf or self.run.init_tissue
		physics_conf = physics_conf or self.run.init_physics
		if not tissue_config or not physics_conf:
			return

		self.tissue = tissue_config.create()
		self.tissue.config = tissue_config
		self.physics = physics_conf.create(self.tissue.parameters())
		self.physics.config = physics_conf

		self.run.schedule_config.tissue = tissue_config
		self.run.schedule_config.physics = physics_conf

	def load_ckpt(self, step: Optional[int] = None, name: Optional[str] = None) -> None:
		"""Load tissue (model) state from a checkpoint"""
		if not self.tissue:
			return

		# Get checkpoint path
		ckpt_path = self.run.get_checkpoint_path(step if step is not None else name)
		if ckpt_path is None:
			raise ValueError("Must specify either step or name for checkpoint")

		# Load checkpoint
		checkpoint = torch.load(ckpt_path)

		# Restore tissue state
		self.tissue.load_state_dict(checkpoint['tissue_state'])

		# Restore physics state if it exists
		if self.physics and 'physics_state' in checkpoint:
			self.physics.load_state_dict(checkpoint['physics_state'])

	def schedule_step(self, step: TrainingStep=None) -> None:
		"""Schedule a training step"""
		schedule = self.run.schedule_config
		if step is None:
			step = schedule.create_step()

		schedule.schedule.append(step)

	def consume_schedule(self) -> None:
		"""Consume a training step from the schedule"""
		if len(self.run.schedule_config.schedule) == 0:
			return

		step  = self.run.schedule_config.schedule.pop(0)  # TODO: pop from the end
		self.train(step)

	def train(self, step: TrainingStep = None) -> None:
		"""Execute a single training step using the Petri interface"""
		if not self.tissue:
			return

		run = self.run
		if step is None:
			if len(run.schedule_config.schedule) == 0:
				return

			step = run.schedule_config.schedule.pop(0)  # TODO: pop from the end

		# Recreate data loaders if batch size has changed
		if step.batch_size != self.train_loader.batch_size:
			self.train_loader = self.create_training_loader(step.batch_size)
			logger.debug(f"Recreated training loader with batch size {step.batch_size}")
		if step.batch_size != self.val_loader.batch_size:
			self.val_loader = self.create_validation_loader(step.batch_size)
			logger.debug(f"Recreated validation loader with batch size {step.batch_size}")


		# Recreate physics if type has changed
		# current_type = type(self.physics)
		# target_type = const.get_optimizer_for_physics(step.)

		# Run training step through Petri interface
		step.set(self)
		self.tissue.to(const.device)
		self.tensorviews.clear()
		step_metrics = self.train_step(run.i, step)  # Batch handled by Petri
		torch.cuda.empty_cache()
		gc.collect()

		# Record metrics in history
		if step_metrics:
			run.stats.extend(1)
			# Iterate through step metrics and update stats
			for key, val in step_metrics.__dict__.items():
				if hasattr(run.stats, key):
					# If metric exists in RunStats, update array
					getattr(run.stats, key)[-1] = val
				else:
					arr = np.zeros(len(run.stats.loss), dtype=np.float32)
					run.stats.extra[key] = arr

			# Optional validation metrics
			if step_metrics.loss_val is not None:
				run.stats.val_loss[-1] = step_metrics.loss_val
			# TODO investigate with claude if this is needed and what else
			# if metrics.val_accuracy is not None:
			#     self.history.val_accuracy[-1] = metrics.val_accuracy
			#
			# # Tissue-specific metrics
			# self.history.activation_sparsity[-1] = metrics.activation_sparsity
			# self.history.weight_sparsity[-1] = metrics.weight_sparsity
			# if metrics.layer_gradients is not None:
			#     self.history.layer_gradients[-1] = metrics.layer_gradients

		self.run.last_step = step_metrics
		self.run.i += 1
		self.run.write()

	def create_training_loader(self, batch_size: int) -> DataLoader:
		return DataLoader(self.train_dataset,
			batch_size=batch_size,
			shuffle=True,
			num_workers=0
		)

	def create_validation_loader(self, batch_size: int) -> DataLoader:
		return DataLoader(self.val_dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=0
		)

	@abstractmethod
	def train_step(self, i: int, step: TrainingStep) -> StepMetrics:
		"""Single training step"""
		pass

	@abstractmethod
	def validate(self) -> Dict[str, float]:
		"""Run validation"""
		pass

	def gui(self):
		"""Render training controls and visualization"""
		pass

	def inference(self):
		pass

	def train_epoch(self) -> Dict[str, float]:
		# """Run full training epoch"""
		# self.model.train()
		# epoch_metrics: Dict[str, float] = {}
		#
		# for batch in self.train_loader:
		#     metrics = self.train_step(batch)
		#     for k, v in metrics.__dict__.items():
		#         epoch_metrics[k] = epoch_metrics.get(k, 0) + v
		#
		# # Average metrics
		# for k in epoch_metrics:
		#     epoch_metrics[k] /= len(self.train_loader)
		#
		# self.current_epoch += 1
		# self.metrics.update(epoch_metrics)
		# return epoch_metrics

		# TODO probably should be implemented from scratch in each subclass
		raise NotImplementedError

@dataclass
class PetriMapping:
	"""
	Mapping between a model and its Petri visualization/training interface
	for the __petri__ attribute in module file
	"""
	tissue_class: Type[nn.Module]
	config_class: Type[TissueConfig]
	petri_class: Type[Petri]


const.discover_petri_modules()