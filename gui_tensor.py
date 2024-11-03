import json
from enum import auto, Enum
from typing import Union

import cv2
import numpy as np
from imgui_bundle import imgui as gui, implot, ImVec2, ImVec4

from core import Run
from imguio import IGUI, IPlottable
from TexturePool import TexturePool

TensorLike = Union[np.ndarray, "torch.Tensor"]  # type: ignore

class TensorModality(Enum):
    SCALAR = auto()            # float
    RGB_IMAGE = auto()         # [H,W,3] torch tensor
    CHW_IMAGE_GRAYSCALE = auto()   # [H,W,1] torch tensor
    BCHW_IMAGE_GRAYSCALE = auto()   # [H,W,1] torch tensor
    HW_SEGMENTATION_MASK = auto()  # [H,W] torch tensor with class indices
    HWC_FLOW_FIELD = auto()        # [H,W,2] flow vectors
    HWC_DEPTH_MAP = auto()          # [H,W] depth values
    VQGAN_LATENTS = auto()     # [num_tokens, latent_dim]
    ATTENTION_HEATMAP = auto()  # [H,W] attention weights
    CLASS_PROBABILITIES = auto() # [num_classes] probability distribution
    EMBEDDINGS = auto()         # [num_points, 2] for 2D embeddings
    LOGITS = auto()            # [vocab_size] raw logits
    TEXT = auto()              # String of generated text
    TOKEN_PROBABILITIES = auto() # [num_tokens, vocab_size] probability distribution
    ATTENTION_PATTERNS = auto()  # [num_heads, seq_len, seq_len] attention weights
    POINT_CLOUD = auto()        # [num_points, 3] 3D point coordinates
    NORMAL_MAP = auto()         # [H,W,3] surface normal vectors
    SEMANTIC_MAP = auto()       # [H,W,C] multi-class semantic segmentation
    INSTANCE_MAP = auto()       # [H,W] instance segmentation IDs
    KEYPOINTS = auto()          # [num_keypoints, 2] 2D keypoint coordinates
    POSE = auto()              # [num_joints, 3] 3D joint positions
    BOUNDING_BOXES = auto()     # [num_boxes, 4] box coordinates (x1,y1,x2,y2)
    AUDIO_WAVEFORM = auto()     # [samples, channels] raw audio samples
    SPECTROGRAM = auto()        # [freq_bins, time_steps] audio spectrogram
    FEATURE_VECTORS = auto()    # [batch_size, feature_dim] generic feature embeddings
    GRAPH = auto()             # Graph structure with nodes and edges
    VOXELS = auto()            # [D,H,W] 3D voxel grid
    MESH = auto()              # 3D mesh vertices and faces
    CAMERA_PARAMS = auto()      # Camera intrinsics/extrinsics
    OPTICAL_FLOW = auto()       # [H,W,2] optical flow vectors
    DISPARITY = auto()         # [H,W] stereo disparity map
    SURFACE_NORMALS = auto()    # [H,W,3] surface normal vectors
    MATERIAL_PARAMS = auto()    # [H,W,C] material properties (roughness, metallic etc)
    LANGUAGE_EMBEDDINGS = auto() # [seq_len, embed_dim] text embeddings
    LATENT_CODES = auto()       # [batch_size, latent_dim] latent space encodings
    STYLE_VECTORS = auto()      # [batch_size, style_dim] style transfer embeddings
    MASKS = auto()              # [H,W] binary masks
    CONFIDENCE_SCORES = auto()  # [batch_size] confidence/uncertainty values
    JOINT_ANGLES = auto()       # [num_joints] robot joint configurations
    TACTILE_READINGS = auto()   # [H,W,C] tactile sensor data
    IMU_DATA = auto()          # [timesteps, 6] IMU acceleration/gyro readings
    LIDAR_SCAN = auto()        # [num_points, 3] LiDAR point cloud
    RADAR_DATA = auto()        # [H,W] radar intensity readings
    THERMAL_IMAGE = auto()      # [H,W] thermal camera readings
    EVENT_STREAM = auto()       # [num_events, 4] neuromorphic camera events
    FORCE_TORQUE = auto()      # [6] force/torque sensor readings
    OCCUPANCY_GRID = auto()     # [H,W] 2D occupancy map
    SIGNED_DISTANCE = auto()    # [H,W] signed distance field
    TRAJECTORY = auto()         # [timesteps, dims] motion trajectory
    ACTION_MASK = auto()        # [num_actions] valid action mask for RL
    REWARD_SIGNAL = auto()      # Scalar reward value for RL
    STATE_VECTOR = auto()       # [state_dim] system state representation

class TensorView(IGUI, IPlottable):
    """Handles visualization of tensor outputs based on modality"""

    def __init__(self, id: str, tensor: TensorLike, modality: TensorModality):
        self.id = id
        self.tensor: TensorLike = tensor
        self.modality: TensorModality = modality
        self.scale: float = 3.0
        self.vector_spacing: int = 20  # For flow field display
        self.vector_scale: float = 1.0

        self._initialized = False
        self._texture_pool: TexturePool = TexturePool()
        self._texids: list[int] = []  # List of texture IDs for complex n-dimensional tensors that can decompose in many different ways
        self._texture_sizes: list[ImVec2] = []  # List of texture sizes for complex n-dimensional tensors that can decompose in many different ways

    def __del__(self):
        """Release texture when destroyed"""
        for texid in self._texids:
            self._texture_pool.release(texid)

    def _filter_config_keys(self) -> dict:
        """Filter configuration keys to only include valid settings"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
               and k not in ['tensor', 'modality', 'id']
        }

    def write(self, run: Run):
        """
        Write tensor to disk for a recorded history of this view,
        stored in runs/<run_id>/tensors/<id>/<i>.bin
        """
        # Create tensors directory if needed
        tensor_dir = run.run_dir / "tensors" / self.id
        tensor_dir.mkdir(parents=True, exist_ok=True)

        # Save tensor data
        tensor_path = tensor_dir / f"{run.i}.bin"
        with open(tensor_path, "wb") as f:
            np.save(f, self.tensor)

        # Save metadata
        meta_path = tensor_dir / f"{run.i}.json"
        meta = {
            "modality": self.modality.name,
            "config":   self._filter_config_keys()
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def read(self, run: Run):
        """Read tensor from disk for a recorded history of this view"""
        tensor_path = run.run_dir / "tensors" / self.id / f"{run.i}.bin"
        meta_path = run.run_dir / "tensors" / self.id / f"{run.i}.json"

        if tensor_path.exists() and meta_path.exists():
            # Load tensor data
            with open(tensor_path, "rb") as f:
                self.tensor = np.load(f)

            # Load metadata
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.modality = TensorModality[meta["modality"]]
                # Update config settings
                for k, v in meta["config"].items():
                    setattr(self, k, v)

    def _get_rgb_visualization(self) -> np.ndarray:
        """Convert tensor to RGB visualization based on modality"""
        # Convert to numpy if needed
        import torch
        tensor = self.tensor
        if isinstance(self.tensor, torch.Tensor):
            tensor = self.tensor.detach().cpu().numpy()

        # Convert to RGB based on modality
        match self.modality:
            case TensorModality.RGB_IMAGE:
                return tensor

            case TensorModality.BCHW_IMAGE_GRAYSCALE:
                # Convert BCHW grayscale to RGB by repeating channel 3 times to HWC
                # tensor shape: (batch, channels=1, height, width)
                return np.repeat(tensor[0], 3, axis=0).transpose(0,2,1)

            case TensorModality.CHW_IMAGE_GRAYSCALE:
                # Fix: Handle (1,H,W) shape properly and ensure contiguous array
                if tensor.shape[0] == 1:  # Single channel case
                    tensor = np.ascontiguousarray(tensor[0])  # Remove channel dim to get (H,W)
                
                # Normalize and convert to RGB
                normalized = (tensor - tensor.min()) / (tensor.max() - tensor.min())
                return np.ascontiguousarray(np.stack([normalized] * 3, axis=-1))  # (H,W) -> (H,W,3)

            case TensorModality.HW_SEGMENTATION_MASK:
                return cv2.applyColorMap(
                    (tensor * 255 / tensor.max()).astype(np.uint8),
                    cv2.COLORMAP_JET
                )

            case TensorModality.HWC_FLOW_FIELD:
                mag, ang = cv2.cartToPolar(tensor[..., 0], tensor[..., 1])
                hsv = np.zeros((*tensor.shape[:2], 3), dtype=np.uint8)
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 1] = 255
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            case TensorModality.HWC_DEPTH_MAP | TensorModality.DISPARITY:
                return cv2.applyColorMap(
                    (tensor * 255 / tensor.max()).astype(np.uint8),
                    cv2.COLORMAP_TURBO
                )

            case TensorModality.NORMAL_MAP | TensorModality.SURFACE_NORMALS:
                return ((tensor + 1) * 127.5).astype(np.uint8)

            case TensorModality.SEMANTIC_MAP:
                num_classes = tensor.shape[-1]
                colors = np.random.randint(0, 255, (num_classes, 3))
                rgb = np.zeros((*tensor.shape[:2], 3))
                for i in range(num_classes):
                    rgb[tensor[..., i] > 0.5] = colors[i]
                return rgb.astype(np.uint8)

            case TensorModality.INSTANCE_MAP:
                instances = np.unique(tensor)
                colors = np.random.randint(0, 255, (len(instances), 3))
                rgb = np.zeros((*tensor.shape, 3))
                for i, inst in enumerate(instances):
                    rgb[tensor == inst] = colors[i]
                return rgb.astype(np.uint8)

            case _:
                return np.zeros((64, 64, 3), dtype=np.uint8)

    def initialize(self, texture_pool: TexturePool):
        """Initialize texture from tensor visualization using the texture pool"""
        if self._initialized: return
        self._initialized = True
        self._texture_pool = texture_pool

        # Get RGB visualization
        rgb_data = self._get_rgb_visualization()

        # Get texture from pool and update it
        texid = texture_pool.rent()
        self._texids.append(texid)
        self._texture_sizes.append(ImVec2(rgb_data.shape[1], rgb_data.shape[0]))
        texture_pool.update_texture(texid, rgb_data)

    def gui(self):
        """Render tensor visualization with ImGui"""
        if not self._initialized:
            return

        tensor = self.tensor
        modality = self.modality
        avail = gui.get_content_region_avail()

        # Render based on modality
        if modality == TensorModality.HWC_FLOW_FIELD:
            # Show flow field visualization in ImPlot
            if implot.begin_plot(f"{self.id}##flow_plot"):
                # Controls
                gui.same_line()
                gui.begin_group()
                _, self.vector_spacing = gui.slider_int("Grid Spacing", self.vector_spacing, 5, 50)
                _, self.vector_scale = gui.slider_float("Vector Scale", self.vector_scale, 0.1, 5.0)
                gui.end_group()

                # Draw vector field
                h, w = tensor.shape[:2]
                draw_list = implot.get_plot_draw_list()
                implot.push_plot_clip_rect()

                for y in range(0, h, self.vector_spacing):
                    for x in range(0, w, self.vector_spacing):
                        flow = tensor[y, x]
                        # Convert grid coordinates to plot space
                        start = implot.plot_to_pixels(implot.Point(x, y))
                        end = implot.plot_to_pixels(implot.Point(
                            x + flow[0] * self.vector_scale,
                            y + flow[1] * self.vector_scale
                        ))

                        draw_list.add_line(
                            ImVec2(start.x, start.y),
                            ImVec2(end.x, end.y),
                            gui.get_color_u32(ImVec4(1, 1, 1, 0.8)),
                            1.0
                        )

                implot.pop_plot_clip_rect()
                implot.end_plot()

        elif modality == TensorModality.TEXT:
            gui.text_wrapped(str(tensor))

        else:
            # Default image display
            for i, texid in enumerate(self._texids):
                gui.image(texid, ImVec2(self._texture_sizes[i].x*self.scale, self._texture_sizes[i].y*self.scale))


    def plot(self, label: str, tensor, modality: TensorModality):
        """
        Plot tensor data within an existing ImPlot visualization.
        Assumes implot.begin_plot() has already been called.
        """
        # Skip if no data to plot
        if not self._initialized:
            return

        if tensor is None:
            return

        # Handle different modalities
        if modality == TensorModality.SCALAR:
            # Plot scalar value as line
            if len(tensor.shape) == 1:
                implot.plot_line(f"{label}##plot", tensor)
            elif len(tensor.shape) == 2:
                # Plot each row as separate line
                for i in range(tensor.shape[0]):
                    implot.plot_line(f"{label}_{i}##plot", tensor[i, :])

        elif modality == TensorModality.HWC_FLOW_FIELD:
            # Plot flow field vectors
            h, w = tensor.shape[:2]
            draw_list = implot.get_plot_draw_list()
            implot.push_plot_clip_rect()

            # Draw each vector
            for y in range(0, h, self.vector_spacing):
                for x in range(0, w, self.vector_spacing):
                    flow = tensor[y, x]
                    start = implot.plot_to_pixels(implot.Point(x, y))
                    end = implot.plot_to_pixels(implot.Point(
                        x + flow[0] * self.vector_scale,
                        y + flow[1] * self.vector_scale
                    ))

                    draw_list.add_line(
                        ImVec2(start.x, start.y),
                        ImVec2(end.x, end.y),
                        gui.get_color_u32(ImVec4(1, 1, 1, 0.8)),
                        1.0
                    )

            implot.pop_plot_clip_rect()
        elif modality == TensorModality.TEXT:
            pass  # text cannot be plotted
        else:
            # Default to heatmap for 2D data
            if len(tensor.shape) == 2:
                implot.plot_heatmap(f"{label}##heatmap", tensor)
            else:
                gui.text_wrapped(f"Unsupported tensor shape for plotting: {tensor.shape}")

