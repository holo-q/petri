from pathlib import Path

import numpy as np
import OpenGL.GL as gl
from imgui_bundle import hello_imgui, imgui as gui, imgui_ctx, implot, ImVec2
from imgui_bundle.python_backends.opengl_backend import ProgrammablePipelineRenderer
from PyQt6 import QtCore, QtOpenGLWidgets, QtWidgets

import gui_utils
import imguio as guio
from core import AdamPhysics, CBDPhysics, Petri, PhysicsConfig, Run, SGDPhysics, TissueConfig
from models.basinae import BasinTissue
from models.holo import HoloTissue
from models.retnetae import RetNetTissue
from optimizers.dadam import DivineAdamPhysics
from TexturePool import TexturePool

def cached_workvars(**kwargs):
    """
    Decorator that allows attaching static data to a function.
    The data persists between function calls and can be accessed via fn._data.

    Args:
        fn: Function to decorate
        **kwargs: Initial values for the function's static data

    Example:
        @cached_workvars(counter=0, items=[])
        def my_func():
            # Access static data with my_func._data
            my_func._data.counter += 1
            my_func._data.items.append(1)
            return my_func._data
    """
    def decorator(fn):
        # Create a data container for the function with provided initial values
        class DataContainer:
            pass

        fn._data = DataContainer()
        for key, value in kwargs.items():
            setattr(fn._data, key, value)
        return fn

    return decorator

# Nomenclature:
#   tissue: a living growing manifold and mathematical apparatus (formerly called 'model') which can model reality, with exotic emergent capabilities (e.g. neural networks, nn modules, weight matrices, etc.)
#   physics: the weight update mechanics and integration of gradients (formerly called 'optimizer')
#   petri: the editing window into a tissue, which can be used to modify the tissue's parameters and weights (made using imgui)
#   run: a training environment with a petri, model, and physics which can be modified, saved, and loaded efficiently. (a directory in runs/)


# ----------------------------------------
# PETRI WINDOW
# ----------------------------------------

class PlotUI(guio.IGUI):
    """
    Plot panel that displays RunHistory data with configurable value selection
    Uses a horizontal sidebar layout with value toggles on the left
    """

    def __init__(self, run: Run):
        self.run = run

        # Map of history attribute names to display names
        self.value_names = {
            'loss':                'Training Loss',
            'grad_norm':           'Gradient Norm',
            'param_norm':          'Parameter Norm',
            'learning_rate':       'Learning Rate',
            'batch_time':          'Batch Time',
            'memory_used':         'Memory Usage',
            'val_loss':            'Validation Loss',
            'val_accuracy':        'Validation Accuracy',
            'activation_sparsity': 'Activation Sparsity',
            'weight_sparsity':     'Weight Sparsity'
        }

        self._display_i = 0
        self._display_plots = {'loss', 'val_loss'}  # Which history values to plot
        self._display_arrays = []
        self.refresh_plots()

    @property
    def stats(self):
        return self.run.stats

    @property
    def selected_values(self):
        """Currently selected values to plot"""
        return self._display_plots

    def refresh_plots(self):
        """Rebuild display arrays from selected history values"""
        self._display_i = self.run
        self._display_arrays = []
        for attr in self.selected_values:
            # Get values from history and add to display arrays if data exists
            values = self.stats.__getattribute__(attr)
            if values is not None and len(values) > 0:
                self._display_arrays.append(values)

    def gui(self)->'PlotUI':
        """Render plot GUI with sidebar value selection"""
        ret_plot = None

        def gui_left_checkboxes():
            nonlocal ret_plot
            for attr_key, name in self.value_names.items():
                # Handle shift-click to isolate value in new plot
                shift_held = gui.get_io().key_shift
                clicked = gui.selectable(attr_key, attr_key in self.selected_values)[0]

                # SHIFT-CLICK to isolate value in new plot
                if clicked and shift_held:
                    # Create new plot with just this value
                    new_plot = PlotUI(self)
                    new_plot.selected_values.add(attr_key)
                    ret_plot = new_plot
                # CLICK to toggle value in current plot
                elif clicked:
                    # Toggle value in current plot
                    if attr_key in self.selected_values:
                        self.selected_values.remove(attr_key)
                        self.refresh_plots()
                    else:
                        self.selected_values.add(attr_key)
                        self.refresh_plots()

        def gui_right_plot():
            # Get plot height and divide by number of plots (or 1 if empty
            avail_width, avail_height = gui.get_content_region_avail()

            # Create empty plot if no values selected
            if not self._display_arrays:
                if implot.begin_plot("##empty_plot", ImVec2(avail_width, avail_height)):
                    implot.setup_axes("Steps", "Value")
                    implot.end_plot()
            else:
                # Plot all metrics in a single plot
                if implot.begin_plot("##metrics_plot", ImVec2(avail_width, avail_height)):
                    implot.setup_axes("Steps", "Value")
                    for name, values in zip(self._display_plots, self._display_arrays):
                        implot.plot_line(name, values)
                    implot.end_plot()

        if self._display_i != self.run.i:
            self.refresh_plots()

        guio.imgui_sidebar_layout(200, gui_left_checkboxes, gui_right_plot)

        return ret_plot

class RunWindow(guio.IGUI):
    """
    Individual petri simulation run / training adashboard instance.
    """

    def __init__(self, run: Run):
        assert isinstance(run, Run)

        self.run: Run = run
        self.petri: Petri | None = None
        self.plots = [PlotUI(run)]
        self.is_autostepping = False
        self.texture_pool = TexturePool()

    def insert_plot(self, index: int, plot: PlotUI):
        """Insert a plot at a specific index"""
        self.plots.insert(index, plot)

    def gui(self):
        """Render Petri window with sidebar layout"""
        if not gui.begin(f"Petri - {self.run.run_id}", True):
            gui.end()
            return False

        self.petri.consume_schedule()  # TODO move to a main thread so the GUI doesn't freeze

        # Tab bar
        if gui.begin_tab_bar("PetriTabs"):
            # Training tab
            if gui.begin_tab_item("Training")[0]:
                guio.imgui_sidebar_layout(300, self.run_left_dash, self.gui_right_viz)
                gui.end_tab_item()

            # Inference tab
            if gui.begin_tab_item("Tensors")[0]:
                # Render petri outputs based on type
                # Assumes petri.output_type and petri.output_data are available
                if gui.begin_child("inference_view", ImVec2(0, 0)):
                    # Get available width/height for display
                    avail_width = gui.get_content_region_avail().x
                    avail_height = gui.get_content_region_avail().y
                    tensor_height = avail_height / len(self.petri.tensorviews) if len(self.petri.tensorviews) > 0 else 0

                    # Render output views
                    for id,view in self.petri.tensorviews.items():
                        if gui.begin_child(f"view_{id}", ImVec2(0, tensor_height)):
                            view.gui()
                        gui.end_child()

                    if len(self.petri.tensorviews) == 0:
                        gui.text("No tensors.")

                gui.end_child()
                gui.end_tab_item()

            gui.end_tab_bar()

        gui.end()

        # Save checkpoint by name popup
        if gui.begin_popup_modal("SaveCheckpointByName", False)[1]:
            gui.text("Name:")
            gui.same_line()
            name = gui.input_text("", self.run.run_id, 256)[1]
            if gui.button("Save"):
                self.run.write_ckpt(name=name,
                    tissue=self.petri.tissue,
                    physics=self.petri.physics)
            if gui.button("Cancel"):
                gui.close_current_popup()
            gui.end_popup()

        if self.is_autostepping and len(self.run.schedule_config.schedule) == 0:
            self.petri.schedule_step()

        return True

    def run_left_dash(self):
        # RUN CONFIGURATION ----------------------------------------
        # Tissue configuration header
        if not self.petri.tissue:
            self.run.init_tissue = gui_utils.gui_object_header("Tissue Config", self.run.schedule_config.tissue, TissueConfig)
        else:
            self.run.schedule_config.tissue = gui_utils.gui_object_header("Tissue Config", self.run.schedule_config.tissue, TissueConfig)

        # Physics configuration header
        if not self.petri.tissue:
            self.run.init_physics = gui_utils.gui_object_header("Physics Config", self.run.init_tissue, PhysicsConfig)
        else:
            self.run.schedule_config.physics = gui_utils.gui_object_header("Physics Config", self.run.schedule_config.physics, PhysicsConfig)

        # TRAINING ----------------------------------------
        guio.header(f"Petri")

        if self.petri.tissue is None:
            if gui.button("Initialize"):
                self.petri = self.run.create_petri()
                self.run.schedule_config.tissue = self.run.init_tissue
                self.run.schedule_config.physics = self.run.init_physics
        else:
            gui.text("step: ")
            gui.same_line()
            gui.text(f"{self.run.i}")

            gui.text("epoch: ")
            gui.same_line()
            gui.text(f"n/a")

            gui.begin_disabled(self.run.last_step is None)
            gui.text("loss: ")
            gui.same_line()
            gui.text(f"{self.run.last_step.loss if self.run.last_step else 'n/a'}")

            gui.text("val loss: ")
            gui.same_line()
            gui.text(f"{self.run.last_step.loss_val if self.run.last_step else 'n/a'}")
            gui.end_disabled()

            if gui.button("Start" if not self.is_autostepping else "Stop"):
                self.is_autostepping = not self.is_autostepping
            gui.same_line()
            if gui.button("Step"):
                for _ in range(int(self.run.schedule_config.num_steps)):
                    self.petri.schedule_step()
            gui.same_line()
            _, self.run.schedule_config.num_steps = gui.slider_int("##num_steps", self.run.schedule_config.num_steps, 1, 100)

        # CHECKPOINTS ----------------------------------------
        guio.header(f"Checkpoints Management")
        gui.begin_disabled(self.petri.tissue is None)

        gui.text("Indices: ")
        gui.same_line()
        gui.text(f"{len(self.run.checkpoint_indices)}")

        gui.text("Names: ")
        gui.same_line()
        gui.text(f"{len(self.run.checkpoint_names)}")

        if gui.button(f"Save {self.run.i}##save_step"):
            self.run.write_ckpt(
                step=self.run.i,
                tissue=self.petri.tissue,
                physics=self.petri.physics)
        gui.same_line()
        checkpoints = self.run.checkpoint_indices
        chg, selected_step = gui.combo("step##load_checkpoint_by_step", -1, [str(i) for i in checkpoints])
        if chg and selected_step >= 0:  # If a step was selected
            self.run.read_ckpt(step=checkpoints[selected_step])

        if gui.button("Save##save_name"):
            gui.open_popup("SaveCheckpointByName")

        gui.same_line()
        chg, sel_name = gui.combo("name##load_checkpoint_by_name", -1, [str(i) for i in self.run.checkpoint_names])
        if chg and sel_name >= 0:  # If a name was selected
            name = self.run.checkpoint_names[sel_name]
            self.run.read_ckpt(name=name)

        gui.end_disabled()

    @cached_workvars(guis=[])
    def gui_right_viz(self):
        # Get shift key state
        shift_held = gui.get_io().key_shift

        guis = self.gui_right_viz._data.guis
        guis.clear()
        guis.extend(self.plots)
        guis.extend(self.petri.tensorviews.values())  # TODO only output & input tensors

        # Calculate height for each plot
        HEADER_HEIGHT = 28
        IMGUI_SPACING = gui.get_style().item_spacing.y

        available_height = gui.get_content_region_avail()[1]
        num_guis = len(guis)
        gui_height = 0
        if num_guis > 0:
            # Reserve space for "Add Plot Bottom" button if shift held
            if len(self.plots) > 0: available_height -= HEADER_HEIGHT
            if len(self.petri.tensorviews) > 0: available_height -= HEADER_HEIGHT

            if shift_held:
                available_height -= 30  # Approximate button height
            gui_height = available_height / num_guis - IMGUI_SPACING

        # ----------------------------------------

        guio.header("Plots", False)

        if not self.plots:
            if shift_held and gui.button("Add Plot##add_plot"):
                self.plots.append(PlotUI(self.run))
            return

        if shift_held and gui.button("+##add_plot_above", ImVec2(-1, 0)):
            self.insert_plot(0, PlotUI(self.run))

        # Render each plot in an evenly sized child window
        removed_plot = None
        for i, plot in enumerate(self.plots):
            with imgui_ctx.push_id(i):
                with imgui_ctx.begin_group():
                    # X button that spans the plot height (only when shift held)
                    if shift_held:
                        if gui.button("X", ImVec2(20, gui_height if num_guis > 0 else 0)):
                            removed_plot = plot
                        gui.same_line()

                    # Plot content
                    if gui.begin_child(f"plot", ImVec2(0, gui_height if num_guis > 0 else 0)):
                        new_plot = plot.gui()
                        if new_plot:
                            self.insert_plot(i + 1, new_plot)
                    gui.end_child()

        if self.petri.tensorviews:
            guio.header("Tensors")
            for i, (label, tensorview) in enumerate(self.petri.tensorviews.items()):
                with imgui_ctx.push_id(i):
                    if gui.begin_child(f"tensorview", ImVec2(0, gui_height if num_guis > 0 else 0)):
                        gui.text(label)
                        tensorview.initialize(self.texture_pool)
                        tensorview.gui()
                    gui.end_child()

        if removed_plot:
            self.plots.remove(removed_plot)

        if shift_held and gui.button("+##add_plot_below", ImVec2(-1, 0)):
            self.plots.append(PlotUI(self.run))


# ----------------------------------------
# PETRI APP
# ----------------------------------------

class PetriApp:
    """Main application managing multiple Petri windows"""

    def __init__(self):
        self.petris: dict[str, RunWindow] = {}
        self.run_names = self.get_available_runs()
        self.maximizing_window = None

        from models.autoencodermodern import AutoencoderTissue
        wnd = self.on_btn_new_run()
        wnd.run.init_tissue = AutoencoderTissue()
        wnd.run.init_physics = AdamPhysics()
        wnd.petri = wnd.run.create_petri()
        wnd.petri.init()
        pass


    def get_available_runs(self) -> list[str]:
        """Get list of available run names from the runs/ directory"""
        runs_dir = Path("runs")
        if not runs_dir.exists():
            return []

        # Get all directories in runs/ that contain a config.json file
        # These represent valid run directories
        return [
            d.name for d in runs_dir.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        ]

    def gui_init(self):
        hello_imgui.apply_theme(hello_imgui.ImGuiTheme_.photoshop_style)

    def gui(self):
        """Render main menu bar and all Petri windows"""
        if gui.begin_main_menu_bar():
            if gui.button("new"):
                self.on_btn_new_run()
            gui.text(f"Runs: {len(self.petris)}")
            gui.set_next_item_width(200)
            if gui.begin_combo("##load_run", ""):
                for run_name in self.run_names:
                    is_selected = gui.selectable(run_name, False)[0]
                    if is_selected:
                        self.on_load_petri_name(run_name)
                gui.end_combo()
            gui.end_main_menu_bar()

        # Render all windows
        to_close = []
        for run_id, window in self.petris.items():
            # If this is the maximizing window, set size to fill available space
            if window == self.maximizing_window:
                # Get available screen space accounting for menu bar
                viewport_size = gui.get_main_viewport().size
                menu_height = gui.get_frame_height()
                gui.set_next_window_size(ImVec2(viewport_size.x, viewport_size.y - menu_height))
                gui.set_next_window_pos(ImVec2(0, menu_height))
                self.maximizing_window = None

            if not window.gui():
                to_close.append(run_id)

        # Clean up closed windows
        for run_id in to_close:
            del self.petris[run_id]

    def add_petri(self, run: Run) -> RunWindow:
        """Add a new RunWindow for the given Run"""
        wnd = RunWindow(run)
        self.petris[run.run_id] = wnd
        if len(self.petris) == 1:
            self.maximizing_window = wnd
        return wnd

    def on_btn_new_run(self):
        # Create a new run with minimal config, tissue will be configured in GUI
        tissue_config = None
        physics_config = AdamPhysics()
        run = Run.NewRun(tissue_config, physics_config)
        wnd = self.add_petri(run)
        return wnd

    def on_load_petri_name(self, run_name):
        run = Run.LoadRun(run_name)
        self.petris[run.run_id] = RunWindow(run)

# ----------------------------------------
# PYQT RENDERER
# ----------------------------------------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.app = PetriApp()
        self.renderer = PyQtRenderer(self.app)
        self.renderer.app = self.app
        self.setCentralWidget(self.renderer.widget)
        self.resize(1280, 720)
        self.setWindowTitle("Petri")

class PyQtRenderer:
    def __init__(self, app: PetriApp, parent=None):
        self.widget = QtOpenGLWidgets.QOpenGLWidget(parent)
        self.imgui_renderer = None
        self.time = QtCore.QElapsedTimer()
        self.time.start()
        self.last_frame_time = None
        self.app: PetriApp = app

        # Configure widget
        self.widget.setMouseTracking(True)
        self.widget.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.widget.initializeGL = self.initializeGL
        self.widget.paintGL = self.paintGL
        self.widget.resizeGL = self.resizeGL
        self.widget.installEventFilter(self.widget)
        self.widget.eventFilter = self.eventFilter

        self.gl_initialized = False

    def initializeGL(self):
        """Initialize OpenGL context and ImGui"""
        self.gl_initialized = True

        # Qt will call this when the context is ready
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)

        # Initialize ImGui
        gui.create_context()
        implot.create_context()
        self.app.gui_init()

        # Initialize renderer with ProgrammablePipeline
        self.imgui_renderer = ProgrammablePipelineRenderer()

        # Configure IO
        io = gui.get_io()
        io.display_size = self.widget.width(), self.widget.height()
        io.config_flags |= gui.ConfigFlags_.nav_enable_keyboard
        io.config_flags |= gui.ConfigFlags_.docking_enable

        # Set backend flags
        io.backend_flags |= gui.BackendFlags_.has_mouse_cursors
        io.backend_flags |= gui.BackendFlags_.has_set_mouse_pos

    def resizeGL(self, width: int, height: int):
        """Handle window resize events"""
        io = gui.get_io()
        io.display_size = width, height
        gl.glViewport(0, 0, width, height)

    def paintGL(self):
        """Main render loop"""
        # Clear the frame
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Update time delta
        current_time = self.time.elapsed() / 1000.0
        if self.last_frame_time is not None:
            gui.get_io().delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        # Start new ImGui frame
        gui.new_frame()

        # Render all UI components
        self.app.gui()

        # End frame and render
        gui.end_frame()
        gui.render()
        self.imgui_renderer.render(gui.get_draw_data())

        # Request next frame
        self.widget.update()

    def eventFilter(self, watched, event) -> bool:
        if not self.gl_initialized:
            return False

        if watched == self.widget:
            io = gui.get_io()

            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    io.mouse_down[0] = True
                elif event.button() == QtCore.Qt.MouseButton.RightButton:
                    io.mouse_down[1] = True
                elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
                    io.mouse_down[2] = True
                return True

            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    io.mouse_down[0] = False
                elif event.button() == QtCore.Qt.MouseButton.RightButton:
                    io.mouse_down[1] = False
                elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
                    io.mouse_down[2] = False
                return True

            elif event.type() == QtCore.QEvent.Type.MouseMove:
                pos = event.position()
                io.mouse_pos = pos.x(), pos.y() - 5
                return True

            elif event.type() == QtCore.QEvent.Type.Wheel:
                io.mouse_wheel = event.angleDelta().y() / 120.0
                return True

            elif event.type() == QtCore.QEvent.Type.KeyPress:
                return True

            elif event.type() == QtCore.QEvent.Type.KeyRelease:
                return True

        return super(self.widget.__class__, self.widget).eventFilter(watched, event)

    def create_texture(self, width: int, height: int, channels: int = 4) -> int:
        """Create an OpenGL texture"""
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        if channels == 4:
            gl_format = gl.GL_RGBA
        else:
            gl_format = gl.GL_RGB

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl_format, width, height, 0,
            gl_format, gl.GL_UNSIGNED_BYTE, None
        )
        return texture_id

    def update_texture(self, texture_id: int, data: np.ndarray):
        """Update texture data"""
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D, 0, 0, 0,
            data.shape[1], data.shape[0],  # width, height
            gl.GL_RGBA if data.shape[2] == 4 else gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            data.tobytes()
        )

def main():
    import sys

    # Create Qt Application
    app = QtWidgets.QApplication(sys.argv)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
