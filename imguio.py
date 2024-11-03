# disco? imguio?
# ----------------------------------------
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

from imgui_bundle import imgui, imgui_ctx, ImVec2

class IGUI(ABC):
	@abstractmethod
	def gui(self):
		raise NotImplementedError

class IPlottable(ABC):
	@abstractmethod
	def gui(self):
		raise NotImplementedError


def header(label: str, space: bool = True):
	if space:
		imgui.dummy(imgui.ImVec2(0, 10))
	imgui.text(label)
	imgui.separator()

def toggle_button(label: str, state: bool) -> Tuple[bool, bool]:
	c1 = imgui.get_style_color_vec4(imgui.Col_.button.value)
	c2 = imgui.get_style_color_vec4(imgui.Col_.button_active.value)
	color = c2 if state else c1
	with imgui_ctx.push_style_color(imgui.Col_.button.value, color):
		changed = imgui.button(label)
		if changed:
			state = not state
	return changed, state

def imgui_sidebar_layout(width, gui_sidebar, gui_main):
	with imgui_ctx.begin_child("LeftPanel", ImVec2(width, 0), imgui.ChildFlags_.border.value):
		gui_sidebar()

	imgui.same_line()
	with imgui_ctx.begin_child("RightPanel", ImVec2(0, 0)):
		gui_main()

def format_float(value: float) -> str:
	return f"{value:.2f}"

def format_float_gb(value: float) -> str:
	return f"{value:.1f} GB"

def format_int(value: int) -> str:
	return str(value)

def format_str(value: str) -> str:
	return str(value)

@dataclass
class ColumnEntry:
	key: str
	label: str
	format_spec: Union[Callable[[Any], str], str] = None
	min: float = 0
	max: float = 0
	inverted: bool = False


def get_nested_attr(obj: Any, attr_path: str) -> Any:
	"""
	Safely get nested attributes using dot notation.
	Example: get_nested_attr(instance, "stats.gpu_util")
	"""
	attrs = attr_path.split('.')
	value = obj
	for attr in attrs:
		try:
			value = getattr(value, attr)
		except AttributeError:
			return None
	return value

def draw_sortable_table(items: List[Any], columns: List[ColumnEntry], table_name: str, selectable=False, flags=None) -> Optional[Any]:
	selected = None

	if flags is None:
		flags = (
			imgui.TableFlags_.scroll_y.value |
			imgui.TableFlags_.row_bg.value |
			imgui.TableFlags_.borders_outer.value |
			imgui.TableFlags_.borders_v.value |
			imgui.TableFlags_.resizable.value |
			imgui.TableFlags_.reorderable.value |
			imgui.TableFlags_.hideable.value |
			imgui.TableFlags_.sortable.value
		)

	if imgui.begin_table(table_name, len(columns), flags):
		for column in columns:
			imgui.table_setup_column(column.label)
		imgui.table_headers_row()

		sort_specs = imgui.table_get_sort_specs()
		if sort_specs and sort_specs.specs_dirty:
			if sort_specs.specs_count > 0:
				specs = sort_specs.get_specs(0)
				sort_key = columns[specs.column_index].key
				items.sort(
					key=lambda x: get_nested_attr(x, sort_key) or 0,  # Handle None values
					reverse=specs.sort_direction == imgui.SortDirection.descending.value
				)
			sort_specs.specs_dirty = False

		for j, dataobj in enumerate(items):
			imgui.table_next_row()
			for i, column in enumerate(columns):
				value = get_nested_attr(dataobj, column.key)
				if value is None:
					value = 0  # Or another default value appropriate for your use case

				is_percent = column.min != column.max
				percent = value / column.max if column.max > 0 else 0
				if column.inverted:
					percent = 1 - percent

				imgui.table_set_column_index(i)
				if isinstance(column.format_spec, Callable):
					formatted_value = column.format_spec(value)
				elif isinstance(column.format_spec, str):
					formatted_value = column.format_spec.format(value)
				else:
					formatted_value = str(value)

				if selectable and i == 0:
					if imgui.selectable(f"#{j}: {formatted_value}", False, imgui.SelectableFlags_.span_all_columns.value)[0]:
						selected = dataobj
				else:
					if is_percent:
						# white to green
						color = imgui.ImVec4(1 - percent, 1, 1 - percent, 1)  # Transition from white to green
						imgui.text_colored(color, formatted_value)
					else:
						imgui.text(formatted_value)

		imgui.end_table()

	return selected