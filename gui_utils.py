from dataclasses import is_dataclass, fields
from typing import Any, Dict, List, Optional, Set
from imgui_bundle import imgui, imgui_ctx

import gui

def gui_none(name: str, obj: Any, obj_type: Any) -> Any:
    """Render a type selector for None values, allowing instantiation of obj_type or subclasses"""
    # Get all subclasses recursively including the base class
    def get_all_subclasses(cls):
        subclasses = []
        for subclass in cls.__subclasses__():
            subclasses.append(subclass)
            subclasses.extend(get_all_subclasses(subclass))
        return subclasses

    subclasses = [obj_type] + get_all_subclasses(obj_type)

    # Create inline combo box with class options
    current_class = obj_type.__name__
    imgui.text(f"{name}:")
    imgui.same_line()
    if imgui.begin_combo("##type_selector", current_class):
        for cls in subclasses:
            is_selected = current_class == cls.__name__
            if imgui.selectable(cls.__name__, is_selected)[0]:
                # Create new instance of selected class
                obj = cls()
            if is_selected:
                imgui.set_item_default_focus()
        imgui.end_combo()

    return obj

def gui_any(name: str, obj: Any, obj_type: Any = None, collapsible_header: bool = False) -> Any:
    """Render an object with editable fields in ImGui. Returns modified value."""
    if is_dataclass(obj):
        return gui_object(name, obj, collapsible_header)
    elif isinstance(obj, dict):
        result = gui_dict(name, obj)
        return result
    elif isinstance(obj, list):
        result = gui_list(name, obj)
        return result
    elif isinstance(obj, tuple):
        result = gui_tuple(name, obj)
        return result
    elif isinstance(obj, set):
        result = gui_set(name, obj)
        return result
    elif isinstance(obj, (int, float)):
        _, value = imgui.drag_float(f"{name}", obj, 0.001)
        return value
    elif isinstance(obj, str):
        _, value = imgui.input_text(f"{name}", obj, 256)
        return value
    elif isinstance(obj, bool):
        _, value = imgui.checkbox(f"{name}", obj)
        return value
    elif obj_type is not None:
        imgui.pop_id()
        return gui_none(name, obj, obj_type)
    elif obj_type is None:
        imgui.text(f"{name}: {str(obj)}")
        imgui.pop_id()
        return obj
    else:
        imgui.text(f"{name}: {str(obj)}")
        imgui.pop_id()
        return obj

def gui_object(label: str, obj: Any, obj_type: Any = None, fields_to_render: Optional[List[str]] = None, enable: bool = True) -> Any:
    """Render a dataclass object's fields inline in ImGui. Returns modified value.

    Args:
        label: Name label for the object
        obj: The object whose fields should be rendered. Can be None if obj_type provided.
        obj_type: Optional type to use for creating new object if obj is None
        fields_to_render: Optional list of specific field names to display. If None, shows all fields.
        enable: Whether the fields should be editable. Defaults to True.
    Returns:
        Modified object if changes were made
    """
    if not enable:
        imgui.begin_disabled()

    if obj is None and obj_type is not None:
        return gui_none(label, obj, obj_type)
    elif is_dataclass(obj):
        modified = False
        with imgui_ctx.push_id(label):
            for field in fields(obj):
                if fields_to_render is None or field.name in fields_to_render:
                    val = getattr(obj, field.name)
                    newValue = gui_any(field.name, val)
                    setattr(obj, field.name, newValue)
                    # modified = True
            result = obj if modified else obj
    elif isinstance(obj, dict):
        modified = False
        new_dict = obj.copy()
        for key, value in obj.items():
            if fields_to_render is None or key in fields_to_render:
                newValue = gui_any(key, value)
                if newValue != value:
                    new_dict[key] = newValue
                    modified = True
        result = new_dict if modified else obj
    else:
        result = gui_any(label, obj, obj_type)

    if not enable:
        imgui.end_disabled()

    return result

def gui_object_header(name: str, obj: Any, obj_type: Any = None, enable: bool = True) -> Any:
    """Render a dataclass object with collapsing header in ImGui. Returns modified value."""
    with imgui_ctx.push_id(name):
        if obj is None and obj_type is not None:
            label = f"{name} ({obj_type.__name__}: None)"
        else:
            label = f"{name} ({obj.__class__.__name__}: {len(fields(obj))} fields)"

        if imgui.collapsing_header(label):
            gui_object(name, obj, obj_type, enable=enable)

        # Right click popup menu
        if imgui.begin_popup_context_item(f"{name}_header_popup"):
            obj = gui_none(f"Change {name} Type", obj, obj_type)
            imgui.end_popup()
    return obj

def gui_object_node(name: str, obj: Any, obj_type: Any = None, enable: bool = True) -> Any:
    """Render a dataclass object with tree node in ImGui. Returns modified value."""
    with imgui_ctx.push_id(name):
        if obj is None and obj_type is not None:
            label = f"{name} ({obj_type.__name__}: None)"
        else:
            label = f"{name} ({obj.__class__.__name__}: {len(fields(obj))} fields)"

        if imgui.tree_node(label):
            obj = gui_object(name, obj, obj_type, enable=enable)
            imgui.tree_pop()

        # Right click popup menu
        if imgui.begin_popup_context_item(f"{name}_node_popup"):
            obj = gui_none(f"Change {name} Type", obj, obj_type)
            imgui.end_popup()
    return obj


# TUPLE
# ----------------------------------------------------------------------------

def gui_tuple(name: str, t: tuple) -> tuple:
    """Render a tuple with editable items inline. Returns modified tuple."""
    # Special handling for numeric tuples remains unchanged
    if len(t) <= 4 and all(isinstance(x, (int, float)) for x in t):
        if len(t) == 2:
            changed, values = imgui.input_float2(f"{name}##value", t)
        elif len(t) == 3:
            changed, values = imgui.input_float3(f"{name}##value", t)
        elif len(t) == 4:
            changed, values = imgui.input_float4(f"{name}##value",t)
        return tuple(values) if changed else t

    # For non-numeric tuples, render inline
    modified = False
    values = list(t)
    for i, item in enumerate(values):
        imgui.push_id(str(i))
        new_val = gui_any(f"Item {i}:", item)
        if new_val != item:
            values[i] = new_val
            modified = True
        imgui.pop_id()
    return tuple(values) if modified else t

def gui_tuple_node(name: str, t: tuple) -> tuple:
    """Render a tuple with editable items in a tree node. Returns modified tuple."""
    label = f"{name} (Tuple: {len(t)} items)"
    if imgui.tree_node(label):
        result = gui_tuple(name, t)
        imgui.tree_pop()
        return result
    return t

def gui_tuple_header(name: str, t: tuple) -> tuple:
    """Render a tuple with editable items in a collapsing header. Returns modified tuple."""
    label = f"{name} (Tuple: {len(t)} items)"
    if imgui.collapsing_header(label):
        return gui_tuple(name, t)
    return t

# DICT
# ----------------------------------------------------------------------------

def gui_dict(name: str, d: Dict[Any, Any]) -> Dict[Any, Any]:
    """Render a dictionary with editable items inline. Returns modified dict."""
    modified = False
    new_dict = d.copy()
    for key, value in d.items():
        imgui.text(f"{key}:")
        imgui.same_line()
        new_val = gui_any(key, value)
        if new_val != value:
            new_dict[key] = new_val
            modified = True
    return new_dict if modified else d

def gui_dict_node(name: str, d: Dict[Any, Any]) -> Dict[Any, Any]:
    """Render a dictionary with editable items in a tree node. Returns modified dict."""
    label = f"{name} (Dict: {len(d)} entries)"
    if imgui.tree_node(label):
        result = gui_dict(name, d)
        imgui.tree_pop()
        return result
    return d

def gui_dict_header(name: str, d: Dict[Any, Any]) -> Dict[Any, Any]:
    """Render a dictionary with editable items in a collapsing header. Returns modified dict."""
    label = f"{name} (Dict: {len(d)} entries)"
    if imgui.collapsing_header(label):
        return gui_dict(name, d)
    return d

# LIST
# ----------------------------------------------------------------------------

def gui_list(name: str, l: List[Any]) -> List[Any]:
    """Render a list with editable items inline. Returns modified list."""
    modified = False
    operations = []

    # Add new item button
    if imgui.button(f"Add to {name}"):
        operations.append(("add", len(l), None))
        modified = True

    # Display items without outer tree node
    for i, item in enumerate(l):
        imgui.push_id(str(i))

        # Compact controls on one line
        if imgui.button("×"): # Using × instead of X for compactness
            operations.append(("delete", i, None))
            modified = True
        imgui.same_line()
        if i > 0:
            if imgui.arrow_button("up", imgui.Dir.up):
                operations.append(("swap", i, i-1))
                modified = True
            imgui.same_line()
        if i < len(l)-1:
            if imgui.arrow_button("down", imgui.Dir.down):
                operations.append(("swap", i, i+1))
                modified = True
            imgui.same_line()

        new_val = gui_any(f"Item {i}", item)
        if new_val != item:
            operations.append(("edit", i, new_val))
            modified = True

        imgui.pop_id()

    # Apply all operations in reverse order
    if modified:
        for op, idx, val in reversed(operations):
            if op == "swap":
                l[idx], l[val] = l[val], l[idx]
            elif op == "delete":
                l.pop(idx)
            elif op == "add":
                l.append(val if val is not None else "New Item")
            elif op == "edit":
                l[idx] = val

    return l if modified else l

def gui_list_node(name: str, l: List[Any]) -> List[Any]:
    """Render a list with editable items in a tree node. Returns modified list."""
    label = f"{name} (List: {len(l)} items)"
    if imgui.tree_node(label):
        result = gui_list(name, l)
        imgui.tree_pop()
        return result
    return l

def gui_list_header(name: str, l: List[Any]) -> List[Any]:
    """Render a list with editable items in a collapsing header. Returns modified list."""
    label = f"{name} (List: {len(l)} items)"
    if imgui.collapsing_header(label):
        return gui_list(name, l)
    return l

# SET
# ----------------------------------------------------------------------------

def gui_set_node(name: str, s: Set[Any]) -> Set[Any]:
    """Render a set with editable items in a tree node. Returns modified set."""
    label = f"{name} (Set: {len(s)} items)"
    if imgui.tree_node(label):
        result = gui_set(name, s)
        imgui.tree_pop()
        return result
    return s

def gui_set_header(name: str, s: Set[Any]) -> Set[Any]:
    """Render a set with editable items in a collapsing header. Returns modified set."""
    label = f"{name} (Set: {len(s)} items)"
    if imgui.collapsing_header(label):
        return gui_set(name, s)
    return s

def gui_set(name: str, s: Set[Any]) -> Set[Any]:
    """Render a set with editable items. Returns modified set."""
    modified = False
    new_set = set()
    for item in s:
        imgui.indent()
        new_val = gui_any(name, item)
        if new_val != item:
            modified = True
        new_set.add(new_val)
        imgui.unindent()
    return new_set if modified else s

