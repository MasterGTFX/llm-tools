"""Comprehensive tests for convert_functions_to_map function."""

import pytest
from enum import Enum
from typing import Any, Optional, Union, Literal, List, Dict, Tuple

from llmtools.interfaces.openai_llm import convert_functions_to_map


# Test Enums
class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class Temperature(Enum):
    COLD = -10.5
    WARM = 20.0
    HOT = 35.5


# Dummy functions for testing (20+ examples with different parameter combinations)

def func_no_params():
    """Function with no parameters."""
    return "no params"


def func_basic_str(name: str):
    """Function with string parameter.
    
    Args:
        name: The name to process
    """
    return f"Hello {name}"


def func_basic_int(count: int):
    """Function with integer parameter.
    
    Args:
        count: Number of items
    """
    return count * 2


def func_basic_float(price: float):
    """Function with float parameter.
    
    Args:
        price: The price value
    """
    return price * 1.1


def func_basic_bool(active: bool):
    """Function with boolean parameter.
    
    Args:
        active: Whether the item is active
    """
    return not active


def func_optional_str(message: Optional[str]):
    """Function with optional string parameter.
    
    Args:
        message: Optional message to display
    """
    return message or "default"


def func_union_types(value: Union[str, int]):
    """Function with union parameter.
    
    Args:
        value: Either a string or integer value
    """
    return str(value)


def func_list_str(items: list[str]):
    """Function with list of strings parameter.
    
    Args:
        items: List of string items
    """
    return ", ".join(items)


def func_list_int(numbers: list[int]):
    """Function with list of integers parameter.
    
    Args:
        numbers: List of integer values
    """
    return sum(numbers)


def func_list_untyped(items: list):
    """Function with untyped list parameter.
    
    Args:
        items: List of items (any type)
    """
    return len(items)


def func_dict_typed(data: dict[str, int]):
    """Function with typed dictionary parameter.
    
    Args:
        data: Dictionary mapping strings to integers
    """
    return sum(data.values())


def func_dict_untyped(data: dict):
    """Function with untyped dictionary parameter.
    
    Args:
        data: Dictionary of any key-value pairs
    """
    return len(data)


def func_tuple_homogeneous(coords: tuple[int, int]):
    """Function with homogeneous tuple parameter.
    
    Args:
        coords: Tuple of two integers representing coordinates
    """
    return coords[0] + coords[1]


def func_tuple_heterogeneous(info: tuple[str, int, bool]):
    """Function with heterogeneous tuple parameter.
    
    Args:
        info: Tuple containing name, age, and active status
    """
    return f"{info[0]}: {info[1]}, active: {info[2]}"


def func_tuple_untyped(data: tuple):
    """Function with untyped tuple parameter.
    
    Args:
        data: Tuple of any values
    """
    return len(data)


def func_enum_str(color: Color):
    """Function with string enum parameter.
    
    Args:
        color: The color selection
    """
    return f"Selected color: {color.value}"


def func_enum_int(priority: Priority):
    """Function with integer enum parameter.
    
    Args:
        priority: The priority level
    """
    return f"Priority level: {priority.value}"


def func_enum_float(temp: Temperature):
    """Function with float enum parameter.
    
    Args:
        temp: The temperature setting
    """
    return f"Temperature: {temp.value}°C"


def func_literal_str(mode: Literal["read", "write", "append"]):
    """Function with string literal parameter.
    
    Args:
        mode: File operation mode
    """
    return f"Mode: {mode}"


def func_literal_int(level: Literal[1, 2, 3, 4, 5]):
    """Function with integer literal parameter.
    
    Args:
        level: The difficulty level
    """
    return f"Level: {level}"


def func_literal_mixed(status: Literal["active", "inactive", 1, 0]):
    """Function with mixed literal parameter.
    
    Args:
        status: The status (string or numeric)
    """
    return f"Status: {status}"


def func_multiple_basic(name: str, age: int, active: bool):
    """Function with multiple basic parameters.
    
    Args:
        name: Person's name
        age: Person's age
        active: Whether person is active
    """
    return f"{name}, {age}, {active}"


def func_optional_with_default(message: str, count: int = 1):
    """Function with optional parameter having default value.
    
    Args:
        message: The message to repeat
        count: Number of times to repeat (default: 1)
    """
    return message * count


def func_mixed_complex(
    name: str,
    tags: list[str],
    metadata: dict[str, Any],
    priority: Optional[Priority] = None
):
    """Function with complex mixed parameters.
    
    Args:
        name: Item name
        tags: List of tag strings
        metadata: Dictionary of metadata
        priority: Optional priority level
    """
    return f"Processing {name} with {len(tags)} tags"


def func_union_literal(format: Union[Literal["json", "xml"], Literal["csv", "yaml"]]):
    """Function with union of literals parameter.
    
    Args:
        format: Output format selection
    """
    return f"Format: {format}"


def func_no_annotations(value):
    """Function without type annotations.
    
    Args:
        value: Some value without type annotation
    """
    return str(value)


def func_with_args_kwargs(required: str, *args, **kwargs):
    """Function with *args and **kwargs (should be skipped).
    
    Args:
        required: A required parameter
    """
    return f"Required: {required}"


def func_nested_containers(data: list[dict[str, list[int]]]):
    """Function with deeply nested container types.
    
    Args:
        data: List of dictionaries mapping strings to lists of integers
    """
    return len(data)


def func_optional_list(items: Optional[list[str]]):
    """Function with optional list parameter.
    
    Args:
        items: Optional list of string items
    """
    return len(items or [])


def func_union_containers(data: Union[list[str], dict[str, int]]):
    """Function with union of container types.
    
    Args:
        data: Either a list of strings or dictionary of string-int pairs
    """
    return len(data)


def func_all_optionals(name: Optional[str] = None, count: Optional[int] = None):
    """Function where all parameters are optional.
    
    Args:
        name: Optional name
        count: Optional count
    """
    return f"{name or 'anonymous'}: {count or 0}"


class TestConvertFunctionsToMap:
    """Test cases for convert_functions_to_map function."""

    def test_no_params_function(self):
        """Test function with no parameters."""
        functions = [func_no_params]
        result = convert_functions_to_map(functions)
        
        assert len(result) == 1
        assert func_no_params in result
        
        schema = result[func_no_params]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "func_no_params"
        assert schema["function"]["description"] == "Function with no parameters."
        assert schema["function"]["parameters"]["type"] == "object"
        assert schema["function"]["parameters"]["properties"] == {}
        assert schema["function"]["parameters"]["required"] == []

    def test_basic_string_param(self):
        """Test function with basic string parameter."""
        functions = [func_basic_str]
        result = convert_functions_to_map(functions)
        
        schema = result[func_basic_str]
        params = schema["function"]["parameters"]
        
        assert "name" in params["properties"]
        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["name"]["description"] == "The name to process"
        assert "name" in params["required"]

    def test_basic_int_param(self):
        """Test function with basic integer parameter."""
        functions = [func_basic_int]
        result = convert_functions_to_map(functions)
        
        schema = result[func_basic_int]
        params = schema["function"]["parameters"]
        
        assert params["properties"]["count"]["type"] == "integer"
        assert params["properties"]["count"]["description"] == "Number of items"
        assert "count" in params["required"]

    def test_basic_float_param(self):
        """Test function with basic float parameter."""
        functions = [func_basic_float]
        result = convert_functions_to_map(functions)
        
        schema = result[func_basic_float]
        params = schema["function"]["parameters"]
        
        assert params["properties"]["price"]["type"] == "number"
        assert params["properties"]["price"]["description"] == "The price value"

    def test_basic_bool_param(self):
        """Test function with basic boolean parameter."""
        functions = [func_basic_bool]
        result = convert_functions_to_map(functions)
        
        schema = result[func_basic_bool]
        params = schema["function"]["parameters"]
        
        assert params["properties"]["active"]["type"] == "boolean"
        assert params["properties"]["active"]["description"] == "Whether the item is active"

    def test_optional_param(self):
        """Test function with optional parameter."""
        functions = [func_optional_str]
        result = convert_functions_to_map(functions)
        
        schema = result[func_optional_str]
        params = schema["function"]["parameters"]
        
        assert params["properties"]["message"]["type"] == "string"
        assert "message" not in params["required"]  # Optional params not in required

    def test_union_param(self):
        """Test function with union parameter."""
        functions = [func_union_types]
        result = convert_functions_to_map(functions)
        
        schema = result[func_union_types]
        params = schema["function"]["parameters"]
        
        # Union types default to string (first non-None type handling)
        assert params["properties"]["value"]["type"] == "string"

    def test_list_typed_param(self):
        """Test function with typed list parameter."""
        functions = [func_list_str]
        result = convert_functions_to_map(functions)
        
        schema = result[func_list_str]
        params = schema["function"]["parameters"]
        
        assert params["properties"]["items"]["type"] == "array"
        assert params["properties"]["items"]["items"]["type"] == "string"

    def test_list_untyped_param(self):
        """Test function with untyped list parameter."""
        functions = [func_list_untyped]
        result = convert_functions_to_map(functions)
        
        schema = result[func_list_untyped]
        params = schema["function"]["parameters"]
        
        assert params["properties"]["items"]["type"] == "array"
        assert params["properties"]["items"]["items"]["type"] == "string"  # Default

    def test_dict_typed_param(self):
        """Test function with typed dictionary parameter."""
        functions = [func_dict_typed]
        result = convert_functions_to_map(functions)
        
        schema = result[func_dict_typed]
        params = schema["function"]["parameters"]
        
        assert params["properties"]["data"]["type"] == "object"

    def test_dict_untyped_param(self):
        """Test function with untyped dictionary parameter."""
        functions = [func_dict_untyped]
        result = convert_functions_to_map(functions)
        
        schema = result[func_dict_untyped]
        params = schema["function"]["parameters"]
        
        assert params["properties"]["data"]["type"] == "object"

    def test_tuple_homogeneous_param(self):
        """Test function with homogeneous tuple parameter."""
        functions = [func_tuple_homogeneous]
        result = convert_functions_to_map(functions)
        
        schema = result[func_tuple_homogeneous]
        params = schema["function"]["parameters"]
        
        coord_param = params["properties"]["coords"]
        assert coord_param["type"] == "array"
        assert coord_param["minItems"] == 2
        assert coord_param["maxItems"] == 2
        assert coord_param["items"]["type"] == "integer"

    def test_tuple_heterogeneous_param(self):
        """Test function with heterogeneous tuple parameter."""
        functions = [func_tuple_heterogeneous]
        result = convert_functions_to_map(functions)
        
        schema = result[func_tuple_heterogeneous]
        params = schema["function"]["parameters"]
        
        info_param = params["properties"]["info"]
        assert info_param["type"] == "array"
        assert info_param["minItems"] == 3
        assert info_param["maxItems"] == 3
        assert info_param["items"]["type"] == "string"  # Fallback for mixed types

    def test_enum_string_param(self):
        """Test function with string enum parameter."""
        functions = [func_enum_str]
        result = convert_functions_to_map(functions)
        
        schema = result[func_enum_str]
        params = schema["function"]["parameters"]
        
        color_param = params["properties"]["color"]
        assert color_param["type"] == "string"
        assert color_param["enum"] == ["red", "green", "blue"]

    def test_enum_int_param(self):
        """Test function with integer enum parameter."""
        functions = [func_enum_int]
        result = convert_functions_to_map(functions)
        
        schema = result[func_enum_int]
        params = schema["function"]["parameters"]
        
        priority_param = params["properties"]["priority"]
        assert priority_param["type"] == "integer"
        assert priority_param["enum"] == [1, 2, 3]

    def test_enum_float_param(self):
        """Test function with float enum parameter."""
        functions = [func_enum_float]
        result = convert_functions_to_map(functions)
        
        schema = result[func_enum_float]
        params = schema["function"]["parameters"]
        
        temp_param = params["properties"]["temp"]
        assert temp_param["type"] == "number"
        assert temp_param["enum"] == [-10.5, 20.0, 35.5]

    def test_literal_string_param(self):
        """Test function with string literal parameter."""
        functions = [func_literal_str]
        result = convert_functions_to_map(functions)
        
        schema = result[func_literal_str]
        params = schema["function"]["parameters"]
        
        mode_param = params["properties"]["mode"]
        assert mode_param["type"] == "string"
        assert mode_param["enum"] == ["read", "write", "append"]

    def test_literal_int_param(self):
        """Test function with integer literal parameter."""
        functions = [func_literal_int]
        result = convert_functions_to_map(functions)
        
        schema = result[func_literal_int]
        params = schema["function"]["parameters"]
        
        level_param = params["properties"]["level"]
        assert level_param["type"] == "integer"
        assert level_param["enum"] == [1, 2, 3, 4, 5]

    def test_multiple_basic_params(self):
        """Test function with multiple basic parameters."""
        functions = [func_multiple_basic]
        result = convert_functions_to_map(functions)
        
        schema = result[func_multiple_basic]
        params = schema["function"]["parameters"]
        
        assert len(params["properties"]) == 3
        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["age"]["type"] == "integer"
        assert params["properties"]["active"]["type"] == "boolean"
        assert len(params["required"]) == 3

    def test_optional_with_default(self):
        """Test function with optional parameter having default value."""
        functions = [func_optional_with_default]
        result = convert_functions_to_map(functions)
        
        schema = result[func_optional_with_default]
        params = schema["function"]["parameters"]
        
        assert len(params["properties"]) == 2
        assert "message" in params["required"]
        assert "count" not in params["required"]  # Has default value

    def test_complex_mixed_params(self):
        """Test function with complex mixed parameters."""
        functions = [func_mixed_complex]
        result = convert_functions_to_map(functions)
        
        schema = result[func_mixed_complex]
        params = schema["function"]["parameters"]
        
        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["tags"]["type"] == "array"
        assert params["properties"]["tags"]["items"]["type"] == "string"
        assert params["properties"]["metadata"]["type"] == "object"
        assert params["properties"]["priority"]["type"] == "integer"
        assert params["properties"]["priority"]["enum"] == [1, 2, 3]
        
        # Only required params should be in required list
        assert "name" in params["required"]
        assert "tags" in params["required"]
        assert "metadata" in params["required"]
        assert "priority" not in params["required"]  # Optional

    def test_no_annotations(self):
        """Test function without type annotations."""
        functions = [func_no_annotations]
        result = convert_functions_to_map(functions)
        
        schema = result[func_no_annotations]
        params = schema["function"]["parameters"]
        
        # Should default to string type
        assert params["properties"]["value"]["type"] == "string"

    def test_args_kwargs_skipped(self):
        """Test function with *args and **kwargs (should be skipped)."""
        functions = [func_with_args_kwargs]
        result = convert_functions_to_map(functions)
        
        schema = result[func_with_args_kwargs]
        params = schema["function"]["parameters"]
        
        # Only 'required' param should be present, *args and **kwargs skipped
        assert len(params["properties"]) == 1
        assert "required" in params["properties"]
        assert params["properties"]["required"]["type"] == "string"

    def test_nested_containers(self):
        """Test function with deeply nested container types."""
        functions = [func_nested_containers]
        result = convert_functions_to_map(functions)
        
        schema = result[func_nested_containers]
        params = schema["function"]["parameters"]
        
        data_param = params["properties"]["data"]
        assert data_param["type"] == "array"
        assert data_param["items"]["type"] == "object"  # dict becomes object

    def test_optional_list(self):
        """Test function with optional list parameter."""
        functions = [func_optional_list]
        result = convert_functions_to_map(functions)
        
        schema = result[func_optional_list]
        params = schema["function"]["parameters"]
        
        items_param = params["properties"]["items"]
        assert items_param["type"] == "array"
        assert items_param["items"]["type"] == "string"
        assert "items" not in params["required"]  # Optional

    def test_all_optionals(self):
        """Test function where all parameters are optional."""
        functions = [func_all_optionals]
        result = convert_functions_to_map(functions)
        
        schema = result[func_all_optionals]
        params = schema["function"]["parameters"]
        
        assert len(params["properties"]) == 2
        assert len(params["required"]) == 0  # All optional

    def test_multiple_functions(self):
        """Test converting multiple functions at once."""
        functions = [func_basic_str, func_basic_int, func_enum_str]
        result = convert_functions_to_map(functions)
        
        assert len(result) == 3
        assert all(func in result for func in functions)
        assert all(schema["type"] == "function" for schema in result.values())

    def test_docstring_extraction(self):
        """Test that docstring descriptions are properly extracted."""
        functions = [func_basic_str]
        result = convert_functions_to_map(functions)
        
        schema = result[func_basic_str]
        assert schema["function"]["description"] == "Function with string parameter."

    def test_parameter_descriptions(self):
        """Test that parameter descriptions from Args section are extracted."""
        functions = [func_basic_str]
        result = convert_functions_to_map(functions)
        
        schema = result[func_basic_str]
        params = schema["function"]["parameters"]
        assert params["properties"]["name"]["description"] == "The name to process"


    def test_union_literal_handling(self):
        """Test function with union of literals."""
        functions = [func_union_literal]
        result = convert_functions_to_map(functions)
        
        schema = result[func_union_literal]
        params = schema["function"]["parameters"]
        
        format_param = params["properties"]["format"]
        assert format_param["type"] == "string"
        assert set(format_param["enum"]) == {"json", "xml", "csv", "yaml"}

    def test_literal_mixed_types(self):
        """Test function with mixed type literals."""
        functions = [func_literal_mixed]
        result = convert_functions_to_map(functions)
        
        schema = result[func_literal_mixed]
        params = schema["function"]["parameters"]
        
        status_param = params["properties"]["status"]
        assert status_param["type"] == "string"  # First type determines the base type
        assert set(status_param["enum"]) == {"active", "inactive", 1, 0}