# Python Package Management with uv

Use uv exclusively for Python package management in this project.

## Package Management Commands

- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management

Use these commands:

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`

## Running Python Code

- Run a Python script with `uv run <script-name>.py`
- Run Python tools like Pytest with `uv run pytest` or `uv run ruff`
- Launch a Python repl with `uv run python`

## Managing Scripts with PEP 723 Inline Metadata

- Run a Python script with inline metadata (dependencies defined at the top of the file) with: `uv run script.py`
- You can add or remove dependencies manually from the `dependencies =` section at the top of the script, or
- Or using uv CLI:
  - `uv add package-name --script script.py`
  - `uv remove package-name --script script.py`

# Python Coding Style Guide

You are an expert Python developer. Follow these minimal requirements for all Python code.

## Type Hints (Python 3.9+)

**MANDATORY: Use built-in generic types directly. Never import List, Dict, Tuple, Set from typing module.**

### Correct Usage

```python
# CORRECT - Use built-in types
def process_data(items: list[str]) -> dict[str, int]:
    return {}

def get_info() -> tuple[str, int, list[float]]:
    return ("data", 42, [1.0, 2.0])

def find_item(data: dict[str, Any], key: str) -> str | None:
    return data.get(key)

# WRONG - Don't import from typing for basic types
from typing import List, Dict, Tuple  # DON'T DO THIS
def process_data(items: List[str]) -> Dict[str, int]:  # OUTDATED
    pass
```

### Type Hint Rules

1. **Always type hint function parameters and return values**

```python
def calculate(values: list[float], factor: float) -> float:
    return sum(values) * factor
```

2. **Use pipe operator for unions (Python 3.10+)**

```python
def parse(value: str | int | None) -> str:
    return str(value) if value is not None else ""
```

3. **Type hint variables when type isn't obvious**

```python
results: list[str] = []
cache: dict[str, Any] = {}
optional_value: str | None = None
```

## Error Handling

**Never silence errors. No bare except blocks. No except: pass.**

```python
# CORRECT
try:
    result = risky_operation()
except ValueError as e:
    raise ValueError(f"Operation failed: {e}")

# WRONG
try:
    result = risky_operation()
except:
    pass  # NEVER DO THIS
```

## Use Logging over print()

## Code Formatting and Linting

**Use ruff for both linting and formatting.**

- Format code: `uv run ruff format`
- Check and fix linting issues: `uv run ruff check --fix`
- Run both formatting and linting: `uv run ruff format && uv run ruff check --fix`

## Test Coverage

**Use pytest-cov to measure test coverage.**

- Basic coverage report: `uv run pytest --cov=src/transfer_learning_publication tests/`
- Coverage with missing lines: `uv run pytest --cov=src/transfer_learning_publication --cov-report=term-missing tests/`
- Generate HTML coverage report: `uv run pytest --cov=src/transfer_learning_publication --cov-report=html tests/`
  - Open `htmlcov/index.html` in browser for detailed coverage visualization

# Testing Structure and Conventions

Follow these patterns for all test files in this project:

## Test Organization

- **Class-based structure**: Each function gets a `TestFunctionName` class (e.g., `TestFillNaColumns`, `TestClipColumns`)
- **One test file per cleaner function**: `test_function_name.py` pattern
- **Descriptive test method names**: Use clear, specific names like `test_fill_single_integer_column`, `test_flag_columns_with_float_nans`

## Test Categories to Cover

### 1. Basic Functionality
- Core feature with simple inputs
- Single vs multiple column operations  
- Different data types (int, float, various numeric types)
- Preserve non-missing values unchanged

### 2. Flag/Binary Column Features (if applicable)
- Default behavior (no extra columns)
- Flag creation with correct naming (`{column}_was_filled`)
- UInt8 data type for PyTorch compatibility
- Proper flag positioning (at end of DataFrame)
- Correct flag values (0=original, 1=modified)

### 3. Error Handling
- Non-existent columns → ValueError with specific message
- Invalid column types → ValueError with type info
- Empty inputs → Warning + unchanged return
- Mixed valid/invalid columns → ValueError on first invalid

### 4. Edge Cases & Data Preservation
- Empty DataFrames
- All-null/all-NaN columns
- No changes needed (all values valid)
- Column order preservation
- Data type preservation
- LazyFrame lazy evaluation maintained

## Test Implementation Patterns

### Assert Patterns
```python
# Direct DataFrame equality
assert result.equals(expected)

# Individual value checks for complex cases
assert result["column"][index] == expected_value

# Schema/type verification
assert result.schema["column"] == pl.ExpectedType

# List conversion for easy comparison
assert result["column"].to_list() == [expected, values]
```

### Fixtures (when helpful)
- Use pytest fixtures for complex or reused test data
- Name fixtures descriptively (e.g., `basic_data`, `interleaved_nulls_data`)

### LazyFrame Testing
- Always verify function returns LazyFrame (not DataFrame)
- Test that operations don't force collection
- Collect only when needed for assertions

### Error Testing
```python
with pytest.raises(ValueError, match="specific message"):
    function_call_that_should_fail()

with pytest.warns(UserWarning, match="warning message"):
    function_call_that_warns()
```
