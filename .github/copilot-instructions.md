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
