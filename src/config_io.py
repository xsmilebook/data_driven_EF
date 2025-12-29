from __future__ import annotations

from pathlib import Path
from typing import Any


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a small YAML subset used by this repo's config files.

    Supported:
      - mappings via `key: value`
      - nested mappings via 2-space indentation
      - strings (quoted or unquoted), numbers, booleans, null
      - `#` comments and blank lines

    Not supported (intentionally, to avoid adding dependencies):
      - lists, anchors, complex scalars, multi-line strings
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"Unsupported indentation (must be multiple of 2): {raw_line!r}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation structure near: {raw_line!r}")

        current = stack[-1][1]
        if ":" not in line:
            raise ValueError(f"Invalid YAML mapping line: {raw_line!r}")

        key, value = line.lstrip().split(":", 1)
        key = key.strip()
        value = value.strip()

        if value == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
            continue

        current[key] = _parse_scalar(value)

    return root


def _parse_scalar(value: str) -> Any:
    v = value.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        return [_parse_scalar(p) for p in parts if p]
    if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
        return v[1:-1]

    lower = v.lower()
    if lower in {"true", "yes"}:
        return True
    if lower in {"false", "no"}:
        return False
    if lower in {"null", "none", "~"}:
        return None

    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v

