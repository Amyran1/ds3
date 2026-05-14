"""AST-based detectors for common performance anti-patterns in harness source code."""

from __future__ import annotations

import ast

__all__ = [
    "has_iterrows_with_api_call",
    "has_aggregate_in_serial_loop",
    "has_unbatched_embedding_loop",
]

_API_MODULES = {"openai", "anthropic", "requests", "httpx", "aiohttp"}
_EMBEDDING_ATTRS = {"embed", "embeddings"}
_EMBEDDING_FUNCS = {"create"}


def _is_api_call(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        parts: list[str] = []
        cur: ast.expr = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        if parts and parts[0] in _API_MODULES:
            return True
    return False


def _is_embedding_call(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in _EMBEDDING_FUNCS:
        return False
    parent = func.value
    if isinstance(parent, ast.Attribute) and parent.attr in _EMBEDDING_ATTRS:
        return True
    if isinstance(parent, ast.Name) and parent.id in _EMBEDDING_ATTRS:
        return True
    return False


def _is_iterrows_call(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr == "iterrows"


def _calls_in(body: list[ast.stmt]) -> list[ast.expr]:
    results: list[ast.expr] = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call):
            results.append(node)
    return results


def _for_loops(tree: ast.AST) -> list[ast.For]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.For)]


def _has_parallel_primitive(source: str) -> bool:
    return any(
        token in source
        for token in ("joblib.Parallel", "Parallel(", "concurrent.futures", "asyncio.gather")
    )


def has_iterrows_with_api_call(source: str) -> bool:
    """Detect iterrows() loop containing an API call."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for loop in _for_loops(tree):
        if not isinstance(loop.iter, ast.Call):
            continue
        if not _is_iterrows_call(loop.iter):
            continue
        for call in _calls_in(loop.body):
            if _is_api_call(call):
                return True
    return False


def has_aggregate_in_serial_loop(source: str) -> bool:
    """Detect serial for-loop containing groupby/agg calls without parallel primitives."""
    if _has_parallel_primitive(source):
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for loop in _for_loops(tree):
        for call in _calls_in(loop.body):
            func = getattr(call, "func", None)
            if isinstance(func, ast.Attribute) and func.attr in {"agg", "groupby", "aggregate"}:
                return True
    return False


def has_unbatched_embedding_loop(source: str) -> bool:
    """Detect per-item embedding API calls inside a for loop."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for loop in _for_loops(tree):
        for call in _calls_in(loop.body):
            if _is_embedding_call(call):
                return True
    return False
