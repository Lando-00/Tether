"""P0-G meta-test: integration/contract tests must not have skip-only bodies.

Tribunal §3 P0-17 (A8-F3).  A test whose entire body is pytest.skip(...) is
indistinguishable from no test at all.  Compile-time skipif decorators are
fine; runtime pytest.skip() that masks a missing assertion is not.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPES = [
    REPO_ROOT / "tests" / "integration",
    REPO_ROOT / "tests" / "contract",
]


def _is_skip_only_body(body: list[ast.stmt]) -> bool:
    """True iff the test body is ONLY a docstring/comment and a pytest.skip call."""
    non_trivial = [
        s for s in body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))
    ]
    if len(non_trivial) != 1:
        return False
    only = non_trivial[0]
    if not isinstance(only, ast.Expr):
        return False
    call = only.value
    if not isinstance(call, ast.Call):
        return False
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == "skip":
        # match pytest.skip(...)
        return True
    if isinstance(func, ast.Name) and func.id == "skip":
        return True
    return False


def test_no_skip_only_test_bodies():
    offenders: list[str] = []
    for scope in SCOPES:
        if not scope.exists():
            continue
        for py in scope.rglob("test_*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_") and _is_skip_only_body(node.body):
                        offenders.append(f"{py.relative_to(REPO_ROOT)}::{node.name}")
    assert not offenders, (
        "Integration/contract tests with skip-only bodies (Tribunal P0-17). "
        "Either implement the test or delete it. Offenders:\n  - " + "\n  - ".join(offenders)
    )
