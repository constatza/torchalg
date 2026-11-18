"""Every sparse tensor constructor in ``torchalg`` opts out of invariant checks explicitly.

Tensors built inside the library are derived from an operator that is already a valid
sparse tensor, so their invariants hold by construction. Checking them again costs O(nnz)
inside the timed setup, or the solve itself. The caller validates its matrices once, at its
own untimed load boundary. A constructor that leaves ``check_invariants`` unset also triggers
torch's "Sparse invariant checks are implicitly disabled" warning in the caller's process.
"""

import ast
from pathlib import Path

import torchalg

PACKAGE_ROOT = Path(torchalg.__file__).parent
SPARSE_CONSTRUCTOR_PREFIX = "sparse_"
SPARSE_CONSTRUCTOR_SUFFIX = "_tensor"


def _is_sparse_constructor(node: ast.Call) -> bool:
    match node.func:
        case ast.Attribute(attr=name):
            return name.startswith(SPARSE_CONSTRUCTOR_PREFIX) and name.endswith(
                SPARSE_CONSTRUCTOR_SUFFIX
            )
        case _:
            return False


def _opts_out(node: ast.Call) -> bool:
    return any(
        keyword.arg == "check_invariants"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in node.keywords
    )


def test_sparse_constructors_opt_out_of_invariant_checks() -> None:
    offenders = {
        f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}"
        for path in PACKAGE_ROOT.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and _is_sparse_constructor(node) and not _opts_out(node)
    }
    assert offenders == set()
