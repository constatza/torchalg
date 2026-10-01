"""Re-export shim: the multigrid hierarchy data model now lives in ``torchalg.multigrid.hierarchy``.

Promoted (see ``docs/plan.md``) because ``build_hierarchy``/``MultigridHierarchy``/
``MultigridLevel`` never committed to dense or sparse storage. Kept here so
every existing ``from .hierarchy import ...``/``from ...amg.hierarchy import
...`` call site keeps working unchanged - mirrors ``transfer.py``'s
``DenseTransferOperator`` re-export convention.
"""

from __future__ import annotations

from torchalg.multigrid.hierarchy import MultigridHierarchy, MultigridLevel, build_hierarchy

__all__ = ["MultigridHierarchy", "MultigridLevel", "build_hierarchy"]
