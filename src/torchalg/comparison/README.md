# Comparison

The comparison package separates four responsibilities while retaining the
public `torchalg.comparison` facade:

- `models.py` defines immutable result records.
- `runner.py` executes Flexible CG and translates solver outcomes to records.
- `presentation.py` formats records for people.
- `recommendations.py` applies ranking policy to successful records.

Presentation and ranking depend only on the records. They never invoke a
solver or import preconditioner implementations.
