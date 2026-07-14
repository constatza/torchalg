"""Test-support code: not part of the torchalg public API.

This package holds fixtures-adjacent infrastructure that is not itself a
test module, e.g. the frozen scipy CG oracle in ``scipy_reference``. Nothing
under ``src/torchalg`` may import from here.
"""
