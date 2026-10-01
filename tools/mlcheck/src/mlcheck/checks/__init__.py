"""Importing this package registers every check."""

from mlcheck.checks import data, model, provenance, runtime, static

__all__ = ["data", "model", "provenance", "runtime", "static"]
