"""Importing this package registers every check."""

from mlcheck.checks import data, provenance, static

__all__ = ["data", "provenance", "static"]
