"""Importing this package registers every check."""

from mlcheck.checks import data, static

__all__ = ["data", "static"]
