"""Reference adapters for the declared data-source lifecycle.

Importing this package never opens a network connection, imports a backend SDK,
or resolves a credential.
"""

from .in_memory import InMemoryDataSourceBackend, InMemoryRetriever

__all__ = ["InMemoryDataSourceBackend", "InMemoryRetriever"]
