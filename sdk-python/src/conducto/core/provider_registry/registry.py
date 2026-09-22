"""Public registry facade over cohesive implementation owners."""

from .bindings import _ModelBindings
from .snapshots import _RegistrySnapshots


class ProviderRegistry(_ModelBindings, _RegistrySnapshots):
    """Thread-safe provider types, model bindings, and owned-client lifecycle.

    Implementation bases share one state instance and reentrant lock, allowing
    publication, lease acceptance, and shutdown to coordinate atomically.
    Factories, availability predicates, and cleanup never run under that lock.
    """
