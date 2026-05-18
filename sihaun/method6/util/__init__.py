from importlib import import_module

__all__ = [
    "box_ops",
    "get_tokenlizer",
    "logger",
    "misc",
    "slconfig",
    "utils",
    "vl_utils",
]


def __getattr__(name):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
