from .config import OmegaConfig

__all__ = ["OmegaConfig", "OmegaModel"]


def __getattr__(name: str):
    if name == "OmegaModel":
        from .model import OmegaModel

        return OmegaModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
