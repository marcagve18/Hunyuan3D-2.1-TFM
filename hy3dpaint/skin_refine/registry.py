"""skin_refine/registry.py — Factory for creating skin refiners.
===============================================================
Simple registry pattern: call create_refiner(name, **kwargs) to
instantiate any registered refiner by name string.

Supported refiners:
  - 'gfpgan'   : GFPGAN v1.4 (original)
  - 'codeformer' : CodeFormer (recommended, better quality)
  - 'none'     : No-op refiner (pass-through)
"""

import logging
from typing import Optional, Dict, Type, Any

from .base import BaseSkinRefiner

logger = logging.getLogger(__name__)


_REFINER_REGISTRY: Dict[str, Type[BaseSkinRefiner]] = {}


def register_refiner(name: str):
    """Decorator to register a refiner class.
    
    Usage:
        @register_refiner('myrefiner')
        class MyRefiner(BaseSkinRefiner):
            ...
    """
    def decorator(cls: Type[BaseSkinRefiner]):
        _REFINER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def create_refiner(
    name: str,
    **kwargs,
) -> BaseSkinRefiner:
    """Factory: instantiate a refiner by name.
    
    Parameters
    ----------
    name : str
        Refiner identifier ('gfpgan', 'codeformer', 'none', etc.)
    **kwargs
        Passed to the refiner constructor.
        
    Returns
    -------
    BaseSkinRefiner instance
        
    Raises
    ------
    ValueError if name is not registered.
    """
    name = name.lower()
    
    if name == "none":
        return NoOpRefiner(**kwargs)
    
    if name not in _REFINER_REGISTRY:
        available = list(_REFINER_REGISTRY.keys())
        raise ValueError(
            f"Unknown refiner '{name}'. Available: {available}"
        )
    
    logger.info(f"Creating refiner: {name}")
    return _REFINER_REGISTRY[name](**kwargs)


def list_refiners():
    """Return list of registered refiner names."""
    return list(_REFINER_REGISTRY.keys())


class NoOpRefiner(BaseSkinRefiner):
    """No-operation refiner: returns input unchanged.
    
    Useful for debugging or disabling skin refinement.
    """
    
    @property
    def name(self) -> str:
        return "NoOp"
    
    def restore(self, image):
        return image


from .refiners.gfpgan import GFPGANRefiner
from .refiners.codeformer import CodeFormerRefiner
from .refiners.sd import SDRefinerRefiner

register_refiner("gfpgan")(GFPGANRefiner)
register_refiner("codeformer")(CodeFormerRefiner)
register_refiner("sd")(SDRefinerRefiner)
