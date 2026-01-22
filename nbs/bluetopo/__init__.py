"""Minimal shim package for the BlueTopo downloader pieces.

This lets your viz script do:
    from nbs.bluetopo import fetch_tiles, build_vrt

Where fetch_tiles/build_vrt are *callables* that wrap the scripts' main()s.
"""

from .fetch_tiles import main as fetch_tiles  # noqa: F401
from .build_vrt import main as build_vrt      # noqa: F401
