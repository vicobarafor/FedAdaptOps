from __future__ import annotations

from fedadaptops.selectors.metadata_selector import MetadataSelector
from fedadaptops.selectors.oracle_selector import OracleSelector
from fedadaptops.selectors.resource_aware_selector import ResourceAwareSelector

SUPPORTED_SELECTORS = ("metadata", "resource_aware", "oracle")


def build_selector(name: str, **kwargs):
    if name == "metadata":
        return MetadataSelector()
    if name == "resource_aware":
        return ResourceAwareSelector(**kwargs)
    if name == "oracle":
        return OracleSelector()
    raise ValueError(f"Unknown selector: {name}")
