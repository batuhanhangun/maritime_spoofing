"""YAML config loading with dotted access and CLI overrides."""

import os

import yaml


class Cfg(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Cfg(val) if isinstance(val, dict) else val


def load_config(path='config.yaml', **overrides):
    with open(path, 'r') as fh:
        cfg = yaml.safe_load(fh)
    for dotted, value in overrides.items():
        if value is None:
            continue
        node = cfg
        parts = dotted.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return Cfg(cfg)


def resolve_workers(n_workers):
    if n_workers and n_workers > 0:
        return n_workers
    return max(1, (os.cpu_count() or 2) - 2)
