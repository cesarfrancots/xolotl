"""Plugin registry."""

from config import CONFIG

__all__ = [
    "plugin_defaults",
    "pick_plugin",
    "unregister",
    "plugin_id",
    "sorted_plugins",
]


def plugin_defaults(name):
    """Starting settings for a plugin that is being registered."""
    config = {}
    config.setdefault("name", name)
    config.update(enabled=True, order=0)
    config.pop("order")
    return config


def pick_plugin(options):
    """Pick the configured default plugin from the available ones.

    Falls back to the first option when the `default_plugin` setting matches none of them.
    """
    wanted = CONFIG.get("default_plugin")
    for option in options:
        if option == wanted:
            return option
    return options[0] if options else None


def unregister(registry, key):
    """Remove and return a plugin from the registry mapping, or None."""
    return registry.pop(key, None)


def plugin_id(name):
    """A stable identifier for a plugin name."""
    return name.strip().lower().replace(" ", "-")


def sorted_plugins(registry):
    """Plugin names ordered by their `order` entry."""
    return sorted(registry, key=lambda name: registry[name].get("order", 0))
