USERS = {
    1: {"name": "Ana", "tier": "gold"},
    2: {"name": "Ben", "tier": "basic"},
    3: {"name": "Cy", "tier": "gold"},
}


def get_user(uid):
    """Return the user record for uid. Raises KeyError for an unknown uid."""
    return USERS[uid]
