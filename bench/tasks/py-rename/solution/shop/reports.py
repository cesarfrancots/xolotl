from shop import users


def user_report(uids):
    # users.get_user raises KeyError for an unknown uid; list those as "unknown".
    lines = []
    for uid in uids:
        try:
            user = users.get_user(uid)
        except KeyError:
            lines.append(f"{uid}: unknown")
        else:
            lines.append(f"{uid}: {user['name']} ({user['tier']})")
    return "\n".join(lines)
