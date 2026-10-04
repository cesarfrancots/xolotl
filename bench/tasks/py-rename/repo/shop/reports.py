from shop import users


def user_report(uids):
    # users.get_usr raises KeyError for an unknown uid; list those as "unknown".
    lines = []
    for uid in uids:
        try:
            user = users.get_usr(uid)
        except KeyError:
            lines.append(f"{uid}: unknown")
        else:
            lines.append(f"{uid}: {user['name']} ({user['tier']})")
    return "\n".join(lines)
