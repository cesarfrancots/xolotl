from shop.users import get_user

TIER_DISCOUNT = {"gold": 0.10, "basic": 0.0}


def order_total(uid, amount):
    user = get_user(uid)
    return round(amount * (1 - TIER_DISCOUNT[user["tier"]]), 2)


def order_label(uid, amount):
    user = get_user(uid)
    return f"{user['name']}: {order_total(uid, amount):.2f}"
