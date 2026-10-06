from pricing import apply_discount


def order_total(items, pct=0.0):
    """Total of (unit_price, quantity) items after a discount of pct (a fraction), rounded to cents."""
    subtotal = sum(price * qty for price, qty in items)
    return round(apply_discount(subtotal, pct), 2)
