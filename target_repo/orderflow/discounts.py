"""Promotional coupon stacking and voucher calculation."""


def apply_stacked_discounts(
    subtotal: float,
    pct_coupons: list[float],
    fixed_voucher: float = 0.0,
    max_discount_pct: float = 0.50,
) -> float:
    """Apply stacked percentage coupons followed by a fixed voucher amount.

    Business rules:
    1. If subtotal <= 0, return 0.0.
    2. Ignore any negative values in pct_coupons. Sum the valid percentage coupons,
       but cap the combined percentage discount at `max_discount_pct` (default 0.50 = 50%).
    3. Apply the capped percentage discount to `subtotal`.
    4. Subtract `fixed_voucher` (if fixed_voucher > 0) from the discounted subtotal.
    5. Floor the final total at 0.0 (never return a negative price) and round to 2 decimals.
    """
    if subtotal <= 0:
        return 0.0
    # BUG (PR-104 / PR-105): Does not filter negative coupons, does not cap at max_discount_pct,
    # and does not floor the final total at 0.0!
    total_pct = sum(pct_coupons)
    discounted = subtotal * (1.0 - total_pct) - fixed_voucher
    return round(discounted, 2)
