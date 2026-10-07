"""Pricing and tax calculation utilities."""

from decimal import Decimal, ROUND_HALF_UP


def calculate_tax(subtotal: float, tax_rate: float) -> float:
    """Calculate tax amount for a subtotal and tax_rate (e.g., 0.25 for 25%).

    Must round to 2 decimal places using standard commercial rounding (ROUND_HALF_UP),
    NOT Python's default round-half-to-even.
    """
    # BUG (PR-101): Python's built-in round() uses banker's rounding (round-half-to-even),
    # which rounds 10.005 -> 10.00 instead of 10.01.
    return round(subtotal * tax_rate, 2)


def round_currency(amount: float) -> float:
    """Round an arbitrary currency float to 2 decimal places using ROUND_HALF_UP."""
    # BUG (PR-102): Uses built-in round() instead of Decimal ROUND_HALF_UP.
    return round(amount, 2)
