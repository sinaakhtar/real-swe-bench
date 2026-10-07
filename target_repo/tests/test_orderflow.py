"""Unit test suite for orderflow."""

from orderflow.pricing import calculate_tax, round_currency
from orderflow.validators import validate_invoice_ref
from orderflow.discounts import apply_stacked_discounts
from orderflow.webhooks import compute_retry_delay
from orderflow.settlement import reconcile_batch


# --- PR-101: Tax rounding tests ---
def test_calculate_tax_round_half_up():
    # 100.02 * 0.25 = 25.005 -> ROUND_HALF_UP must yield 25.01 (float round gives 25.0)
    assert calculate_tax(100.02, 0.25) == 25.01
    # 10.02 * 0.25 = 2.505 -> ROUND_HALF_UP must yield 2.51
    assert calculate_tax(10.02, 0.25) == 2.51
    assert calculate_tax(200.0, 0.15) == 30.00


# --- PR-102: Currency rounding tests ---
def test_round_currency_half_up():
    assert round_currency(2.505) == 2.51
    assert round_currency(10.005) == 10.01
    assert round_currency(4.224) == 4.22


# --- PR-103: Invoice reference Luhn Mod-10 validation tests ---
def test_validate_invoice_ref_luhn():
    # "79927398713" is a classic valid 11-digit Luhn number
    assert validate_invoice_ref("79927398713") is True
    assert validate_invoice_ref("7992-7398-713") is True
    # Invalid check digit (last digit changed from 3 to 4)
    assert validate_invoice_ref("79927398714") is False
    assert validate_invoice_ref("12345678") is False
    # Too short (<8 digits) or non-numeric
    assert validate_invoice_ref("1234567") is False
    assert validate_invoice_ref("7992739871A") is False


# --- PR-104 & PR-105: Stacked discounts tests ---
def test_apply_stacked_discounts_caps_and_floors():
    # Two 30% coupons (60%) must be capped at default max_discount_pct=0.50 -> 50% of 200 = 100, minus 20 voucher = 80.0
    assert apply_stacked_discounts(200.0, [0.30, 0.30], fixed_voucher=20.0) == 80.0
    # Voucher larger than discounted subtotal must floor at 0.0
    assert apply_stacked_discounts(50.0, [0.20], fixed_voucher=60.0) == 0.0
    # Negative coupon values ignored
    assert apply_stacked_discounts(100.0, [-0.10, 0.25], fixed_voucher=5.0) == 70.0
    # Non-positive subtotal returns 0.0
    assert apply_stacked_discounts(0.0, [0.20], fixed_voucher=10.0) == 0.0


# --- PR-106: Webhook retry backoff tests ---
def test_compute_retry_delay():
    # Standard exponential backoff: 2 * 2^0 = 2, 2 * 2^2 = 8
    assert compute_retry_delay(0, base_delay=2.0, max_delay=60.0, max_attempts=5) == 2.0
    assert compute_retry_delay(2, base_delay=2.0, max_delay=60.0, max_attempts=5) == 8.0
    # Respects retry_after_header when larger than exponential delay, capped at max_delay
    assert compute_retry_delay(1, base_delay=2.0, max_delay=60.0, max_attempts=5, retry_after_header=15.0) == 15.0
    assert compute_retry_delay(1, base_delay=2.0, max_delay=60.0, max_attempts=5, retry_after_header=120.0) == 60.0
    # Capped at max_delay on high attempt
    assert compute_retry_delay(4, base_delay=10.0, max_delay=60.0, max_attempts=5) == 60.0
    # Exhausted attempts return -1.0
    assert compute_retry_delay(5, base_delay=2.0, max_delay=60.0, max_attempts=5) == -1.0
    assert compute_retry_delay(-1, base_delay=2.0, max_delay=60.0, max_attempts=5) == -1.0


# --- PR-107: Multi-currency settlement reconciliation tests ---
def test_reconcile_batch_balanced_and_imbalanced():
    entries = [
        {"amount": 1000.0, "currency": "NOK", "fx_rate_to_usd": 0.095, "fee_bps": 100, "entry_type": "credit"},  # 95.00 USD, fee 0.95
        {"amount": 50.0, "currency": "EUR", "fx_rate_to_usd": 1.10, "fee_bps": 200, "entry_type": "credit"},     # 55.00 USD, fee 1.10
        {"amount": -20.0, "currency": "USD", "fx_rate_to_usd": 1.0, "fee_bps": 100, "entry_type": "credit"},     # skipped (<= 0)
        {"amount": 150.0, "currency": "USD", "fx_rate_to_usd": 1.0, "fee_bps": 500, "entry_type": "debit"},      # 150.00 USD, fee 0.0
    ]
    res = reconcile_batch(entries)
    assert res["total_credits_usd"] == 150.00
    assert res["total_debits_usd"] == 150.00
    assert res["total_fees_usd"] == 2.05
    assert res["net_settlement_usd"] == -2.05
    assert res["imbalance_usd"] == 0.00
    assert res["status"] == "BALANCED"

    imbalanced = reconcile_batch([
        {"amount": 100.0, "currency": "USD", "fx_rate_to_usd": 1.0, "fee_bps": 50, "entry_type": "credit"},
        {"amount": 80.0, "currency": "USD", "fx_rate_to_usd": 1.0, "fee_bps": 0, "entry_type": "debit"},
    ])
    assert imbalanced["status"] == "IMBALANCED"
    assert imbalanced["imbalance_usd"] == 20.00
    assert imbalanced["net_settlement_usd"] == 19.50
