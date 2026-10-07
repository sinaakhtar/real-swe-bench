"""Step 0: Initialize a realistic target Git repository and simulated GitHub merged PR history.

Creates `target_repo/` (an enterprise order & settlement Python library `orderflow`) with:
- A baseline commit (`base_sha`) containing 6 realistic defects / missing features across 3 complexity tiers.
- 8 merged PR branches (`pr-101` .. `pr-108`) with real git commits and unified diffs:
  - PR #101 (Tier 1 - Simple): Fix tax rounding to use Decimal ROUND_HALF_UP in `orderflow/pricing.py`
  - PR #102 (Tier 1 - Simple, semantic duplicate of #101): Fix currency helper rounding with ROUND_HALF_UP in `orderflow/pricing.py`
  - PR #103 (Tier 1 - Simple): Validate Mod-10 Luhn check digit on invoice reference numbers in `orderflow/validators.py`
  - PR #104 (Tier 2 - Medium): Cap stacked percentage coupon discounts at `max_discount_pct` and floor at 0.0 in `orderflow/discounts.py`
  - PR #105 (Tier 2 - Medium, semantic duplicate of #104): Enforce maximum percentage cap when stacking promotional codes in `orderflow/discounts.py`
  - PR #106 (Tier 2 - Medium): Honor `retry_after_header`, cap exponential backoff at `max_delay`, and return -1 past `max_attempts` in `orderflow/webhooks.py`
  - PR #107 (Tier 3 - Complex): Multi-currency batch settlement reconciliation with FX conversion, basis-point fees on credits, and imbalance detection in `orderflow/settlement.py`
  - PR #108 (Low edit-survival PR): Developer asked Opus for a quick helper, then rewrote the entire settlement class manually before merging.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
TARGET_REPO = ROOT_DIR / "target_repo"
DATA_DIR = ROOT_DIR / "data"


BASELINE_FILES: dict[str, str] = {
    "orderflow/__init__.py": '"""Orderflow core pricing, validation, discount, webhook, and settlement package."""\n',
    "orderflow/pricing.py": '''"""Pricing and tax calculation utilities."""

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
''',
    "orderflow/validators.py": '''"""Invoice and payment reference validators."""


def validate_invoice_ref(ref: str) -> bool:
    """Validate an 8-to-12 digit numeric invoice reference with a Mod-10 (Luhn) check digit.

    Rules:
    1. Strip spaces/hyphens; must be all digits and between 8 and 12 characters inclusive.
    2. The last digit must be a valid Luhn Mod-10 check digit computed over the preceding digits
       (doubling every second digit from the right, subtracting 9 if > 9, total sum % 10 == 0).
    """
    cleaned = ref.replace(" ", "").replace("-", "")
    if not cleaned.isdigit():
        return False
    if not (8 <= len(cleaned) <= 12):
        return False
    # BUG (PR-103): Missing Luhn Mod-10 check digit validation!
    return True
''',
    "orderflow/discounts.py": '''"""Promotional coupon stacking and voucher calculation."""


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
''',
    "orderflow/webhooks.py": '''"""Webhook delivery retry and backoff scheduling."""


def compute_retry_delay(
    attempt: int,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    max_attempts: int = 5,
    retry_after_header: float | None = None,
) -> float:
    """Compute the next retry delay in seconds for a failed webhook delivery.

    Rules:
    1. `attempt` is 0-indexed (0 is the first retry). If `attempt >= max_attempts` or `attempt < 0`,
       return -1.0 to indicate retries are exhausted.
    2. Standard exponential backoff is `base_delay * (2 ** attempt)`.
    3. If `retry_after_header` is provided and positive, the delay before capping must be
       `max(exponential_delay, retry_after_header)`.
    4. Cap the returned delay at `max_delay`.
    """
    # BUG (PR-106): Does not check max_attempts, ignores retry_after_header, and forgets max_delay cap.
    return float(base_delay * (2 ** attempt))
''',
    "orderflow/settlement.py": '''"""Multi-currency ledger settlement and reconciliation engine."""

from typing import Any


def reconcile_batch(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconcile a batch of multi-currency ledger entries into USD.

    Each entry dict contains:
      - "amount": float (must be > 0; skip entries with amount <= 0)
      - "currency": str (e.g. "USD", "EUR", "NOK")
      - "fx_rate_to_usd": float (multiplier to convert `amount` to USD)
      - "fee_bps": int | float (basis points fee, e.g. 150 = 1.50%, applied ONLY to "credit" entries)
      - "entry_type": str ("credit" or "debit")

    Calculation rules:
    1. Skip any entry where `amount <= 0`.
    2. For each valid entry, `usd_amount = round(amount * fx_rate_to_usd, 2)`.
    3. If `entry_type == "credit"`:
         - add `usd_amount` to `total_credits_usd`
         - compute `entry_fee = round(usd_amount * (fee_bps / 10000.0), 2)` and add to `total_fees_usd`
       If `entry_type == "debit"`:
         - add `usd_amount` to `total_debits_usd`
         - debits incur 0 fee.
    4. Round `total_credits_usd`, `total_debits_usd`, and `total_fees_usd` to 2 decimals.
    5. `net_settlement_usd = round(total_credits_usd - total_debits_usd - total_fees_usd, 2)`
    6. `imbalance_usd = round(abs(total_credits_usd - total_debits_usd), 2)`
    7. `status = "BALANCED"` if `imbalance_usd < 0.01` else `"IMBALANCED"`.

    Returns dict with keys:
      total_credits_usd, total_debits_usd, total_fees_usd, net_settlement_usd, imbalance_usd, status
    """
    # BUG (PR-107): Stub implementation does not convert FX, deduct fees only on credits, or detect imbalance.
    total = round(sum(e.get("amount", 0.0) for e in entries), 2)
    return {
        "total_credits_usd": total,
        "total_debits_usd": 0.0,
        "total_fees_usd": 0.0,
        "net_settlement_usd": total,
        "imbalance_usd": total,
        "status": "IMBALANCED",
    }
''',
    "tests/test_orderflow.py": '''"""Unit test suite for orderflow."""

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
''',
}


PR_DEFINITIONS = [
    {
        "pr_number": 101,
        "branch": "fix/pr-101-tax-rounding",
        "title": "fix(pricing): use Decimal ROUND_HALF_UP in calculate_tax",
        "author": "dev-anders",
        "merged_at": "2026-10-02T09:15:00Z",
        "file_path": "orderflow/pricing.py",
        "pytest_k": "test_calculate_tax_round_half_up",
        "complexity_tier_ground_truth": "tier_1_simple",
        "user_prompt": (
            "In `orderflow/pricing.py`, `calculate_tax` uses Python's built-in `round()` which "
            "causes round-half-to-even errors on .005 tax amounts (e.g. 100.02 * 0.25 rounds to 25.0 instead of 25.01). "
            "Please fix `calculate_tax` to use `Decimal(str(subtotal)) * Decimal(str(tax_rate))` with `ROUND_HALF_UP` to `0.01`."
        ),
        "new_file_content": '''"""Pricing and tax calculation utilities."""

from decimal import Decimal, ROUND_HALF_UP


def calculate_tax(subtotal: float, tax_rate: float) -> float:
    """Calculate tax amount for a subtotal and tax_rate (e.g., 0.25 for 25%).

    Must round to 2 decimal places using standard commercial rounding (ROUND_HALF_UP),
    NOT Python's default round-half-to-even.
    """
    tax = Decimal(str(subtotal)) * Decimal(str(tax_rate))
    return float(tax.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def round_currency(amount: float) -> float:
    """Round an arbitrary currency float to 2 decimal places using ROUND_HALF_UP."""
    # BUG (PR-102): Uses built-in round() instead of Decimal ROUND_HALF_UP.
    return round(amount, 2)
''',
    },
    {
        "pr_number": 102,
        "branch": "fix/pr-102-currency-rounding",
        "title": "fix(pricing): use Decimal ROUND_HALF_UP in round_currency helper",
        "author": "dev-ingrid",
        "merged_at": "2026-10-02T09:30:00Z",
        "file_path": "orderflow/pricing.py",
        "pytest_k": "test_round_currency_half_up",
        "complexity_tier_ground_truth": "tier_1_simple",
        "user_prompt": (
            "In `orderflow/pricing.py`, `round_currency` uses Python's built-in `round(amount, 2)` which "
            "does banker's rounding instead of commercial rounding on .005 amounts. "
            "Update `round_currency` to use `Decimal(str(amount)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)`."
        ),
        "new_file_content": '''"""Pricing and tax calculation utilities."""

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
    return float(Decimal(str(amount)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
''',
    },
    {
        "pr_number": 103,
        "branch": "feat/pr-103-invoice-luhn",
        "title": "feat(validators): verify Luhn Mod-10 check digit in validate_invoice_ref",
        "author": "dev-henrik",
        "merged_at": "2026-10-02T09:45:00Z",
        "file_path": "orderflow/validators.py",
        "pytest_k": "test_validate_invoice_ref_luhn",
        "complexity_tier_ground_truth": "tier_1_simple",
        "user_prompt": (
            "Fix `validate_invoice_ref` in `orderflow/validators.py`. Right now it only checks length (8-12) and digits, "
            "but skips the Luhn Mod-10 check digit verification. Implement the Luhn check across all digits of `cleaned`."
        ),
        "new_file_content": '''"""Invoice and payment reference validators."""


def validate_invoice_ref(ref: str) -> bool:
    """Validate an 8-to-12 digit numeric invoice reference with a Mod-10 (Luhn) check digit.

    Rules:
    1. Strip spaces/hyphens; must be all digits and between 8 and 12 characters inclusive.
    2. The last digit must be a valid Luhn Mod-10 check digit computed over the preceding digits
       (doubling every second digit from the right, subtracting 9 if > 9, total sum % 10 == 0).
    """
    cleaned = ref.replace(" ", "").replace("-", "")
    if not cleaned.isdigit():
        return False
    if not (8 <= len(cleaned) <= 12):
        return False
    total = 0
    for i, ch in enumerate(reversed(cleaned)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0
''',
    },
    {
        "pr_number": 104,
        "branch": "fix/pr-104-discount-stacking",
        "title": "fix(discounts): cap stacked coupon percentage and floor total at zero",
        "author": "dev-maja",
        "merged_at": "2026-10-02T10:00:00Z",
        "file_path": "orderflow/discounts.py",
        "pytest_k": "test_apply_stacked_discounts_caps_and_floors",
        "complexity_tier_ground_truth": "tier_2_medium",
        "user_prompt": (
            "Fix `apply_stacked_discounts` in `orderflow/discounts.py`. It needs to ignore negative numbers in `pct_coupons`, "
            "cap the summed percentage discount at `max_discount_pct`, subtract `max(0.0, fixed_voucher)`, "
            "and floor the final price at `0.0` rounded to 2 decimal places."
        ),
        "new_file_content": '''"""Promotional coupon stacking and voucher calculation."""


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
    valid_pct = sum(c for c in pct_coupons if c > 0)
    capped_pct = min(valid_pct, max_discount_pct)
    discounted = subtotal * (1.0 - capped_pct) - max(0.0, fixed_voucher)
    return round(max(0.0, discounted), 2)
''',
    },
    {
        "pr_number": 105,
        "branch": "fix/pr-105-coupon-cap-bound",
        "title": "fix(discounts): enforce max_discount_pct cap when stacking promo percentages",
        "author": "dev-lars",
        "merged_at": "2026-10-02T10:15:00Z",
        "file_path": "orderflow/discounts.py",
        "pytest_k": "test_apply_stacked_discounts_caps_and_floors",
        "complexity_tier_ground_truth": "tier_2_medium",
        "user_prompt": (
            "In `orderflow/discounts.py`, `apply_stacked_discounts` lets stacked percentage coupons exceed `max_discount_pct` "
            "and allows negative totals when `fixed_voucher` is large. Filter out negative coupons, cap total percentage at `max_discount_pct`, "
            "and clamp the final total to `>= 0.0`."
        ),
        "new_file_content": '''"""Promotional coupon stacking and voucher calculation."""


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
    total_pct = sum(c for c in pct_coupons if c > 0)
    total_pct = min(total_pct, max_discount_pct)
    discounted = subtotal * (1.0 - total_pct)
    if fixed_voucher > 0:
        discounted -= fixed_voucher
    return round(max(discounted, 0.0), 2)
''',
    },
    {
        "pr_number": 106,
        "branch": "fix/pr-106-webhook-backoff",
        "title": "fix(webhooks): honor Retry-After header, cap max_delay, and check max_attempts",
        "author": "dev-sofie",
        "merged_at": "2026-10-02T10:30:00Z",
        "file_path": "orderflow/webhooks.py",
        "pytest_k": "test_compute_retry_delay",
        "complexity_tier_ground_truth": "tier_2_medium",
        "user_prompt": (
            "Fix `compute_retry_delay` in `orderflow/webhooks.py`. Return `-1.0` if `attempt < 0` or `attempt >= max_attempts`. "
            "Compute `delay = base_delay * (2 ** attempt)`, if `retry_after_header` is provided and positive take `max(delay, retry_after_header)`, "
            "and cap the returned value at `max_delay`."
        ),
        "new_file_content": '''"""Webhook delivery retry and backoff scheduling."""


def compute_retry_delay(
    attempt: int,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    max_attempts: int = 5,
    retry_after_header: float | None = None,
) -> float:
    """Compute the next retry delay in seconds for a failed webhook delivery.

    Rules:
    1. `attempt` is 0-indexed (0 is the first retry). If `attempt >= max_attempts` or `attempt < 0`,
       return -1.0 to indicate retries are exhausted.
    2. Standard exponential backoff is `base_delay * (2 ** attempt)`.
    3. If `retry_after_header` is provided and positive, the delay before capping must be
       `max(exponential_delay, retry_after_header)`.
    4. Cap the returned delay at `max_delay`.
    """
    if attempt < 0 or attempt >= max_attempts:
        return -1.0
    delay = float(base_delay * (2 ** attempt))
    if retry_after_header is not None and retry_after_header > 0:
        delay = max(delay, float(retry_after_header))
    return float(min(delay, max_delay))
''',
    },
    {
        "pr_number": 107,
        "branch": "feat/pr-107-settlement-reconcile",
        "title": "feat(settlement): implement multi-currency FX reconciliation, credit fee bps, and imbalance check",
        "author": "dev-kasper",
        "merged_at": "2026-10-02T10:50:00Z",
        "file_path": "orderflow/settlement.py",
        "pytest_k": "test_reconcile_batch_balanced_and_imbalanced",
        "complexity_tier_ground_truth": "tier_3_complex",
        "user_prompt": (
            "Implement `reconcile_batch` in `orderflow/settlement.py` according to its docstring specification: "
            "skip entries with `amount <= 0`, convert each entry to USD via `round(amount * fx_rate_to_usd, 2)`, "
            "accumulate credits and debits separately, apply `fee_bps` (`round(usd_amount * (fee_bps / 10000.0), 2)`) ONLY to credit entries, "
            "compute `net_settlement_usd = round(total_credits_usd - total_debits_usd - total_fees_usd, 2)`, "
            "`imbalance_usd = round(abs(total_credits_usd - total_debits_usd), 2)`, and set `status` to `'BALANCED'` if `imbalance_usd < 0.01` else `'IMBALANCED'`."
        ),
        "new_file_content": '''"""Multi-currency ledger settlement and reconciliation engine."""

from typing import Any


def reconcile_batch(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconcile a batch of multi-currency ledger entries into USD.

    Each entry dict contains:
      - "amount": float (must be > 0; skip entries with amount <= 0)
      - "currency": str (e.g. "USD", "EUR", "NOK")
      - "fx_rate_to_usd": float (multiplier to convert `amount` to USD)
      - "fee_bps": int | float (basis points fee, e.g. 150 = 1.50%, applied ONLY to "credit" entries)
      - "entry_type": str ("credit" or "debit")

    Calculation rules:
    1. Skip any entry where `amount <= 0`.
    2. For each valid entry, `usd_amount = round(amount * fx_rate_to_usd, 2)`.
    3. If `entry_type == "credit"`:
         - add `usd_amount` to `total_credits_usd`
         - compute `entry_fee = round(usd_amount * (fee_bps / 10000.0), 2)` and add to `total_fees_usd`
       If `entry_type == "debit"`:
         - add `usd_amount` to `total_debits_usd`
         - debits incur 0 fee.
    4. Round `total_credits_usd`, `total_debits_usd`, and `total_fees_usd` to 2 decimals.
    5. `net_settlement_usd = round(total_credits_usd - total_debits_usd - total_fees_usd, 2)`
    6. `imbalance_usd = round(abs(total_credits_usd - total_debits_usd), 2)`
    7. `status = "BALANCED"` if `imbalance_usd < 0.01` else `"IMBALANCED"`.

    Returns dict with keys:
      total_credits_usd, total_debits_usd, total_fees_usd, net_settlement_usd, imbalance_usd, status
    """
    total_credits_usd = 0.0
    total_debits_usd = 0.0
    total_fees_usd = 0.0

    for entry in entries:
        amount = float(entry.get("amount", 0.0))
        if amount <= 0:
            continue
        fx_rate = float(entry.get("fx_rate_to_usd", 1.0))
        usd_amount = round(amount * fx_rate, 2)
        entry_type = entry.get("entry_type", "")

        if entry_type == "credit":
            total_credits_usd = round(total_credits_usd + usd_amount, 2)
            fee_bps = float(entry.get("fee_bps", 0.0))
            fee_usd = round(usd_amount * (fee_bps / 10000.0), 2)
            total_fees_usd = round(total_fees_usd + fee_usd, 2)
        elif entry_type == "debit":
            total_debits_usd = round(total_debits_usd + usd_amount, 2)

    net_settlement_usd = round(total_credits_usd - total_debits_usd - total_fees_usd, 2)
    imbalance_usd = round(abs(total_credits_usd - total_debits_usd), 2)
    status = "BALANCED" if imbalance_usd < 0.01 else "IMBALANCED"

    return {
        "total_credits_usd": total_credits_usd,
        "total_debits_usd": total_debits_usd,
        "total_fees_usd": total_fees_usd,
        "net_settlement_usd": net_settlement_usd,
        "imbalance_usd": imbalance_usd,
        "status": status,
    }
''',
    },
    {
        "pr_number": 108,
        "branch": "refactor/pr-108-custom-audit-trail",
        "title": "refactor(validators): replace invoice validator with custom country IBAN registry",
        "author": "dev-erik",
        "merged_at": "2026-10-02T11:10:00Z",
        "file_path": "orderflow/validators.py",
        "pytest_k": "test_validate_invoice_ref_luhn",
        "complexity_tier_ground_truth": "tier_3_complex",
        "user_prompt": (
            "Add a one-line docstring note in `orderflow/validators.py` mentioning whitespace stripping."
        ),
        # NOTE: In PR-108, the developer asked Opus for a 1-line comment, but then manually rewrote
        # the entire file with an IBAN registry before merging the PR. This simulates low edit survival rate!
        "new_file_content": '''"""Invoice and payment reference validators with custom Nordic IBAN registry."""

SUPPORTED_COUNTRY_PREFIXES = {"NO": 15, "SE": 24, "DK": 18, "FI": 18}


def validate_iban_prefix(iban: str) -> bool:
    cleaned = iban.replace(" ", "").upper()
    prefix = cleaned[:2]
    if prefix not in SUPPORTED_COUNTRY_PREFIXES:
        return False
    return len(cleaned) == SUPPORTED_COUNTRY_PREFIXES[prefix]


def validate_invoice_ref(ref: str) -> bool:
    cleaned = ref.replace(" ", "").replace("-", "")
    return cleaned.isdigit() and (8 <= len(cleaned) <= 12)
''',
    },
]


def run_git(args: list[str], cwd: Path) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return res.stdout.strip()


def setup_target_repo() -> None:
    if TARGET_REPO.exists():
        shutil.rmtree(TARGET_REPO)
    TARGET_REPO.mkdir(parents=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    run_git(["init", "-b", "main"], TARGET_REPO)
    run_git(["config", "user.name", "Orderflow CI"], TARGET_REPO)
    run_git(["config", "user.email", "ci@orderflow.example.com"], TARGET_REPO)

    for rel_path, content in BASELINE_FILES.items():
        full_path = TARGET_REPO / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    run_git(["add", "."], TARGET_REPO)
    run_git(["commit", "-m", "chore: initial baseline orderflow service"], TARGET_REPO)
    base_sha = run_git(["rev-parse", "HEAD"], TARGET_REPO)

    pr_records = []
    for pr in PR_DEFINITIONS:
        run_git(["checkout", "-b", pr["branch"], base_sha], TARGET_REPO)
        target_file = TARGET_REPO / pr["file_path"]
        target_file.write_text(pr["new_file_content"], encoding="utf-8")
        run_git(["add", pr["file_path"]], TARGET_REPO)
        run_git(["commit", "-m", f"{pr['title']} (#{pr['pr_number']})"], TARGET_REPO)
        head_sha = run_git(["rev-parse", "HEAD"], TARGET_REPO)
        diff_text = run_git(["diff", base_sha, head_sha, "--", pr["file_path"]], TARGET_REPO)

        pr_records.append({
            "pr_number": pr["pr_number"],
            "title": pr["title"],
            "author": pr["author"],
            "branch": pr["branch"],
            "base_sha": base_sha,
            "merge_commit_sha": head_sha,
            "merged_at": pr["merged_at"],
            "files_changed": [pr["file_path"]],
            "pytest_target": pr["pytest_k"],
            "complexity_tier_ground_truth": pr["complexity_tier_ground_truth"],
            "user_prompt_ground_truth": pr["user_prompt"],
            "merged_diff": diff_text,
        })

    # Return working tree to base_sha so agent tools inspect the pre-PR baseline state
    run_git(["checkout", "main"], TARGET_REPO)

    out_path = DATA_DIR / "github_merged_prs.json"
    out_path.write_text(json.dumps(pr_records, indent=2), encoding="utf-8")
    print(f"Initialized target_repo at base_sha={base_sha}")
    print(f"Created {len(pr_records)} merged PR branches and wrote {out_path}")


if __name__ == "__main__":
    setup_target_repo()
