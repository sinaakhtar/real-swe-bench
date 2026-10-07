"""Invoice and payment reference validators."""


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
