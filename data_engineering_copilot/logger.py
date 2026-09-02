"""Logging utilities with safe percentage escaping for stdlib logging compatibility."""


def safe_pct(value: str) -> str:
    """Escape percent signs in a string to prevent TypeError in stdlib logging.

    When logging progress bars or other formatted strings that may contain '%',
    this function replaces '%' with '%%' so that the literal '%' character
    is preserved in the log output while preventing ValueError in standard
    library logging calls.

    Args:
        value: The string that may contain '%' characters.

    Returns:
        The escaped string with '%' replaced by '%%'.
    """
    return value.replace("%", "%%")
