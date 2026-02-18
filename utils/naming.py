import re


def sanitize_filename(name: str) -> str:
    """Sanitize logical asset name into a safe, lowercase filename stem.

    Rules are shared across all data loaders and processors to ensure
    that the same logical asset name always maps to the same CSV file
    name on disk.
    """
    s = re.sub(r"[^\w\s-]", "", name).strip()
    s = re.sub(r"[-\s]+", "_", s)
    return s.lower()

