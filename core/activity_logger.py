
import re

from .models import SystemActivity


# Sensitive metadata keys whose values must never be stored.
SENSITIVE_KEYS = (
    "password",
    "passwd",
    "token",
    "secret",
    "authorization",
    "cookie",
    "session",
    "credential",
    "apikey",
    "accesskey",
    "privatekey",
    "csrf",
    "otp",
    "pin",
)


def sanitize_metadata(value):
    """
    Recursively removes sensitive values from activity metadata.
    Supports dictionaries, lists and nested structures.
    """

    if isinstance(value, dict):
        cleaned = {}

        for key, item in value.items():
            normalized_key = re.sub(
                r"[^a-z0-9]",
                "",
                str(key).lower()
            )

            if any(
                sensitive in normalized_key
                for sensitive in SENSITIVE_KEYS
            ):
                cleaned[key] = "[REDACTED]"
            else:
                cleaned[key] = sanitize_metadata(item)

        return cleaned

    if isinstance(value, list):
        return [
            sanitize_metadata(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            sanitize_metadata(item)
            for item in value
        ]

    return value


def log_system_activity(
    actor,
    category,
    action,
    description="",
    target_type="",
    target_id="",
    previous_value="",
    new_value="",
    metadata=None
):
    """
    Records an important event performed by a user
    or administrator in the AjiraSafe system.

    Sensitive metadata values are redacted before storage.
    """

    # Validate activity category.
    if category not in SystemActivity.Category.values:
        raise ValueError(
            f"Invalid system activity category: {category}"
        )

    # Validate action.
    if not isinstance(action, str) or not action.strip():
        raise ValueError(
            "System activity action cannot be empty."
        )

    if len(action.strip()) > 100:
        raise ValueError(
            "System activity action cannot exceed 100 characters."
        )

    # Only associate an authenticated Django user.
    if actor is not None and not getattr(
        actor,
        "is_authenticated",
        False
    ):
        actor = None

    # Preserve the username even if the account is
    # subsequently deleted.
    actor_username = (
        actor.get_username()
        if actor is not None
        else "System"
    )

    # Validate metadata structure.
    if metadata is None:
        metadata = {}

    if not isinstance(metadata, dict):
        raise ValueError(
            "Activity metadata must be a dictionary."
        )

    # Remove sensitive information recursively.
    safe_metadata = sanitize_metadata(metadata)

    # Create the activity record.
    activity = SystemActivity.objects.create(
        actor=actor,
        actor_username=actor_username,
        category=category,
        action=action.strip(),
        description=str(description or ""),
        target_type=str(target_type or ""),
        target_id=str(target_id) if target_id else "",
        previous_value=(
            str(previous_value)
            if previous_value is not None
            else ""
        ),
        new_value=(
            str(new_value)
            if new_value is not None
            else ""
        ),
        metadata=safe_metadata
    )

    return activity
