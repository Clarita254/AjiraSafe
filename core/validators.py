
import re
import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat
from django.core.exceptions import ValidationError
from django.core.validators import (
    URLValidator,
    validate_email,
)


def validate_required_text(value, field_name, min_length=1, max_length=None):
    """
    Validate a required text field.
    """
    value = (value or "").strip()

    if not value:
        raise ValidationError(f"{field_name} is required.")

    if len(value) < min_length:
        raise ValidationError(
            f"{field_name} must contain at least {min_length} characters."
        )

    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            f"{field_name} cannot exceed {max_length} characters."
        )

    return value


def validate_optional_text(value, field_name, max_length):
    """
    Validate an optional text field and its maximum length.
    """
    value = (value or "").strip()

    if len(value) > max_length:
        raise ValidationError(
            f"{field_name} cannot exceed {max_length} characters."
        )

    return value


def validate_job_url(value, field_name="Job URL"):
    """
    Validate an optional URL.
    Only HTTP and HTTPS URLs are accepted.
    """
    value = (value or "").strip()

    if not value:
        return value

    if len(value) > 2048:
        raise ValidationError(f"{field_name} cannot exceed 2048 characters.")

    validator = URLValidator(schemes=["http", "https"])

    try:
        validator(value)
    except ValidationError:
        raise ValidationError(
            f"Enter a valid {field_name} beginning with http:// or https://."
        )

    return value


def validate_job_assessment_data(data):
    """
    Validate the main job advertisement information before ML assessment.
    Returns cleaned data if all validations pass.
    """
    cleaned_data = {}

    # Required fields
    cleaned_data["title"] = validate_required_text(
        data.get("title"),
        "Job title",
        min_length=3,
        max_length=255
    )

    cleaned_data["company_name"] = validate_required_text(
        data.get("company_name"),
        "Company name",
        min_length=2,
        max_length=255
    )

    cleaned_data["description"] = validate_required_text(
        data.get("description"),
        "Job description",
        min_length=50,
        max_length=10000
    )

    # Optional fields
    cleaned_data["location"] = validate_optional_text(
        data.get("location"),
        "Location",
        150
    )

    cleaned_data["job_url"] = validate_job_url(
        data.get("job_url")
    )

    # Structured fields and their allowed values
    allowed_choices = {
        "employment_type": [
            "Full-time",
            "Part-time",
            "Contract",
            "Temporary",
            "Internship",
            "Other",
        ],
        "required_experience": [
            "Entry-level",
            "Internship",
            "1-3 years",
            "3-5 years",
            "5+ years",
            "Other",
        ],
    }

    for field, choices in allowed_choices.items():
        value = (data.get(field) or "").strip()

        if value and value not in choices:
            raise ValidationError(
                f"Please select a valid option for {field.replace('_', ' ')}."
            )

        cleaned_data[field] = value

    # Optional structured text fields
    optional_limits = {
        "salary_range": 150,
        "required_education": 255,
        "department": 150,
        "industry": 150,
        "function": 150,
    }

    for field, max_length in optional_limits.items():
        cleaned_data[field] = validate_optional_text(
            data.get(field),
            field.replace("_", " ").title(),
            max_length
        )

    # Validate Yes/No fields rather than treating invalid values as False.
    boolean_fields = [
        "telecommuting",
        "has_company_logo",
        "has_questions",
    ]

    for field in boolean_fields:
        value = (data.get(field) or "").strip().lower()

        if value not in ["yes", "no"]:
            raise ValidationError(
                f"Please select Yes or No for {field.replace('_', ' ')}."
            )

        cleaned_data[field] = value == "yes"

    return cleaned_data


def validate_employer_data(data):
    cleaned_data = {}

    # Validate the employer's official website.
    cleaned_data["website"] = validate_job_url(
        data.get("website"),
        field_name="Employer website"
    )

    # Validate company registration number.
    cleaned_data["company_reg_number"] = validate_optional_text(
        data.get("company_reg_number"),
        "Company registration number",
        255
    )

    # Validate contact email.
    contact_email = (data.get("contact_email") or "").strip()

    if len(contact_email) > 150:
        raise ValidationError(
            "Contact email cannot exceed 150 characters."
        )

    if contact_email:
        try:
            validate_email(contact_email)
        except ValidationError:
            raise ValidationError(
                "Please enter a valid employer contact email address."
            )

    cleaned_data["contact_email"] = contact_email

    
    # Validate international contact phone number.
    contact_phone = (data.get("contact_phone") or "").strip()

    if contact_phone:
        try:
            # The browser submits the complete international number,
            # for example +254722681945.
            parsed_number = phonenumbers.parse(contact_phone, None)

            # Reject numbers that are not valid for their country.
            if not phonenumbers.is_valid_number(parsed_number):
                raise ValidationError(
                    "Please enter a valid contact phone number for the selected country."
                )

            # Store the number in standard international E.164 format.
            contact_phone = phonenumbers.format_number(
                parsed_number,
                PhoneNumberFormat.E164
            )

        except NumberParseException:
         raise ValidationError(
       "Please enter a valid international contact phone number, including its country code."
    )

    cleaned_data["contact_phone"] = contact_phone

    return cleaned_data


   
def validate_employer_verification_data(data, valid_statuses):
    """
    Validate administrator-submitted employer verification decisions.
    """

    cleaned_data = {}

    # Validate the employer verification status.
    verification_status = (
        data.get("verification_status") or ""
    ).strip().lower()

    if verification_status not in valid_statuses:
        raise ValidationError(
            "Please select a valid employer verification status."
        )

    cleaned_data["verification_status"] = verification_status

    # Validate internal administrator remarks.
    cleaned_data["remarks"] = validate_required_text(
        data.get("remarks"),
        "Administrator remarks",
        min_length=5,
        max_length=2000
    )

    # Validate the applicant-facing review summary.
    cleaned_data["applicant_review_summary"] = validate_required_text(
        data.get("applicant_review_summary"),
        "Applicant-facing review summary",
        min_length=10,
        max_length=1000
    )

    return cleaned_data
 