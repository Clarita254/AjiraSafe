from unittest import result

from django.shortcuts import render, redirect

from django.contrib.auth.models import User, Group

from django.contrib.auth import login, authenticate , logout

from django.contrib import messages

from django.core.validators import validate_email

from django.core.exceptions import ValidationError

from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden, JsonResponse
from django.urls import reverse
from django.db import transaction
from django.utils import timezone
from django.views.csrf import csrf_failure as django_csrf_failure

from django.views.decorators.cache import never_cache
from django.db.models import OuterRef, Prefetch, Subquery, Q
from core.models import TestimonialDecisionLog
from django.db.models import Prefetch
from .models import (
    Testimonial,
    JobApplicant,
    SystemAdministrator,
    Employer,
    JobAdvertisement,
    EmployerVerificationRecord,
    SuspiciousJobReport,
    
    MLModel,
    AnalysisResult,
    RiskAssessment,
    Explanation,
)

from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_POST
from .validators import (
    validate_job_assessment_data,
    validate_employer_data,
    validate_employer_verification_data,
)
import re
import json
import math
from decimal import Decimal
from urllib.parse import urlparse
from .ml_service import (
    assess_job,
    explain_job_lime,
    explain_job_tabular_lime,
)

import logging

logger = logging.getLogger(__name__)

# =========================================================
# HOME
# =========================================================

def home(request):

    testimonials = Testimonial.objects.filter(
        is_approved=True
    ).select_related("user")

    testimonial_data = []

    for testimonial in testimonials:

        testimonial_data.append({
            "feedback": testimonial.feedback,
            "display_name": testimonial.user.get_full_name()
                or testimonial.user.username,
            "role": testimonial.role,
            "rating": testimonial.rating,
        })

    return render(
        request,
        "core/home.html",
        {
            "testimonials": testimonial_data
        }
    )

def testimonials(request):
    approved_testimonials = (
        Testimonial.objects
        .filter(is_approved=True)
        .select_related("user")
        .order_by("-created_at", "-id")
    )

    testimonial_data = []

    for testimonial in approved_testimonials:
        testimonial_data.append({
            "feedback": testimonial.feedback,
            "display_name": (
                testimonial.user.get_full_name()
                or testimonial.user.username
            ),
            "role": testimonial.role,
            "rating": testimonial.rating,
        })

    return render(
        request,
        "core/testimonials.html",
        {"testimonials": testimonial_data},
    )


# =========================================================
# ABOUT
# =========================================================

def about(request):

    return render(
        request,
        "core/about.html"
    )


# =========================================================
# TESTIMONIALS
@login_required(login_url="login")
@require_POST
def testimonial_action(request, testimonial_id, action):
    # Only system administrators may manage feedback.
    if not request.user.groups.filter(
        name="System Administrator"
    ).exists():
        return HttpResponseForbidden(
            "You are not authorised to manage feedback."
        )

    administrator = SystemAdministrator.objects.filter(
        user=request.user
    ).first()

    if administrator is None:
        messages.error(
            request,
            "Your administrator profile could not be found."
        )
        return redirect("admin_dashboard")

    if action not in ("review", "approve", "reject"):
        return HttpResponseForbidden("Invalid feedback action.")

    with transaction.atomic():
        testimonial = get_object_or_404(
            Testimonial.objects.select_for_update(),
            pk=testimonial_id
        )

        # Block changes only after a final decision has been made.
        if testimonial.status == Testimonial.Status.APPROVED:
            messages.warning(
                request,
                "A final decision has already been recorded for "
                "this feedback. It cannot be changed."
            )
            return redirect("admin_dashboard")

        # Actions are allowed while pending or under review.
        if testimonial.status not in (
            Testimonial.Status.PENDING,
            Testimonial.Status.UNDER_REVIEW,
        ):
            messages.warning(
                request,
                "This feedback cannot be processed in its current status."
            )
            return redirect("admin_dashboard")

        # Mark as Under Review only when currently pending.
        if action == "review":
            if testimonial.status != Testimonial.Status.PENDING:
                messages.warning(
                    request,
                    "This feedback is already Under Review."
                )
                return redirect("admin_dashboard")

            testimonial.status = Testimonial.Status.UNDER_REVIEW
            testimonial.is_approved = False
            testimonial.reviewed_by = administrator
            testimonial.reviewed_at = timezone.now()
            testimonial.admin_comment = request.POST.get(
                "admin_comment", ""
            ).strip()
            testimonial.save()

            messages.success(
                request,
                "Feedback marked as Under Review."
            )

        # Approve feedback for publication.
        elif action == "approve":
            testimonial.status = Testimonial.Status.APPROVED
            testimonial.is_approved = True
            testimonial.reviewed_by = administrator
            testimonial.reviewed_at = timezone.now()
            testimonial.admin_comment = request.POST.get(
                "admin_comment", ""
            ).strip()
            testimonial.save()

            messages.success(
                request,
                "Feedback approved for publication."
            )

        # Finalise the feedback and keep it private.
        elif action == "reject":
            testimonial.status = Testimonial.Status.APPROVED
            testimonial.is_approved = False
            testimonial.reviewed_by = administrator
            testimonial.reviewed_at = timezone.now()
            testimonial.admin_comment = request.POST.get(
                "admin_comment", ""
            ).strip()
            testimonial.save()

            messages.success(
                request,
                "Feedback finalised and kept private."
            )

    return redirect("admin_dashboard")

# =========================================================
# SERVICES
# =========================================================

def services(request):

    return render(
        request,
        "core/services.html"
    )


# =========================================================
# SIGN UP
# =========================================================

@never_cache
def signup(request):

    # Clear old messages whenever the signup page
    # is opened normally.
    if request.method == "GET":

        list(messages.get_messages(request))

    if request.method == "POST":

        first_name = request.POST.get(
            "first_name",
            ""
        ).strip()

        last_name = request.POST.get(
            "last_name",
            ""
        ).strip()

        email = request.POST.get(
            "email",
            ""
        ).strip().lower()

        phone_country_code = request.POST.get(
            "phone_country_code",
            "+254"
        ).strip()

        phone_number = request.POST.get(
            "phone_number",
            ""
        ).strip()

        password = request.POST.get(
            "password",
            ""
        )

        confirm_password = request.POST.get(
            "confirm_password",
            ""
        )

        context = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone_number": phone_number,
            "phone_country_code": phone_country_code,
        }

        # =================================================
        # FIRST NAME VALIDATION
        # =================================================

        if not first_name:

            messages.error(
                request,
                "Please enter your first name."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        if not re.fullmatch(
            r"[A-Za-z]+",
            first_name
        ):

            messages.error(
                request,
                "First name can only contain letters."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        # =================================================
        # LAST NAME VALIDATION
        # =================================================

        if not last_name:

            messages.error(
                request,
                "Please enter your last name."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        if not re.fullmatch(
            r"[A-Za-z]+",
            last_name
        ):

            messages.error(
                request,
                "Last name can only contain letters."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        # =================================================
        # EMAIL VALIDATION
        # =================================================

        if not email:

            messages.error(
                request,
                "Please enter your email address."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        email_local, separator, email_domain = email.rpartition("@")

        domain_labels = email_domain.split(".")

        valid_domain = (

            bool(separator)

            and len(email) <= 254

            and len(email_domain) <= 253

            and len(domain_labels) >= 2

            and all(
                re.fullmatch(
                    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?",
                    label
                )
                for label in domain_labels
            )

            and re.fullmatch(
                r"[A-Za-z]{2,63}",
                domain_labels[-1]
            )

        )

        try:

            validate_email(email)

            if len(email_local) > 64 or not valid_domain:

                raise ValidationError(
                    "Invalid email domain"
                )

        except ValidationError:

            messages.error(
                request,
                "Please enter a valid email address with a valid domain."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        # =================================================
        # CHECK WHETHER EMAIL ALREADY EXISTS
        # =================================================

        if User.objects.filter(
            email__iexact=email
        ).exists():

            messages.error(
                request,
                "An account with this email already exists."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        # =================================================
        # PHONE NUMBER VALIDATION
        # =================================================

        if not phone_number:

            messages.error(
                request,
                "Please enter your phone number."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        if (

            not re.fullmatch(
                r"\+\d{1,3}",
                phone_country_code
            )

            or not re.fullmatch(
                r"[1-9]\d{8}",
                phone_number
            )

            or len(
                phone_country_code[1:] + phone_number
            ) > 15

        ):

            messages.error(
                request,
                "Enter a valid phone Number using digits only. "
                "Remove the leading 0; the country calling code includes it."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        # =================================================
        # PASSWORD VALIDATION
        # =================================================

        if len(password) < 8:

            messages.error(
                request,
                "Password must contain at least 8 characters."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        if not re.search(
            r"[A-Z]",
            password
        ):

            messages.error(
                request,
                "Password must contain at least one uppercase letter."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        if not re.search(
            r"[a-z]",
            password
        ):

            messages.error(
                request,
                "Password must contain at least one lowercase letter."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        if not re.search(
            r"\d",
            password
        ):

            messages.error(
                request,
                "Password must contain at least one number."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        if not re.search(
            r"[^A-Za-z0-9]",
            password
        ):

            messages.error(
                request,
                "Password must contain at least one special character."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        # =================================================
        # CONFIRM PASSWORD
        # =================================================

        if password != confirm_password:

            messages.error(
                request,
                "The passwords do not match."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

        # =================================================
        # CREATE USER
        # =================================================

        try:

            international_phone_number = (
                f"{phone_country_code}{phone_number}"
            )

            user = User.objects.create_user(
                username=email,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name
            )

            # =================================================
            # ASSIGN JOB APPLICANT ROLE
            # =================================================

            job_applicant_group = Group.objects.get(
                name="Job Applicant"
            )

            user.groups.add(
                job_applicant_group
            )

            # =================================================
            # CREATE JOB APPLICANT PROFILE
            # =================================================

            JobApplicant.objects.create(
                user=user,
                phone_number=international_phone_number
            )

            # =================================================
            # LOG USER IN
            # =================================================

            login(
                request,
                user
            )

            messages.success(
                request,
                f"Welcome to AjiraSafe, {first_name}!"
            )

            # =================================================
            # REDIRECT TO JOB APPLICANT DASHBOARD
            # =================================================

            if user.groups.filter(
                name="Job Applicant"
            ).exists():

                return redirect(
                    "dashboard"
                )

            return redirect(
                "home"
            )

        except Exception:

            messages.error(
                request,
                "Something went wrong while creating your account. "
                "Please try again."
            )

            return render(
                request,
                "core/signup.html",
                context
            )

    return render(
        request,
        "core/signup.html"
    )


# =========================================================
# SIGN IN
# =========================================================

@never_cache
def login_view(request):

    # =====================================================
    # OPEN LOGIN PAGE
    # =====================================================

    if request.method == "GET":

        # Clear old messages whenever the login page
        # is opened normally.
        list(messages.get_messages(request))

        return render(
            request,
            "core/login.html"
        )


    # =====================================================
    # GET LOGIN FORM DATA
    # =====================================================

    email = request.POST.get(
        "email",
        ""
    ).strip().lower()

    password = request.POST.get(
        "password",
        ""
    )

    # =====================================================
    # VALIDATE REQUIRED FIELDS
    # =====================================================

    # These checks are also performed on the server.
    # JavaScript validation on the page is only for
    # immediate user feedback.

    if not email:

        return render(
            request,
            "core/login.html",
            {
                "email_required": True,
            }
        )


    if not password:

        return render(
            request,
            "core/login.html",
            {
                "password_required": True,
                "email": email,
            }
        )


    # =====================================================
    # FIND ACCOUNT BY EMAIL
    # =====================================================

    account = User.objects.filter(
        email__iexact=email
    ).first()

    user = None

    if account is not None:

        user = authenticate(
            request,
            username=account.get_username(),
            password=password
        )


    # =====================================================
    # VALIDATE LOGIN CREDENTIALS
    # =====================================================

    # Unknown email, incorrect password,
    # or inactive account all receive the same
    # generic error message.

    if (
        account is None
        or user is None
        or not user.is_active
    ):

        return render(
            request,
            "core/login.html",
            {
                "invalid_credentials": True,
            }
        )


    # =====================================================
    # DETERMINE USER ROLE
    # =====================================================

    if user.groups.filter(
        name="System Administrator"
    ).exists():

        dashboard_name = "admin_dashboard"


    elif user.groups.filter(
        name="Job Applicant"
    ).exists():

        dashboard_name = "dashboard"


    else:

        return render(
            request,
            "core/login.html",
            {
                "invalid_credentials": True,
            }
        )


    # =====================================================
    # LOG USER IN
    # =====================================================

    login(
        request,
        user
    )


    # Keep authentication limited to the current browser session.
    request.session.set_expiry(0)


    # =====================================================
    # REDIRECT TO APPROPRIATE DASHBOARD
    # =====================================================

    return redirect(
        dashboard_name
    )

# =========================================================
# CSRF FAILURE
# =========================================================

@never_cache
def csrf_failure(
    request,
    reason=""
):

    if request.path_info == reverse("signup"):

        messages.error(
            request,
            "This form expired or could not be verified. Please try again."
        )

        return render(
            request,
            "core/signup.html",
            status=403
        )

    if request.path_info == reverse("login"):

        messages.error(
            request,
            "This form expired or could not be verified. Please try again."
        )

        return render(
            request,
            "core/login.html",
            status=403
        )

    return django_csrf_failure(
        request,
        reason
    )


# =========================================================
# FORGOT PASSWORD
# =========================================================

def forgot_password(request):

    return render(
        request,
        "core/forgot_password.html"
    )


# =========================================================
# JOB APPLICANT DASHBOARD
# =========================================================

@login_required(login_url="login")
@never_cache
def dashboard(request):
    # Redirect users who are not job applicants.
    if not request.user.groups.filter(name="Job Applicant").exists():
        if request.user.groups.filter(name="System Administrator").exists():
            return redirect("admin_dashboard")
        return redirect("home")

    # Retrieve the applicant profile.
    try:
        applicant = request.user.job_applicant
    except JobApplicant.DoesNotExist:
        messages.error(request, "Your applicant profile could not be found.")
        return redirect("home")

    # Handle applicant feedback submissions.
    if request.method == "POST" and "feedback_type" in request.POST:
        category = request.POST.get("feedback_type", "").strip()
        feedback_text = request.POST.get("feedback", "").strip()
        rating_value = request.POST.get("rating", "").strip()

        valid_categories = [
            choice[0] for choice in Testimonial.Category.choices
        ]

        if category not in valid_categories:
            messages.error(request, "Please select a valid feedback category.")

        elif not feedback_text:
            messages.error(request, "Please enter your feedback.")

        else:
            try:
                rating = int(rating_value)
            except (TypeError, ValueError):
                rating = 0

            if not 1 <= rating <= 5:
                messages.error(
                    request,
                    "Please select a rating between 1 and 5."
                )
            else:
                Testimonial.objects.create(
                    user=request.user,
                    category=category,
                    feedback=feedback_text,
                    rating=rating,
                    status=Testimonial.Status.PENDING,
                    is_approved=False,
                )

                messages.success(
                    request,
                    "Your feedback has been submitted successfully. "
                    "You can track its status in My Feedback Submissions."
                )
                return redirect("dashboard")

    # Display assessment submissions and their validation errors on the form.
    is_assessment_submission = (
        request.method == "POST" and "assessment_submit" in request.POST
    )
    active_tab = (
        "assess-job"
        if is_assessment_submission
        else request.GET.get("tab", "overview")
    )
    assessment_form_data = (
        request.POST if is_assessment_submission else {}
    )
    assessment_result = None

    if active_tab == "assess-job":
        assessment_result = request.session.pop(
            "latest_assessment_result", None
        )

    # Handle job assessment submissions.
    if is_assessment_submission:

        # Job advertisement details.
        title = request.POST.get("title", "").strip()
        company_name = request.POST.get("company_name", "").strip()
        location = request.POST.get("location", "").strip()
        job_url = request.POST.get("job_url", "").strip()
        description = request.POST.get("description", "").strip()
        telecommuting = request.POST.get("telecommuting", "").strip().lower() == "yes"
        has_company_logo = request.POST.get("has_company_logo", "").strip().lower() == "yes"
        has_questions = request.POST.get("has_questions", "").strip().lower() == "yes"  
        # Employer information for independent verification.
        employer_website = request.POST.get("website", "").strip()
        company_reg_number = request.POST.get(
            "company_reg_number", ""
        ).strip()
        contact_email = request.POST.get("contact_email", "").strip()
        contact_phone = request.POST.get("contact_phone", "").strip()

        # Additional structured job fields.
        salary_range = request.POST.get("salary_range", "").strip()
        employment_type = request.POST.get("employment_type", "").strip()
        required_experience = request.POST.get(
            "required_experience", ""
        ).strip()
        required_education = request.POST.get(
            "required_education", ""
        ).strip()
        industry = request.POST.get("industry", "").strip()
        department = request.POST.get("department", "").strip()
        job_function = request.POST.get("function", "").strip()

        if not title or not company_name or not description:
            messages.error(
                request,
                "Please provide the job title, employer name, "
                "and job description."
            )

        else:
            try:
                # Validate job advertisement details and retrieve cleaned values.
                validated_data = validate_job_assessment_data(request.POST)
                # Validate employer information separately.
                validated_employer_data = validate_employer_data(request.POST)

# Use the cleaned employer information.
                employer_website = validated_employer_data["website"]
                company_reg_number = validated_employer_data["company_reg_number"]
                contact_email = validated_employer_data["contact_email"]
                contact_phone = validated_employer_data["contact_phone"]
                title = validated_data["title"]
                company_name = validated_data["company_name"]
                location = validated_data["location"]
                job_url = validated_data["job_url"]
                description = validated_data["description"]

                telecommuting = validated_data["telecommuting"]
                has_company_logo = validated_data["has_company_logo"]
                has_questions = validated_data["has_questions"]

                salary_range = validated_data["salary_range"]
                employment_type = validated_data["employment_type"]
                required_experience = validated_data["required_experience"]
                required_education = validated_data["required_education"]
                industry = validated_data["industry"]
                department = validated_data["department"]
                job_function = validated_data["function"]

                # Validate the employer's official website separately.
                if employer_website:
                    parsed_employer_url = urlparse(employer_website)

                    if (
                        parsed_employer_url.scheme not in ("http", "https")
                        or not parsed_employer_url.hostname
                    ):
                        raise ValueError(
                            "Please enter a valid employer website URL."
                        )

                # Prepare the job data for the machine-learning model.
                job_data = {
                    "title": title,
                    "description": description,
                    "location": location,
                    "company_profile": "",
                    "requirements": "",
                    "benefits": "",
                    "department": department,
                    "salary_range": salary_range,
                    "telecommuting": telecommuting,
                    "has_company_logo": has_company_logo,
                    "has_questions": has_questions,
                    "employment_type": employment_type,
                    "required_experience": required_experience,
                    "required_education": required_education,
                    "industry": industry,
                    "function": job_function,
                }

                # Run the job risk assessment.
                prediction_data = assess_job(job_data)

                # Generate explanations. Explanation failures should not
                # prevent the main assessment from being saved.
                import logging

                logger = logging.getLogger(__name__)

                text_lime_data = None
                structured_lime_data = None

                try:
                    text_lime_data = explain_job_lime(
                        job_data,
                        num_features=8,
                        num_samples=500,
                    )
                except Exception:
                    logger.exception(
                        "LIME text explanation failed."
                    )

                try:
                    structured_lime_data = explain_job_tabular_lime(
                        job_data,
                        num_features=11,
                        num_samples=1000,
                    )
                except Exception:
                    logger.exception(
                        "LIME structured explanation failed."
                    )

                # Extract and format the model results.
                probability = Decimal(
                    str(prediction_data["fraud_probability"])
                ).quantize(Decimal("0.0001"))

                risk_score = Decimal(
                    str(prediction_data["risk_score"])
                ).quantize(Decimal("0.01"))

                prediction = prediction_data["prediction"]
                risk_level = prediction_data["risk_level"]

                fraud_probability_percent = float(probability) * 100

                if prediction == "fraudulent":
                    explanation_text = (
                        "The Random Forest model classified this job "
                        "advertisement as fraudulent and estimated a fraud "
                        f"probability of {fraud_probability_percent:.2f}%. "
                        "AjiraSafe's risk calculation places it in the "
                        f"{str(risk_level).lower()} risk category. "
                        "This is a model-based assessment, not independent "
                        "proof of fraud. Employer verification is a "
                        "separate check."
                    )
                else:
                    explanation_text = (
                        "The Random Forest model classified this job "
                        "advertisement as legitimate and estimated a fraud "
                        f"probability of {fraud_probability_percent:.2f}%. "
                        "AjiraSafe's risk calculation places it in the "
                        f"{str(risk_level).lower()} risk category. "
                        "This classification does not confirm that the "
                        "employer is genuine; employer verification is "
                        "separate."
                    )

                explanation_payload = {
                    "text_lime": text_lime_data,
                    "structured_lime": structured_lime_data,
                }

                available_methods = []

                if text_lime_data is not None:
                    available_methods.append("LIME text")

                if structured_lime_data is not None:
                    available_methods.append("LIME structured features")

                explanation_method = (
                    " + ".join(available_methods)
                    if available_methods
                    else "Model prediction with descriptive indicators"
                )

                # Save the assessment and related records together.
                with transaction.atomic():
                             # Retrieve an existing employer or create a new employer.
                    employer = None

                    # First, check using the company registration number.
                    if company_reg_number:
                        employer = Employer.objects.filter(
                            company_name__iexact=company_name,
                            company_reg_number__iexact=company_reg_number,
                        ).first()

                    # If no match is found, check using the employer's official website.
                    if employer is None and employer_website:
                        employer = Employer.objects.filter(
                            company_name__iexact=company_name,
                            website__iexact=employer_website,
                        ).first()

                    # If no website was supplied, check for an existing record
                    # with the same company name and no recorded website.
                    if employer is None and not employer_website:
                        employer = Employer.objects.filter(
                            company_name__iexact=company_name,
                            website="",
                        ).first()

                    # Create a new employer if no existing record matches.
                    if employer is None:
                        employer = Employer.objects.create(
                            company_name=company_name,
                            website=employer_website,
                            company_reg_number=company_reg_number,
                            contact_email=contact_email,
                            contact_phone=contact_phone,
                            email_domain=None,
                        )

                    # Check whether this employer already has verification history.
                    latest_verification = (
                        employer.verification_records.order_by(
                            "-verification_date",
                            "-created_at",
                        ).first()
                    )

                    # Only create a Pending Verification record if no history exists.
                    if latest_verification is None:
                        EmployerVerificationRecord.objects.create(
                            employer=employer,
                            verified_by=None,
                            verification_status=(
                                EmployerVerificationRecord.VerificationStatus.PENDING
                            ),
                            remarks="Awaiting administrator verification.",
                        )
                    job_status = (
                        JobAdvertisement.Status.FLAGGED
                        if prediction == "fraudulent"
                        else JobAdvertisement.Status.ANALYZED
                    )

                    job = JobAdvertisement.objects.create(
                        applicant=applicant,
                        employer=employer,
                        title=title,
                        description=description,
                        location=location,
                        department=department,
                        salary_range=salary_range,
                        company_profile="",
                        requirements="",
                        benefits="",
                        telecommuting=telecommuting,
                        has_company_logo=has_company_logo,
                        has_questions=has_questions,
                        employment_type=employment_type,
                        required_experience=required_experience,
                        required_education=required_education,
                        industry=industry,
                        function=job_function,
                        status=job_status,
                    )

                    ml_model = MLModel.objects.get(
                        model_name="AjiraSafe Random Forest",
                        model_version="1.0.0",
                    )

                    combined_text = " ".join(
                        str(job_data.get(field, "") or "")
                        for field in (
                            "title",
                            "company_profile",
                            "description",
                            "requirements",
                            "benefits",
                        )
                    ).strip()

                    result = AnalysisResult.objects.create(
                        job=job,
                        model=ml_model,
                        prediction=prediction,
                        prediction_probability=probability,
                        processed_text_length=len(combined_text),
                    )

                    RiskAssessment.objects.create(
                        result=result,
                        risk_score=risk_score,
                        risk_level=risk_level,
                    )

                    Explanation.objects.create(
                        result=result,
                        explanation_text=explanation_text,
                        key_contribution_features=json.dumps(
                            explanation_payload
                        ),
                        method=explanation_method,
                    )

                    # Store the result for the Assess Job tab.
                    assessment_result = {
                        "job_id": job.job_id,
                        "title": title,
                        "company_name": company_name,
                        "prediction": prediction,
                        "fraud_probability": fraud_probability_percent,
                        "risk_score": float(risk_score),
                        "risk_level": risk_level,
                "employer_verification": (
    job.employer.verification_records
    .order_by("-verification_date", "-created_at")
    .first()
    .get_verification_status_display()
    if job.employer.verification_records.exists()
    else "Pending Verification"
),
                        "explanation": explanation_text,
                        "text_lime_features": (
                            text_lime_data.get("features", [])
                            if text_lime_data
                            else []
                        ),
                        "structured_lime_features": (
                            structured_lime_data.get("features", [])
                            if structured_lime_data
                            else []
                        ),
                        "lime_available": (
                            text_lime_data is not None
                            or structured_lime_data is not None
                        ),
                    }

                # Redirect to assessment results after success.
                request.session["latest_assessment_result"] = (
                    assessment_result
                )
                request.session.modified = True

                messages.success(
                    request,
                    "Job assessment completed and saved successfully."
                )

                return redirect(
                    "assessment_results",
                    result_id=result.result_id,
                )

            except MLModel.DoesNotExist:
                messages.error(
                    request,
                    "The registered AjiraSafe model version was not "
                    "found. Please check the MLModel record."
                )
            except ValidationError as exc:
                messages.error(request, " ".join(exc.messages))
            except ValueError as exc:
                messages.error(request, str(exc))

            except Exception:
                logging.getLogger(__name__).exception(
                    "Job assessment failed."
                )
                messages.error(
                    request,
                    "The assessment could not be completed. Please check "
                    " the server logs or contact the administrator."
                )

    # Show only feedback submitted by the logged-in applicant.
    feedback_submissions = (
        Testimonial.objects
        .filter(user=request.user)
        .order_by("-created_at")
    )

    # Load saved assessment history.
    analysis_history = (
        AnalysisResult.objects
        .filter(job__applicant=applicant)
        .select_related("job", "job__employer", "model")
        .order_by("-pk")
    )

    history_rows = []

    for analysis in analysis_history:
        risk_assessment = (
            RiskAssessment.objects
            .filter(result=analysis)
            .first()
        )

        history_rows.append({
            "id": analysis.result_id,
            "title": analysis.job.title,
            "employer": str(analysis.job.employer),
            "analysis_date": analysis.analysis_date,
            "prediction": analysis.prediction,
            "risk_score": (
                risk_assessment.risk_score
                if risk_assessment else None
            ),
            "risk_level": (
                risk_assessment.risk_level
                if risk_assessment else "Unavailable"
            ),
        })

    # Retrieve the latest saved assessment.
    latest_assessment = analysis_history.first()

    # Dashboard summary cards.
    jobs_analyzed = analysis_history.count()

    high_risk_count = RiskAssessment.objects.filter(
        result__job__applicant=applicant,
        risk_level__in=[
            RiskAssessment.RiskLevel.HIGH,
            RiskAssessment.RiskLevel.VERY_HIGH,
        ],
    ).count()

    reports_count = SuspiciousJobReport.objects.filter(
        applicant=applicant
    ).count()

    pending_feedback_count = feedback_submissions.filter(
        status=Testimonial.Status.PENDING
    ).count()

    # Preserve all required dashboard context variables.
    context = {
        "active_tab": active_tab,
        "assessment_form_data": assessment_form_data,
        "jobs_analyzed": jobs_analyzed,
        "high_risk_count": high_risk_count,
        "reports_count": reports_count,
        "pending_feedback_count": pending_feedback_count,
        "feedback_submissions": feedback_submissions,
        "analysis_history": history_rows,
        "latest_assessment": latest_assessment,

    }

    return render(
        request,
        "core/Applicantdashboard.html",
        context,
    )

def build_applicant_lime_summary(lime_data, structured=False):
    """
    Convert saved LIME output into a ranked, applicant-friendly table.

    Features are ranked by absolute LIME weight, with the direction
    derived from the sign of the actual saved weight.
    Raw numerical weights are not included in the returned rows.
    """
    empty_summary = {
        "available": False,
        "ranked_features": [],
    }

    if not isinstance(lime_data, dict):
        return empty_summary

    features = lime_data.get("features", [])

    if not isinstance(features, list):
        return empty_summary

    def readable_feature(raw_feature):
        raw_feature = str(raw_feature).strip()

        if not structured:
            # Keep the actual text term or phrase identified by LIME.
            return raw_feature

        if "=" in raw_feature:
            field, value = raw_feature.split("=", 1)

            field_label = field.replace("_", " ").strip().capitalize()
            value = value.strip()

            if value == "__MISSING__":
                value = "Not specified in the advertisement"
            elif value == "__UNSEEN__":
                value = "Unrecognised category"

            return f"{field_label}: {value}"

        return raw_feature.replace("_", " ").strip().capitalize()

    contributions = []

    for item in features:
        if not isinstance(item, dict):
            continue

        raw_feature = item.get("feature")
        raw_weight = item.get("weight")

        if raw_feature is None:
            continue

        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue

        if not math.isfinite(weight):
            continue

        feature_label = readable_feature(raw_feature)

        if not feature_label:
            continue

        if weight > 0:
            direction = "Increases the estimated fraud probability"
        elif weight < 0:
            direction = "Decreases the estimated fraud probability"
        else:
            direction = "No directional contribution"

        contributions.append((
            abs(weight),
            {
                "feature": feature_label,
                "direction": direction,
            },
        ))

    # Rank by the magnitude of the actual LIME contribution.
    contributions.sort(
        key=lambda contribution: contribution[0],
        reverse=True,
    )

    ranked_features = [
        {
            "rank": rank,
            **feature_data,
        }
        for rank, (_, feature_data) in enumerate(
            contributions,
            start=1,
        )
    ]

    return {
        "available": bool(ranked_features),
        "ranked_features": ranked_features,
    }

@login_required(login_url="login")
@never_cache
def assessment_results(request, result_id):
    # Only job applicants can view applicant assessment results.
    if not request.user.groups.filter(
        name="Job Applicant"
    ).exists():
        if request.user.groups.filter(
            name="System Administrator"
        ).exists():
            return redirect("admin_dashboard")

        return redirect("home")

    # Retrieve the applicant profile.
    try:
        applicant = request.user.job_applicant
    except JobApplicant.DoesNotExist:
        messages.error(
            request,
            "Your applicant profile could not be found."
        )
        return redirect("home")

    # Retrieve only an assessment belonging to this applicant.
    result = get_object_or_404(
        AnalysisResult.objects.select_related(
            "job",
            "job__employer",
            "model",
        ).prefetch_related("explanations"),
        result_id=result_id,
        job__applicant=applicant,
    )

    # Retrieve the risk assessment, if one exists.
    risk_assessment = getattr(
        result,
        "risk_assessment",
        None,
    )

    # Retrieve and safely parse the saved LIME explanation.
    explanation = result.explanations.first()
    explanation_payload = {}

    if explanation and explanation.key_contribution_features:
        try:
            parsed_payload = json.loads(
                explanation.key_contribution_features
            )

            if isinstance(parsed_payload, dict):
                explanation_payload = parsed_payload

        except (json.JSONDecodeError, TypeError):
            explanation_payload = {}

    # Build applicant-friendly summaries from the actual LIME output.
    # These summaries omit the technical weights and raw JSON.
    text_lime_summary = build_applicant_lime_summary(
        explanation_payload.get("text_lime"),
        structured=False,
    )

    structured_lime_summary = build_applicant_lime_summary(
        explanation_payload.get("structured_lime"),
        structured=True,
    )

    # Convert the stored probability to a percentage for display.
    probability_percent = (
        float(result.prediction_probability) * 100
        if result.prediction_probability is not None
        else None
    )

    # Retrieve the stored risk level safely.
    risk_level = (
        risk_assessment.risk_level
        if risk_assessment is not None
        else None
    )

    # Convert the internal risk level into a user-friendly display label.
    risk_level_display = (
        str(risk_level).replace("_", " ").title()
        if risk_level
        else "Unavailable"
    )

    # Provide a risk interpretation based on the assessment risk level.
    risk_interpretations = {
        "low": (
            "This listing has a low assessed risk. "
            "Nevertheless, verify the employer and "
            "review the job details before proceeding."
        ),

        "medium": (
            "This listing has a medium assessed risk. "
            "Exercise caution and independently verify "
            "the employer before proceeding."
        ),

        "high": (
            "This listing has a high assessed risk. "
            "Carefully verify the employer and job details "
            "before sharing personal information."
        ),

        "very_high": (
            "This listing has a very high assessed risk. "
            "Avoid sending money or sensitive personal "
            "information, and independently verify the "
            "employer before proceeding."
        ),
    }

    risk_interpretation = risk_interpretations.get(
        str(risk_level).lower() if risk_level else "",
        "The risk level could not be determined. "
        "Please independently verify the employer."
    )
     # Retrieve the latest employer verification decision and administrator details.
    latest_verification = (
        result.job.employer.verification_records
        .select_related("verified_by__user")
        .order_by("-verification_date", "-created_at")
        .first()
    )

    # Display the actual database verification status.
    employer_verification = (
        latest_verification.get_verification_status_display()
        if latest_verification
        else "Pending Verification"
    )

    employer_verification_status = (
        latest_verification.verification_status
        if latest_verification
        else EmployerVerificationRecord.VerificationStatus.PENDING
    )

    # Retrieve the applicant-facing verification summary.
    employer_review_summary = (
        latest_verification.applicant_review_summary
        if latest_verification and latest_verification.applicant_review_summary
        else "No applicant-facing review summary was provided for this decision."
    )

    # Retrieve the verification decision date.
    employer_verification_date = (
        latest_verification.verification_date
        if latest_verification
        else None
    )

    # Retrieve the administrator who made the decision.
    employer_verification_administrator = None

    if latest_verification and latest_verification.verified_by:
        admin_user = latest_verification.verified_by.user
        employer_verification_administrator = (
            admin_user.get_full_name() or admin_user.username
        )

    # Prepare the template context.
    context = {
        "result": result,
        "job": result.job,
        "risk_assessment": risk_assessment,
        "explanation": explanation,
        "text_lime_summary": text_lime_summary,
        "structured_lime_summary": structured_lime_summary,
        "fraud_probability_percent": probability_percent,
        "risk_level_display": risk_level_display,
        "risk_interpretation": risk_interpretation,
        "employer_verification": employer_verification,
        "employer_verification_status": employer_verification_status,
        "employer_review_summary": employer_review_summary,
        "employer_verification_date": employer_verification_date,
        "employer_verification_administrator": employer_verification_administrator,
    }

    return render(
        request,
        "core/assessment_results.html",
        context,
    )
@login_required(login_url="login")
@never_cache
def admin_lime_explanation(request, result_id):
    # Only system administrators can access technical LIME details.
    if not request.user.groups.filter(
        name="System Administrator"
    ).exists():
        return HttpResponseForbidden(
            "You are not authorised to view technical LIME explanations."
        )

    result = get_object_or_404(
        AnalysisResult.objects.select_related(
            "job",
            "job__employer",
            "model",
        ).prefetch_related("explanations"),
        result_id=result_id,
    )

    explanation = result.explanations.first()
    lime_payload = {}

    if explanation and explanation.key_contribution_features:
        try:
            parsed_payload = json.loads(
                explanation.key_contribution_features
            )

            if isinstance(parsed_payload, dict):
                lime_payload = parsed_payload

        except (json.JSONDecodeError, TypeError):
            lime_payload = {}

    # Format the saved LIME data for administrator display.
    raw_lime_json = json.dumps(
        lime_payload,
        indent=2,
        ensure_ascii=False,
    )

    # Retrieve the saved LIME explanations.
    text_lime = lime_payload.get("text_lime")
    structured_lime = lime_payload.get("structured_lime")

    # Prepare structured LIME diagnostics.
    structured_lime_metrics = None

    if isinstance(structured_lime, dict):
        try:
            model_prediction = float(
                structured_lime["model_prediction"]
            )

            surrogate_output = float(
                structured_lime["surrogate_output"]
            )

            local_fidelity = float(
                structured_lime["local_fidelity"]
            )

            lime_input_prediction = float(
                structured_lime["lime_input_prediction"]
            )

            input_prediction_matches_model = bool(
                structured_lime["input_prediction_matches_model"]
            )

            structured_lime_metrics = {
                "model_prediction_percent": (
                    model_prediction * 100
                ),
                "surrogate_output": surrogate_output,
                "local_fidelity": local_fidelity,
                "lime_input_prediction_percent": (
                    lime_input_prediction * 100
                ),
                "input_prediction_matches_model": (
                    input_prediction_matches_model
                ),
            }

        except (KeyError, TypeError, ValueError):
            structured_lime_metrics = None

    # Prepare the template context.
    context = {
        "result": result,
        "job": result.job,
        "explanation": explanation,
        "text_lime": text_lime,
        "structured_lime": structured_lime,
        "structured_lime_metrics": structured_lime_metrics,
        "raw_lime_json": raw_lime_json,
    }

    return render(
        request,
        "core/admin_lime_explanation.html",
        context,
    )

    # =========================================================
# LOG OUT
# =========================================================

@never_cache
def logout_view(request):

    logout(request)

    return redirect("login")


 
# =========================================================
# SYSTEM ADMINISTRATOR DASHBOARD
# =========================================================



@login_required(login_url="login")
@never_cache
def admin_dashboard(request):
    # Only users in the System Administrator group may access this view.
    if not request.user.groups.filter(
        name="System Administrator"
    ).exists():
        return redirect("home")

    # User account statistics.
    total_users = User.objects.count()

    total_applicants = User.objects.filter(
        groups__name="Job Applicant"
    ).distinct().count()

    total_administrators = User.objects.filter(
        groups__name="System Administrator"
    ).distinct().count()

    # Testimonial statistics.
    pending_testimonials = Testimonial.objects.filter(
         status=Testimonial.Status.PENDING
    ).count()

    approved_testimonials = Testimonial.objects.filter(
        status=Testimonial.Status.APPROVED
    ).count()

    # Suspicious job report statistics.
    total_reports = SuspiciousJobReport.objects.count()

    pending_reports = SuspiciousJobReport.objects.filter(
        status=SuspiciousJobReport.Status.PENDING
    ).count()

       # Employer verification statistics.
    total_employers = Employer.objects.count()

    latest_verification_status = (
        EmployerVerificationRecord.objects.filter(
            employer=OuterRef("pk")
        )
        .order_by("-verification_date", "-created_at")
        .values("verification_status")[:1]
    )

    employers_with_latest_status = Employer.objects.annotate(
        latest_verification_status=Subquery(
            latest_verification_status
        )
    )

    pending_employer_verifications = employers_with_latest_status.filter(
        Q(
            latest_verification_status=
            EmployerVerificationRecord.VerificationStatus.PENDING
        )
        | Q(latest_verification_status__isnull=True)
    ).count()
    # All employers with their latest verification status.
    employer_list = (
        employers_with_latest_status
        .prefetch_related(
            Prefetch(
                "verification_records",
                queryset=(
                    EmployerVerificationRecord.objects
                    .select_related("verified_by__user")
                    .order_by("-verification_date", "-created_at")
                ),
                to_attr="verification_history",
            )
        )
        .order_by("company_name")
    )
    # Feedback and testimonials now use one model.
    total_feedback = Testimonial.objects.count()

    pending_feedback = Testimonial.objects.filter(
        status=Testimonial.Status.PENDING
    ).count()

    # Job advertisement and analysis statistics.
    total_job_advertisements = JobAdvertisement.objects.count()
    total_analysis_results = AnalysisResult.objects.count()

    # Recent users and testimonials.
    recent_users = User.objects.order_by("-date_joined")[:5]

    recent_testimonials = Testimonial.objects.select_related(
        "user"
    ).order_by("-created_at")[:5]

    # Recent suspicious job reports.
    recent_reports = SuspiciousJobReport.objects.select_related(
        "job",
        "applicant__user",
    ).order_by("-report_date")[:5]

    # Recent feedback from the unified Testimonial model.
    recent_feedback = Testimonial.objects.select_related(
        "user",
        "reviewed_by__user",
    ).order_by("-created_at")[:5]

    # Recent employers.
    recent_employers = Employer.objects.order_by(
        "-created_at"
    )[:5]

    # Recent employer verification records.
    recent_verifications = (
        EmployerVerificationRecord.objects.select_related(
            "employer",
            "verified_by__user",
        ).order_by("-verification_date")[:5]
    )
         # Recent job analysis results.
    recent_analysis_results = (
        AnalysisResult.objects
        .select_related("job", "job__employer", "model")
        .order_by("-analysis_date")[:10]
    )
    # Dashboard context.
    context = {
        "total_users": total_users,
        "total_applicants": total_applicants,
        "total_administrators": total_administrators,

        "pending_testimonials": pending_testimonials,
        "approved_testimonials": approved_testimonials,

        "total_reports": total_reports,
        "pending_reports": pending_reports,
        "total_employers": total_employers,
        "pending_employer_verifications": (
            pending_employer_verifications
        ),
        "employer_list": employer_list,
        "total_feedback": total_feedback,
        "pending_feedback": pending_feedback,

        "total_job_advertisements": total_job_advertisements,
        "total_analysis_results": total_analysis_results,

        "recent_users": recent_users,
        "recent_testimonials": recent_testimonials,
        "recent_reports": recent_reports,
        "recent_feedback": recent_feedback,
        "recent_employers": recent_employers,
        "recent_verifications": recent_verifications,
        "recent_analysis_results": recent_analysis_results,
        
    }

    
    return render(
        request,
        "core/AdminDashboard.html",
        context,
    )

@login_required(login_url="login")
@require_POST
@never_cache
def employer_verification_action(request, employer_id):

    # Only System Administrators may review employers.
    if not request.user.groups.filter(
        name="System Administrator"
    ).exists():
        return JsonResponse(
            {
                "success": False,
                "message": "You are not authorized to perform this action."
            },
            status=403
        )

    # Confirm that the logged-in administrator has a profile.
    administrator = SystemAdministrator.objects.filter(
        user=request.user
    ).first()

    if administrator is None:
        return JsonResponse(
            {
                "success": False,
                "message": "Your administrator profile could not be found."
            },
            status=403
        )

    # Retrieve the employer being reviewed.
    employer = get_object_or_404(
        Employer,
        employer_id=employer_id
    )

    # Validate the administrator's submitted verification decision.
    valid_statuses = dict(
        EmployerVerificationRecord.VerificationStatus.choices
    )

    try:
        validated_data = validate_employer_verification_data(
            request.POST,
            valid_statuses
        )

    except ValidationError as error:

        # Convert validation errors into field-specific messages.
        errors = {}

        for message in error.messages:

            if message.startswith("Administrator remarks"):
                errors["remarks"] = message

            elif message.startswith("Applicant-facing review summary"):
                errors["applicant_review_summary"] = message

            elif message.startswith("Please select a valid employer verification status"):
                errors["verification_status"] = message

            else:
                errors["general"] = message

        return JsonResponse(
            {
                "success": False,
                "errors": errors
            },
            status=400
        )

    # Retrieve the cleaned and validated values.
    new_status = validated_data["verification_status"]
    remarks = validated_data["remarks"]
    applicant_review_summary = validated_data["applicant_review_summary"]

      # Create a new record while preventing changes to an already-final decision.
    with transaction.atomic():

        # Lock the employer record to prevent simultaneous verification submissions.
        locked_employer = Employer.objects.select_for_update().get(
            employer_id=employer.employer_id
        )

        # Retrieve the most recent verification decision.
        latest_verification = (
            EmployerVerificationRecord.objects.filter(
                employer=locked_employer
            )
            .order_by("-verification_date", "-created_at")
            .first()
        )

        # Prevent another decision if the employer has already been reviewed.
        if (
            latest_verification
            and latest_verification.verification_status
            != EmployerVerificationRecord.VerificationStatus.PENDING
        ):
            return JsonResponse(
                {
                    "success": False,
                    "locked": True,
                    "message": (
                        "This employer already has a final verification decision. "
                        "Reopen verification before submitting another decision."
                    )
                },
                status=409
            )

        # Create a new record to preserve verification history.
        EmployerVerificationRecord.objects.create(
            employer=locked_employer,
            verified_by=administrator,
            verification_status=new_status,
            verification_date=timezone.now(),
            remarks=remarks,
            applicant_review_summary=applicant_review_summary,
        )

    # Return a success response without redirecting.
    return JsonResponse(
        {
            "success": True,
            "message": (
                f"Verification status for {employer.company_name} "
                "has been updated successfully."
            ),
            "status": new_status,
            "status_label": valid_statuses[new_status],
        }
    )


 
@login_required(login_url="login")
@require_POST
@never_cache
def reopen_employer_verification(request, employer_id):

    # Only System Administrators may reopen employer verification.
    if not request.user.groups.filter(
        name="System Administrator"
    ).exists():
        return JsonResponse(
            {
                "success": False,
                "message": "You are not authorized to perform this action."
            },
            status=403
        )

    # Confirm that the administrator has a profile.
    administrator = SystemAdministrator.objects.filter(
        user=request.user
    ).first()

    if administrator is None:
        return JsonResponse(
            {
                "success": False,
                "message": "Your administrator profile could not be found."
            },
            status=403
        )

    # Retrieve the employer.
    employer = get_object_or_404(
        Employer,
        employer_id=employer_id
    )

    # Retrieve and validate the reopening reason.
    reopening_reason = request.POST.get("reason", "").strip()

    if len(reopening_reason) < 10:
        return JsonResponse(
            {
                "success": False,
                "errors": {
                    "reason": (
                        "Please provide a reopening reason of at least "
                        "10 characters."
                    )
                }
            },
            status=400
        )

    # Reopen verification while preserving the previous decision.
    with transaction.atomic():

        locked_employer = Employer.objects.select_for_update().get(
            employer_id=employer.employer_id
        )

        latest_verification = (
            EmployerVerificationRecord.objects.filter(
                employer=locked_employer
            )
            .order_by("-verification_date", "-created_at")
            .first()
        )

        # Only final decisions can be reopened.
        if (
            latest_verification is None
            or latest_verification.verification_status
            == EmployerVerificationRecord.VerificationStatus.PENDING
        ):
            return JsonResponse(
                {
                    "success": False,
                    "message": (
                        "This employer does not currently have a final "
                        "verification decision to reopen."
                    )
                },
                status=409
            )

        # Create a new pending record; do not alter previous records.
        EmployerVerificationRecord.objects.create(
            employer=locked_employer,
            verified_by=administrator,
            verification_status=(
                EmployerVerificationRecord.VerificationStatus.PENDING
            ),
            verification_date=timezone.now(),
            remarks=f"Verification reopened. Reason: {reopening_reason}",
            applicant_review_summary="",
        )

    return JsonResponse(
        {
            "success": True,
            "message": (
                f"Verification for {employer.company_name} "
                "has been reopened successfully."
            ),
            "status": "pending",
            "status_label": "Pending Verification",
        }
    )
  
# =========================================================
# REVIEW, APPROVE AND REJECT USER FEEDBACK
# ========================================================



@login_required(login_url="login")
@require_POST
def testimonial_action(request, testimonial_id, action):

    # Restrict feedback management to system administrators.
    if not request.user.groups.filter(
        name="System Administrator"
    ).exists():
        return HttpResponseForbidden(
            "You are not authorised to manage feedback."
        )

    administrator = SystemAdministrator.objects.filter(
        user=request.user
    ).first()

    if administrator is None:
        messages.error(
            request,
            "Your administrator profile could not be found."
        )
        return redirect("admin_dashboard")

    if action not in ("review", "approve", "reject", "undo"):
        return HttpResponseForbidden("Invalid feedback action.")

    with transaction.atomic():

        testimonial = get_object_or_404(
            Testimonial.objects.select_for_update(),
            pk=testimonial_id
        )

        # -------------------------------------------------
        # UNDO A FINAL DECISION
        # -------------------------------------------------
        if action == "undo":

            if testimonial.status != Testimonial.Status.APPROVED:
                messages.warning(
                    request,
                    "Only feedback with a final decision can be restored."
                )
                return redirect("admin_dashboard")

            reason = request.POST.get("reason", "").strip()

            if not reason:
                messages.error(
                    request,
                    "Please provide a reason for undoing this decision."
                )
                return redirect("admin_dashboard")

            # Save the previous state before restoring the feedback.
            TestimonialDecisionLog.objects.create(
                testimonial=testimonial,
                administrator=administrator,
                action="undo",
                previous_status=testimonial.status,
                previous_is_approved=testimonial.is_approved,
                new_status=Testimonial.Status.UNDER_REVIEW,
                new_is_approved=False,
                reason=reason
            )

            # Return the feedback to Under Review.
            testimonial.status = Testimonial.Status.UNDER_REVIEW
            testimonial.is_approved = False
            testimonial.reviewed_by = administrator
            testimonial.reviewed_at = timezone.now()
            testimonial.admin_comment = (
                f"Previous decision reversed. Reason: {reason}"
            )

            testimonial.save()

            messages.success(
                request,
                "Decision reversed. Feedback returned to Under Review."
            )

            return redirect("admin_dashboard")

        # -------------------------------------------------
        # PREVENT CHANGES AFTER A FINAL DECISION
        # -------------------------------------------------
        if testimonial.status == Testimonial.Status.APPROVED:
            messages.warning(
                request,
                "A final decision has already been recorded. "
                "Use Undo Decision to reopen this feedback."
            )
            return redirect("admin_dashboard")

        if testimonial.status not in (
            Testimonial.Status.PENDING,
            Testimonial.Status.UNDER_REVIEW,
        ):
            messages.warning(
                request,
                "This feedback cannot be processed in its current status."
            )
            return redirect("admin_dashboard")

        # -------------------------------------------------
        # MARK AS UNDER REVIEW
        # -------------------------------------------------
        if action == "review":

            if testimonial.status != Testimonial.Status.PENDING:
                messages.warning(
                    request,
                    "This feedback is already Under Review."
                )
                return redirect("admin_dashboard")

            testimonial.status = Testimonial.Status.UNDER_REVIEW
            testimonial.is_approved = False

            message = "Feedback marked as Under Review."

        # -------------------------------------------------
        # APPROVE FOR PUBLICATION
        # -------------------------------------------------
        elif action == "approve":

            testimonial.status = Testimonial.Status.APPROVED
            testimonial.is_approved = True

            message = "Feedback approved for publication."

        # -------------------------------------------------
        # KEEP PRIVATE
        # -------------------------------------------------
        elif action == "reject":

            testimonial.status = Testimonial.Status.APPROVED
            testimonial.is_approved = False

            message = "Feedback finalised and kept private."

        testimonial.reviewed_by = administrator
        testimonial.reviewed_at = timezone.now()
        testimonial.admin_comment = request.POST.get(
            "admin_comment", ""
        ).strip()

        testimonial.save()

        messages.success(request, message)

    return redirect("admin_dashboard")