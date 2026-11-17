

from django.shortcuts import render, redirect

from django.contrib.auth.models import User, Group

from django.contrib.auth import login, authenticate , logout

from django.contrib import messages

from django.core.validators import validate_email

from django.core.exceptions import ValidationError

from django.contrib.auth.decorators import login_required
from django.http import (
    HttpResponseForbidden,
    JsonResponse,
    FileResponse,
    Http404,
    request,
)
from django.urls import reverse
from django.db import transaction
from django.utils import timezone
from django.views.csrf import csrf_failure as django_csrf_failure

from django.views.decorators.cache import never_cache
from django.db.models import OuterRef, Prefetch, Subquery, Q
from core.models import TestimonialDecisionLog

from .models import (
    Testimonial,
    JobApplicant,
    SystemAdministrator,
    UserAccountFlag,
    Employer,
    JobAdvertisement,
    EmployerVerificationRecord,
    SuspiciousJobReport,
    SystemActivity,
    MLModel,
    AnalysisResult,
    RiskAssessment,
    Explanation,
    UserAccountFlag,

)

from .activity_logger import log_system_activity
from django.core.paginator import Paginator

from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_POST, require_GET
from .validators import (
    validate_job_assessment_data,
    validate_employer_data,
    validate_employer_verification_data,
)
import re
import json
import math
from decimal import Decimal
from urllib.parse import urlencode, urlparse
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

    testimonials = (
        Testimonial.objects
        .filter(is_approved=True)
        .select_related("user")
        .order_by("-created_at", "-id")[:3]
    )

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

    # Validate the requested action.
    if action not in ("review", "approve", "reject", "undo"):
        return HttpResponseForbidden("Invalid feedback action.")

    with transaction.atomic():

        testimonial = get_object_or_404(
            Testimonial.objects.select_for_update(),
            pk=testimonial_id
        )

        # Capture the previous feedback state before making changes.
        previous_status = testimonial.status
        previous_is_approved = testimonial.is_approved
        previous_admin_comment = testimonial.admin_comment or ""

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

            # Preserve the original decision history.
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

            # Restore the feedback to Under Review.
            testimonial.status = Testimonial.Status.UNDER_REVIEW
            testimonial.is_approved = False
            testimonial.reviewed_by = administrator
            testimonial.reviewed_at = timezone.now()
            testimonial.admin_comment = (
                f"Previous decision reversed. Reason: {reason}"
            )
            testimonial.save()

            # Record the reversal in System Activities.
            log_system_activity(
                actor=request.user,
                category="feedback",
                action="Feedback decision reversed",
                description=(
                    f"Administrator {request.user.get_username()} "
                    f"reversed the final decision for feedback "
                    f"#{testimonial.pk}."
                ),
                target_type="Testimonial",
                target_id=testimonial.pk,
                previous_value=(
                    f"Status: {previous_status}; "
                    f"Published: {previous_is_approved}"
                ),
                new_value=(
                    f"Status: {testimonial.status}; "
                    f"Published: {testimonial.is_approved}"
                ),
                metadata={
                    "administrator": request.user.get_username(),
                    "reason": reason,
                    "previous_admin_comment": previous_admin_comment,
                    "new_admin_comment": testimonial.admin_comment,
                    "action": "undo",
                }
            )

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

            activity_action = "Feedback marked as Under Review"
            new_feedback_state = "Under Review"

        # -------------------------------------------------
        # APPROVE FOR PUBLICATION
        # -------------------------------------------------
        elif action == "approve":

            testimonial.status = Testimonial.Status.APPROVED
            testimonial.is_approved = True

            message = "Feedback approved for publication."

            activity_action = "Feedback approved for publication"
            new_feedback_state = "Approved and published"

        # -------------------------------------------------
        # REJECT AND KEEP PRIVATE
        # -------------------------------------------------
        elif action == "reject":

            testimonial.status = Testimonial.Status.APPROVED
            testimonial.is_approved = False

            message = "Feedback finalised and kept private."

            activity_action = "Feedback rejected and kept private"
            new_feedback_state = "Finalised and kept private"

        # Save the administrator's review details.
        testimonial.reviewed_by = administrator
        testimonial.reviewed_at = timezone.now()
        testimonial.admin_comment = request.POST.get(
            "admin_comment", ""
        ).strip()

        testimonial.save()

        # -------------------------------------------------
        # RECORD FEEDBACK ACTIVITY
        # -------------------------------------------------
        log_system_activity(
            actor=request.user,
            category="feedback",
            action=activity_action,
            description=(
                f"Administrator {request.user.get_username()} "
                f"processed feedback #{testimonial.pk}."
            ),
            target_type="Testimonial",
            target_id=testimonial.pk,
            previous_value=(
                f"Status: {previous_status}; "
                f"Published: {previous_is_approved}"
            ),
            new_value=(
                f"Status: {testimonial.status}; "
                f"Published: {testimonial.is_approved}; "
                f"Result: {new_feedback_state}"
            ),
            metadata={
                "administrator": request.user.get_username(),
                "feedback_category": testimonial.category,
                "rating": testimonial.rating,
                "previous_admin_comment": previous_admin_comment,
                "new_admin_comment": testimonial.admin_comment,
                "action": action,
            }
        )

        messages.success(request, message)

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
            # RECORD USER REGISTRATION
            # =================================================

            log_system_activity(
                actor=user,
                category="user_management",
                action="Account registered",
                description=(
                    f"{user.get_full_name() or user.username} "
                    "registered a new AjiraSafe account."
                ),
                target_type="User",
                target_id=user.id,
                previous_value="No account",
                new_value="Registered and active"
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

    # OPEN LOGIN PAGE
    if request.method == "GET":
        list(messages.get_messages(request))
        return render(request, "core/login.html")

    # GET LOGIN FORM DATA
    email = request.POST.get("email", "").strip().lower()
    password = request.POST.get("password", "")

    # VALIDATE REQUIRED FIELDS
    if not email:
        return render(
            request,
            "core/login.html",
            {"email_required": True}
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

    # FIND ACCOUNT BY EMAIL
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

    # VALIDATE LOGIN CREDENTIALS
    if (
        account is None
        or user is None
        or not user.is_active
    ):
        log_system_activity(
            actor=None,
            category="authentication",
            action="Login failed",
            description=(
                "A login attempt failed because the credentials "
                "were invalid or the account was inactive."
            ),
            target_type="User" if account is not None else "",
            target_id=account.id if account is not None else "",
            previous_value="Unauthenticated",
            new_value="Login unsuccessful",
            metadata={
                "reason": "Invalid credentials or inactive account"
            }
        )

        return render(
            request,
            "core/login.html",
            {"invalid_credentials": True}
        )

    # DETERMINE USER ROLE
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
            {"invalid_credentials": True}
        )

    # LOG USER IN
    login(request, user)

    # RECORD SUCCESSFUL AUTHENTICATION
    log_system_activity(
        actor=user,
        category="authentication",
        action="User logged in",
        description=(
            f"{user.get_full_name() or user.username} "
            "successfully logged into AjiraSafe."
        ),
        target_type="User",
        target_id=user.id,
        previous_value="Logged out",
        new_value="Authenticated",
        metadata={
            "user_role": (
                "System Administrator"
                if user.groups.filter(
                    name="System Administrator"
                ).exists()
                else "Job Applicant"
            )
        }
    )

    # KEEP AUTHENTICATION LIMITED TO CURRENT BROWSER SESSION
    request.session.set_expiry(0)

    # REDIRECT TO APPROPRIATE DASHBOARD
    return redirect(dashboard_name)

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
    
    # Handle suspicious job report submissions.
    if request.method == "POST" and "suspicious_report_submit" in request.POST:

        job_title = request.POST.get("job_title", "").strip()
        company_name = request.POST.get("company_name", "").strip()
        job_url = request.POST.get("job_url", "").strip()
        reason = request.POST.get("reason", "").strip()
        description = request.POST.get("description", "").strip()
        job_id = request.POST.get("job_id", "").strip()

        # Retrieve optional supporting evidence.
        evidence = request.FILES.get("evidence")

        # Accepted report reasons.
        valid_reasons = {
            "payment_request",
            "personal_information",
            "fake_employer",
            "misleading_offer",
            "suspicious_contact",
            "other",
        }

        # Validate optional supporting evidence.
        allowed_extensions = {
            ".jpg": "jpeg",
            ".jpeg": "jpeg",
            ".png": "png",
            ".pdf": "pdf",
        }

        evidence_error = None

        if evidence:
            filename = evidence.name.lower()
            extension = (
                "." + filename.rsplit(".", 1)[-1]
                if "." in filename
                else ""
            )

            if evidence.size > 5 * 1024 * 1024:
                evidence_error = (
                    "Supporting evidence must not exceed 5 MB."
                )

            elif extension not in allowed_extensions:
                evidence_error = (
                    "Only JPG, JPEG, PNG and PDF files are allowed."
                )

            else:
                file_signature = evidence.read(8)
                evidence.seek(0)

                valid_signatures = {
                    "jpeg": file_signature.startswith(b"\xff\xd8\xff"),
                    "png": file_signature.startswith(
                        b"\x89PNG\r\n\x1a\n"
                    ),
                    "pdf": file_signature.startswith(b"%PDF-"),
                }

                expected_type = allowed_extensions[extension]

                if not valid_signatures[expected_type]:
                    evidence_error = (
                        "The uploaded file does not match its file extension."
                    )

        # Validate report information.
        if evidence_error:
            messages.error(request, evidence_error)

        elif not job_title or not company_name:
            messages.error(
                request,
                "Please provide both the job title and company name."
            )

        elif len(job_title) > 255 or len(company_name) > 255:
            messages.error(
                request,
                "The job title and company name must not exceed 255 characters."
            )

        elif reason not in valid_reasons:
            messages.error(
                request,
                "Please select a valid reason for reporting this advertisement."
            )

        elif not description:
            messages.error(
                request,
                "Please explain why you believe this job advertisement is suspicious."
            )

        elif len(job_url) > 500:
            messages.error(
                request,
                "The job advertisement URL must not exceed 500 characters."
            )

        else:
            # Validate the optional advertisement URL.
            valid_url = True

            if job_url:
                parsed_url = urlparse(job_url)

                if (
                    parsed_url.scheme not in ("http", "https")
                    or not parsed_url.hostname
                ):
                    valid_url = False

            if not valid_url:
                messages.error(
                    request,
                    "Please enter a valid job advertisement URL beginning with http:// or https://."
                )

            else:
                # Link to an existing AjiraSafe advertisement if supplied.
                linked_job = None

                if job_id:
                    linked_job = (
                        JobAdvertisement.objects
                        .select_related("employer")
                        .filter(
                            job_id=job_id,
                            applicant=applicant
                        )
                        .first()
                    )

                    if linked_job is None:
                        messages.error(
                            request,
                            "The selected AjiraSafe advertisement could not be found."
                        )

                # Save the report when the advertisement details are valid.
                if not job_id or linked_job is not None:

                    # Use saved details for AjiraSafe advertisements.
                    if linked_job is not None:
                        job_title = linked_job.title
                        company_name = linked_job.employer.company_name

                    # Save BOTH AjiraSafe and external advertisement reports.
                    new_report = SuspiciousJobReport.objects.create(
                        job=linked_job,
                        applicant=applicant,
                        job_title=job_title,
                        company_name=company_name,
                        job_url=job_url,
                        reason=reason,
                        description=description,
                        evidence=evidence,
                        status=SuspiciousJobReport.Status.PENDING,
                    )

                    messages.success(
                        request,
                        f"Your suspicious job report #{new_report.report_id} "
                        "has been submitted successfully. "
                        "You can track its status under My Submitted Reports."
                    )

                    return redirect(
                        f"{reverse('dashboard')}?tab=reports"
                    )

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

                    else:
                        # Update existing employer information when new values are supplied.
                        if employer_website:
                            employer.website = employer_website

                        if company_reg_number:
                            employer.company_reg_number = company_reg_number

                        if contact_email:
                            employer.contact_email = contact_email

                        if contact_phone:
                            employer.contact_phone = contact_phone

                        employer.save(
                            update_fields=[
                                "website",
                                "company_reg_number",
                                "contact_email",
                                "contact_phone",
                                "updated_at",
                            ]
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

    # Retrieve the applicant's reports in ascending report number order.
    my_reports = (
        SuspiciousJobReport.objects
        .filter(applicant=applicant)
        .select_related("job")
        .order_by("report_id")
    )

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
        "my_reports": my_reports,
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
    Convert saved LIME output into an applicant-friendly summary.

    Features are ranked by the absolute magnitude of their actual
    LIME weights. Contributions are also separated according to
    whether they increased or decreased the estimated fraud probability.

    Raw numerical weights are not exposed to applicants.
    """
    empty_summary = {
        "available": False,
        "ranked_features": [],
        "increased_features": [],
        "decreased_features": [],
    }

    if not isinstance(lime_data, dict):
        return empty_summary

    features = lime_data.get("features", [])

    if not isinstance(features, list):
        return empty_summary

    def readable_feature(raw_feature):
        raw_feature = str(raw_feature).strip()

        if not structured:
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
            direction_group = "increased"
        elif weight < 0:
            direction = "Decreases the estimated fraud probability"
            direction_group = "decreased"
        else:
            direction = "No directional contribution"
            direction_group = "neutral"

        contributions.append({
            "magnitude": abs(weight),
            "feature": feature_label,
            "direction": direction,
            "direction_group": direction_group,
        })

    # Rank all contributions by absolute LIME weight.
    contributions.sort(
        key=lambda contribution: contribution["magnitude"],
        reverse=True,
    )

    ranked_features = []
    increased_features = []
    decreased_features = []

    for rank, contribution in enumerate(contributions, start=1):
        feature_data = {
            "rank": rank,
            "feature": contribution["feature"],
            "direction": contribution["direction"],
        }

        ranked_features.append(feature_data)

        if contribution["direction_group"] == "increased":
            increased_features.append(feature_data)
        elif contribution["direction_group"] == "decreased":
            decreased_features.append(feature_data)

    return {
        "available": bool(ranked_features),
        "ranked_features": ranked_features,
        "increased_features": increased_features,
        "decreased_features": decreased_features,
    }

def detect_advertisement_warning_signs(job):
    """
    Identify potential warning signs from the actual job advertisement.

    This rule-based analysis is separate from the machine learning
    prediction and LIME explanations. It does not independently
    establish that an advertisement is fraudulent.
    """

    description = str(getattr(job, "description", "") or "")
    title = str(getattr(job, "title", "") or "")
    company_profile = str(
        getattr(job, "company_profile", "") or ""
    )
    requirements = str(
        getattr(job, "requirements", "") or ""
    )
    benefits = str(
        getattr(job, "benefits", "") or ""
    )

    # Combine the advertisement's text for rule matching.
    advertisement_text = " ".join([
        title,
        description,
        requirements,
        benefits,
    ])

    normalised_text = " ".join(
        advertisement_text.split()
    )

    text_lower = normalised_text.casefold()

    warning_signs = []

    def add_warning(title, explanation, pattern):
        """
        Add a warning only when its supporting wording
        is actually found in the advertisement.
        """
        match = re.search(
            pattern,
            normalised_text,
            flags=re.IGNORECASE,
        )

        if match:
            # Extract the sentence containing the matched wording.
            start = normalised_text.rfind(".", 0, match.start()) + 1
            end = normalised_text.find(".", match.end())

            if end == -1:
                end = len(normalised_text)

            evidence = normalised_text[start:end].strip()

            warning_signs.append({
                "title": title,
                "explanation": explanation,
                "evidence": evidence,
                "severity": "High concern",
            })

    # 1. Upfront recruitment or registration fees.
    add_warning(
        "Upfront payment request",
        (
            "The advertisement appears to request payment "
            "as a condition of securing or obtaining employment."
        ),
        (
            r"(?:registration|processing|application|recruitment|"
            r"training)\s+(?:and\s+)?(?:registration\s+|processing\s+)?"
            r"fees?|"
            r"(?:pay|send|transfer)\s+(?:an?\s+)?(?:upfront\s+)?"
            r"(?:fee|payment|money)"
        ),
    )

    # 2. Guaranteed or immediate employment promises.
    add_warning(
        "Guaranteed employment promise",
        (
            "The advertisement promises guaranteed or immediate "
            "employment, which may pressure applicants to act "
            "without sufficient verification."
        ),
        (
            r"guaranteed\s+(?:employment|job|placement)|"
            r"immediate\s+(?:employment|job|placement)|"
            r"guaranteed\s+employment\s+immediately"
        ),
    )

    # 3. High earnings combined with no experience.
    earnings_claim = re.search(
        r"(?:kes\s*)?[\d,]+\s*(?:per\s+)?(?:week|weekly)",
        normalised_text,
        flags=re.IGNORECASE,
    )

    no_experience_claim = re.search(
        r"no\s+(?:prior\s+)?experience\s+(?:required|needed)|"
        r"without\s+(?:any\s+)?experience|"
        r"no\s+experience\s+required",
        normalised_text,
        flags=re.IGNORECASE,
    )

    if earnings_claim and no_experience_claim:
        warning_signs.append({
            "title": "Weekly earnings claim with no experience required",
            "explanation": (
                "The advertisement combines a weekly earnings claim "
                "with a statement that no prior experience is required. "
                "Applicants should independently verify the advertised "
                "compensation and employment terms."
            ),
            "evidence": (
                earnings_claim.group(0)
                + " ... "
                + no_experience_claim.group(0)
            ),
            "severity": "Warning sign",
        })


   
   
    # 4. Limited employer background information.
    # Check both the advertisement and the saved employer record.

    company_background_pattern = (
        r"\b(?:company|organisation|organization|employer|business|"
        r"institution|bank|firm|corporation)\b"
        r".{0,200}\b(?:provides|providing|offers|offering|"
        r"specialises|specializes|operates|operating|"
        r"delivers|delivering|was established|was founded)\b"
        r"|"
        r"\b(?:is|was)\s+(?:a|an)\s+"
        r"[\w-]+\s+(?:company|institution|organisation|"
        r"organization|bank|business|corporation)\b"
    )

    employer = getattr(job, "employer", None)

    employer_details_available = bool(
        employer and any([
            getattr(employer, "website", ""),
            getattr(employer, "company_reg_number", ""),
            getattr(employer, "contact_email", ""),
            getattr(employer, "contact_phone", ""),
        ])
    )

    has_company_background = bool(
        company_profile.strip()
        or re.search(
            company_background_pattern,
            normalised_text,
            flags=re.IGNORECASE,
        )
        or employer_details_available
    )

    if not has_company_background:
        warning_signs.append({
            "title": "Limited employer background information",
            "explanation": (
                "No meaningful company profile, employer background "
                "description or employer contact and identification "
                "details were provided. Applicants should independently "
                "establish the employer's identity."
            ),
            "evidence": (
                "No company profile or matching employer background "
                "description was identified, and no employer "
                "verification details were recorded."
            ),
            "severity": "Information missing",
        })

    # 5. WhatsApp contact combined with immediate placement.
    whatsapp_contact = re.search(
        r"whatsapp",
        text_lower,
    )

    immediate_placement = re.search(
        r"immediate\s+(?:job\s+)?placement|"
        r"contact\s+us\s+for\s+placement|"
        r"contact\s+us\s+immediately",
        text_lower,
    )

    if whatsapp_contact and immediate_placement:
        warning_signs.append({
            "title": "Informal recruitment contact",
            "explanation": (
                "The advertisement combines WhatsApp contact "
                "with an immediate-placement request. Applicants "
                "should verify the recruiter through independently "
                "sourced employer contact details."
            ),
            "evidence": (
                "WhatsApp contact is requested alongside "
                "an immediate-placement statement."
            ),
            "severity": "Warning sign",
        })

    return {
        "available": bool(warning_signs),
        "count": len(warning_signs),
        "items": warning_signs,
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


    # Detect warning signs directly from the actual advertisement.
    # This analysis is independent of the ML and LIME explanations.
    advertisement_warning_signs = detect_advertisement_warning_signs(
        result.job
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
        "advertisement_warning_signs": advertisement_warning_signs,
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

    # Capture the authenticated user before ending the session.
    if request.user.is_authenticated:

        log_system_activity(
            actor=request.user,
            category="authentication",
            action="User logged out",
            description=(
                f"{request.user.get_full_name() or request.user.username} "
                "successfully logged out of AjiraSafe."
            ),
            target_type="User",
            target_id=request.user.id,
            previous_value="Authenticated",
            new_value="Logged out"
        )

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

    # Recent users for the Overview section.
    # Keep this limited to five accounts.
    recent_users = User.objects.order_by("-date_joined")[:5]

    # All registered users for Manage Users.
    user_queryset = (
    User.objects
    .prefetch_related(
        "groups",
       
    )
    .select_related("account_flag")
    .order_by("-date_joined")
)
    

    # Search by full name, partial name, username, or email.
    user_search = request.GET.get("user_search", "").strip()

    if user_search:
        search_terms = user_search.split()

        for term in search_terms:
            user_queryset = user_queryset.filter(
                Q(username__icontains=term)
                | Q(first_name__icontains=term)
                | Q(last_name__icontains=term)
                | Q(email__icontains=term)
            )

    # Filter by an existing Django group.
    user_role = request.GET.get("user_role", "").strip()

    if user_role:
        user_queryset = user_queryset.filter(
            groups__name=user_role
        )

    # Filter by account activation status.
    user_status = request.GET.get("user_status", "").strip()

    if user_status == "active":
        user_queryset = user_queryset.filter(is_active=True)

    elif user_status == "inactive":
        user_queryset = user_queryset.filter(is_active=False)

    user_queryset = user_queryset.distinct()

    # Paginate the Manage Users list.
    user_paginator = Paginator(user_queryset, 10)

    user_page_number = request.GET.get("user_page")

    all_users = user_paginator.get_page(user_page_number)

    # Existing groups are displayed as role filters.
    user_role_choices = Group.objects.order_by("name")

    # Recent testimonials for the Overview section.
    recent_testimonials = Testimonial.objects.select_related(
        "user"
    ).order_by("-created_at")[:5]

    # Recent suspicious job reports.
    # Keep the five newest reports, displayed in ascending order.
    recent_reports = list(
        SuspiciousJobReport.objects
        .select_related(
            "job",
            "applicant__user",
        )
        .order_by("-report_id")[:5]
    )

    recent_reports.reverse()

    # Retrieve all suspicious job reports in ascending report number order.
    all_suspicious_reports = (
        SuspiciousJobReport.objects
        .select_related(
            "job",
            "applicant__user",
            "reviewed_by__user",
        )
        .order_by("report_id")
    )

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

    # Recent system activities for the System Activities section.
    system_activities = (
        SystemActivity.objects
        .select_related("actor")
        .order_by("-created_at")[:100]
    )

    total_system_activities = SystemActivity.objects.count()
    # Dashboard context.
    context = {
        "total_users": total_users,
        "total_applicants": total_applicants,
        "total_administrators": total_administrators,

        "pending_testimonials": pending_testimonials,
        "approved_testimonials": approved_testimonials,

        "total_reports": total_reports,
        "pending_reports": pending_reports,
        "all_suspicious_reports": all_suspicious_reports,
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
        "all_users": all_users,
        "user_search": user_search,
        "user_role": user_role,
        "user_status": user_status,
        "user_role_choices": user_role_choices,
        "recent_testimonials": recent_testimonials,
        "recent_reports": recent_reports,
        "recent_feedback": recent_feedback,
        "recent_employers": recent_employers,
        "recent_verifications": recent_verifications,
        "recent_analysis_results": recent_analysis_results,
        "system_activities": system_activities,
"total_system_activities": total_system_activities,
        
    }

    
    return render(
        request,
        "core/AdminDashboard.html",
        context,
    )
# =========================================================
# ADMINISTRATOR USER ACCOUNT MANAGEMENT
# =========================================================

@login_required(login_url="login")
@require_POST
@never_cache
def manage_user_account(request, user_id):

    def redirect_to_user_management():
        query = {"tab": "users"}

        page = request.GET.get("user_page", "1").strip()
        if page.isdigit():
            query["user_page"] = page

        for key in ("user_search", "user_role", "user_status"):
            value = request.GET.get(key, "").strip()
            if value:
                query[key] = value

        return redirect(
            f"{reverse('admin_dashboard')}?{urlencode(query)}"
            "#registered-accounts"
        )

    # Only System Administrators may manage accounts.
    if not request.user.groups.filter(
        name="System Administrator"
    ).exists():
        return HttpResponseForbidden(
            "You are not authorized to manage user accounts."
        )

    administrator = SystemAdministrator.objects.filter(
        user=request.user
    ).first()

    if administrator is None:
        return HttpResponseForbidden(
            "Your administrator profile could not be found."
        )

    target_user = get_object_or_404(User, pk=user_id)

    action = request.POST.get("action", "").strip()

    # Prevent an administrator from disabling their own account.
    if (
        action == "deactivate"
        and target_user.pk == request.user.pk
    ):
        messages.error(
            request,
            "You cannot deactivate your own administrator account."
        )
        return redirect_to_user_management()

    # Do not allow the last active administrator to be deactivated.
    if (
        action == "deactivate"
        and target_user.groups.filter(
            name="System Administrator"
        ).exists()
        and target_user.is_active
    ):
        active_admin_count = User.objects.filter(
            is_active=True,
            groups__name="System Administrator"
        ).distinct().count()

        if active_admin_count <= 1:
            messages.error(
                request,
                "The last active administrator cannot be deactivated."
            )
            return redirect_to_user_management()

  
    if action == "activate":

        previous_status = (
            "Active" if target_user.is_active else "Inactive"
        )

        target_user.is_active = True
        target_user.save(update_fields=["is_active"])

        if previous_status == "Inactive":

            log_system_activity(
                actor=request.user,
                category="user_management",
                action="Account activated",
                description=(
                    f"Administrator activated the account "
                    f"belonging to {target_user.username}."
                ),
                target_type="User",
                target_id=target_user.id,
                previous_value="Inactive",
                new_value="Active"
            )

        messages.success(
            request,
            f"{target_user.username} has been activated."
        )

    elif action == "deactivate":

        previous_status = (
            "Active" if target_user.is_active else "Inactive"
        )

        target_user.is_active = False
        target_user.save(update_fields=["is_active"])

        if previous_status == "Active":

            log_system_activity(
                actor=request.user,
                category="user_management",
                action="Account deactivated",
                description=(
                    f"Administrator deactivated the account "
                    f"belonging to {target_user.username}."
                ),
                target_type="User",
                target_id=target_user.id,
                previous_value="Active",
                new_value="Inactive"
            )

        messages.success(
            request,
            f"{target_user.username} has been deactivated."
        )

   
 
    elif action == "flag":

        reason = request.POST.get("reason", "").strip()

        if not reason:
            messages.error(
                request,
                "Please provide a reason for flagging this account."
            )
            return redirect_to_user_management()

        account_flag, created = UserAccountFlag.objects.get_or_create(
            user=target_user
        )

        if account_flag.is_flagged:
            messages.warning(
                request,
                "This account is already flagged."
            )
            return redirect_to_user_management()

        account_flag.is_flagged = True
        account_flag.reason = reason
        account_flag.flagged_by = administrator
        account_flag.flagged_at = timezone.now()
        account_flag.cleared_by = None
        account_flag.cleared_at = None

        account_flag.save()

        messages.success(
            request,
            f"{target_user.username} has been flagged for administrator attention."
        )

        messages.success(
            request,
            f"{target_user.username} has been flagged for administrator attention."
        )

    elif action == "unflag":

        account_flag = UserAccountFlag.objects.filter(
            user=target_user,
            is_flagged=True
        ).first()

        if account_flag is None:
            messages.warning(
                request,
                "This account is not currently flagged."
            )
            return redirect_to_user_management()

        account_flag.is_flagged = False
        account_flag.cleared_by = administrator
        account_flag.cleared_at = timezone.now()
        account_flag.save()
       

        messages.success(
            request,
            f"The flag on {target_user.username} has been cleared."
        )

    else:
        messages.error(
            request,
            "Invalid account management action."
        )

    return redirect_to_user_management()

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

     
        # Capture the previous verification status for the activity log.
        previous_status = (
            latest_verification.verification_status
            if latest_verification
            else "No previous verification"
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

        # Record the verification decision in System Activities.
        log_system_activity(
            actor=request.user,
            category="employer_verification",
            action="Employer verification decision recorded",
            description=(
                f"Administrator {request.user.get_username()} "
                f"recorded a verification decision for "
                f"{locked_employer.company_name}."
            ),
            target_type="Employer",
            target_id=locked_employer.employer_id,
            previous_value=(
                valid_statuses.get(previous_status, previous_status)
            ),
            new_value=valid_statuses[new_status],
            metadata={
                "administrator": request.user.get_username(),
                "remarks": remarks,
                "applicant_review_summary": applicant_review_summary,
            }
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

       
        # Capture the previous final decision before reopening.
        previous_status = latest_verification.verification_status

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

        # Record the reopening event in System Activities.
        valid_statuses = dict(
            EmployerVerificationRecord.VerificationStatus.choices
        )

        log_system_activity(
            actor=request.user,
            category="employer_verification",
            action="Employer verification reopened",
            description=(
                f"Administrator {request.user.get_username()} "
                f"reopened verification for "
                f"{locked_employer.company_name}."
            ),
            target_type="Employer",
            target_id=locked_employer.employer_id,
            previous_value=valid_statuses.get(
                previous_status, previous_status
            ),
            new_value=valid_statuses[
                EmployerVerificationRecord.VerificationStatus.PENDING
            ],
            metadata={
                "administrator": request.user.get_username(),
                "reopening_reason": reopening_reason,
            }
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

        # Capture the previous feedback state before making changes.
        previous_status = testimonial.status
        previous_is_approved = testimonial.is_approved
        previous_admin_comment = testimonial.admin_comment or ""

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

            # Preserve the original decision history.
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

            # Record the reversal in System Activities.
            log_system_activity(
                actor=request.user,
                category="feedback",
                action="Feedback decision reversed",
                description=(
                    f"Administrator {request.user.get_username()} "
                    f"reversed the final decision for feedback "
                    f"#{testimonial.pk}."
                ),
                target_type="Testimonial",
                target_id=testimonial.pk,
                previous_value=(
                    f"Status: {previous_status}; "
                    f"Published: {previous_is_approved}"
                ),
                new_value=(
                    f"Status: {testimonial.status}; "
                    f"Published: {testimonial.is_approved}"
                ),
                metadata={
                    "administrator": request.user.get_username(),
                    "reason": reason,
                    "previous_admin_comment": previous_admin_comment,
                    "new_admin_comment": testimonial.admin_comment,
                    "action": "undo",
                }
            )

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
            activity_action = "Feedback marked as Under Review"
            new_feedback_state = "Under Review"

        # -------------------------------------------------
        # APPROVE FOR PUBLICATION
        # -------------------------------------------------
        elif action == "approve":

            testimonial.status = Testimonial.Status.APPROVED
            testimonial.is_approved = True

            message = "Feedback approved for publication."
            activity_action = "Feedback approved for publication"
            new_feedback_state = "Approved and published"

        # -------------------------------------------------
        # REJECT AND KEEP PRIVATE
        # -------------------------------------------------
        elif action == "reject":

            testimonial.status = Testimonial.Status.APPROVED
            testimonial.is_approved = False

            message = "Feedback finalised and kept private."
            activity_action = "Feedback rejected and kept private"
            new_feedback_state = "Finalised and kept private"

        # Save the administrator's review details.
        testimonial.reviewed_by = administrator
        testimonial.reviewed_at = timezone.now()
        testimonial.admin_comment = request.POST.get(
            "admin_comment", ""
        ).strip()

        testimonial.save()

        # -------------------------------------------------
        # RECORD FEEDBACK ACTIVITY
        # -------------------------------------------------
        log_system_activity(
            actor=request.user,
            category="feedback",
            action=activity_action,
            description=(
                f"Administrator {request.user.get_username()} "
                f"processed feedback #{testimonial.pk}."
            ),
            target_type="Testimonial",
            target_id=testimonial.pk,
            previous_value=(
                f"Status: {previous_status}; "
                f"Published: {previous_is_approved}"
            ),
            new_value=(
                f"Status: {testimonial.status}; "
                f"Published: {testimonial.is_approved}; "
                f"Result: {new_feedback_state}"
            ),
            metadata={
                "administrator": request.user.get_username(),
                "feedback_category": testimonial.category,
                "rating": testimonial.rating,
                "previous_admin_comment": previous_admin_comment,
                "new_admin_comment": testimonial.admin_comment,
                "action": action,
            }
        )

        messages.success(request, message)

    return redirect("admin_dashboard")



@login_required(login_url="login")
@require_POST
@never_cache
def suspicious_job_report_action(request, report_id):

    # Only System Administrators may manage suspicious job reports.
    if not request.user.groups.filter(
        name="System Administrator"
    ).exists():
        return HttpResponseForbidden(
            "You are not authorised to manage suspicious job reports."
        )

    # Confirm that the administrator has a registered profile.
    administrator = SystemAdministrator.objects.filter(
        user=request.user
    ).first()

    if administrator is None:
        messages.error(
            request,
            "Your administrator profile could not be found."
        )
        return redirect("admin_dashboard")

    # Retrieve the submitted status and administrator comment.
    new_status = request.POST.get("status", "").strip()
    admin_comment = request.POST.get("admin_comment", "").strip()

    # Validate the submitted status against the model's actual choices.
    valid_statuses = dict(
        SuspiciousJobReport.Status.choices
    )

    if new_status not in valid_statuses:
        messages.error(
            request,
            "Please select a valid suspicious report status."
        )
        return redirect(
            f"{reverse('admin_dashboard')}?tab=reports"
        )

    # Lock the report while its status is being updated.
    with transaction.atomic():

        report = get_object_or_404(
            SuspiciousJobReport.objects.select_for_update(),
            report_id=report_id
        )

        # Prevent modifications after a report has been resolved.
        if report.status == SuspiciousJobReport.Status.RESOLVED:
            messages.warning(
                request,
                f"Suspicious job report #{report.report_id} "
                "has already been resolved and cannot be modified."
            )
            return redirect(
                f"{reverse('admin_dashboard')}?tab=reports"
            )

        # Require an administrative comment when resolving a report.
        if (
            new_status == SuspiciousJobReport.Status.RESOLVED
            and not admin_comment
        ):
            messages.error(
                request,
                "Please provide a final administrative comment "
                "before resolving this report."
            )
            return redirect(
                f"{reverse('admin_dashboard')}?tab=reports"
            )

        # Capture the previous values before updating the report.
        previous_status = report.status
        previous_admin_comment = report.admin_comment

        # Update the report and preserve the administrator's review details.
        report.status = new_status
        report.reviewed_by = administrator
        report.reviewed_at = timezone.now()
        report.admin_comment = admin_comment

        report.save(
            update_fields=[
                "status",
                "reviewed_by",
                "reviewed_at",
                "admin_comment",
            ]
        )

        # Record the administrative activity only when
        # the status or comment has actually changed.
        if (
            previous_status != new_status
            or previous_admin_comment != admin_comment
        ):

            # Determine the activity action.
            if new_status == SuspiciousJobReport.Status.RESOLVED:
                activity_action = "Suspicious report resolved"

            elif previous_status != new_status:
                activity_action = "Suspicious report status updated"

            else:
                activity_action = "Suspicious report review updated"

            # Record the activity in the central system activity log.
            log_system_activity(
                actor=request.user,
                category="suspicious_reports",
                action=activity_action,
                description=(
                    f"Administrator {request.user.get_username()} "
                    f"updated suspicious job report #{report.report_id}."
                ),
                target_type="Suspicious Job Report",
                target_id=report.report_id,
                previous_value=(
                    f"Status: {valid_statuses.get(previous_status, previous_status)}"
                ),
                new_value=(
                    f"Status: {valid_statuses.get(new_status, new_status)}"
                ),
                metadata={
                    "previous_admin_comment": previous_admin_comment or "",
                    "new_admin_comment": admin_comment,
                }
            )

    messages.success(
        request,
        f"Suspicious job report #{report.report_id} "
        f"has been updated to {valid_statuses[new_status]}."
    )

    return redirect(
        f"{reverse('admin_dashboard')}?tab=reports"
    )

@login_required(login_url="login")
@require_GET
@never_cache
def suspicious_job_report_evidence(request, report_id):

    # Only System Administrators may access submitted evidence.
    if not request.user.groups.filter(
        name="System Administrator"
    ).exists():
        return HttpResponseForbidden(
            "You are not authorised to access suspicious report evidence."
        )

    # Confirm that the administrator has a registered profile.
    administrator_exists = SystemAdministrator.objects.filter(
        user=request.user
    ).exists()

    if not administrator_exists:
        return HttpResponseForbidden(
            "Your administrator profile could not be verified."
        )

    # Retrieve the requested suspicious job report.
    report = get_object_or_404(
        SuspiciousJobReport,
        report_id=report_id
    )

    # Confirm that evidence was attached to this report.
    if not report.evidence:
        raise Http404("No supporting evidence is attached to this report.")

    # Confirm that the file exists in private storage.
    if not report.evidence.storage.exists(report.evidence.name):
        raise Http404("The supporting evidence file could not be found.")

    # Open the file through Django's storage system.
    try:
        evidence_file = report.evidence.open("rb")
    except (OSError, ValueError):
        raise Http404("The supporting evidence file is unavailable.")

    # Use only the stored filename, not a user-supplied file path.
    filename = report.evidence.name.rsplit("/", 1)[-1]

    # Force downloading rather than displaying uploaded content in the browser.
    response = FileResponse(
        evidence_file,
        as_attachment=True,
        filename=filename,
        content_type="application/octet-stream",
    )

    # Apply additional response security headers.
    response["X-Content-Type-Options"] = "nosniff"
    response["Cache-Control"] = "private, no-store"

    return response
