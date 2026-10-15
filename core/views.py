from django.shortcuts import render, redirect

from django.contrib.auth.models import User, Group

from django.contrib.auth import login, authenticate , logout

from django.contrib import messages

from django.core.validators import validate_email

from django.core.exceptions import ValidationError

from django.contrib.auth.decorators import login_required

from django.http import HttpResponseForbidden

from django.urls import reverse
from django.db import transaction
from django.utils import timezone
from django.views.csrf import csrf_failure as django_csrf_failure

from django.views.decorators.cache import never_cache


from core.models import TestimonialDecisionLog
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

import re


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

    remember_me = (
        request.POST.get("remember_me") == "on"
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
                "remember_me": remember_me,
            }
        )


    if not password:

        return render(
            request,
            "core/login.html",
            {
                "password_required": True,
                "email": email,
                "remember_me": remember_me,
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
                "remember_me": remember_me,
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
                "remember_me": remember_me,
            }
        )


    # =====================================================
    # LOG USER IN
    # =====================================================

    login(
        request,
        user
    )


    # =====================================================
    # KEEP ME SIGNED IN
    # =====================================================

    if remember_me:

        # Keep the Django authentication session active
        # for 14 days.
        #
        # This does NOT store the user's password.

        request.session.set_expiry(
            60 * 60 * 24 * 14
        )

    else:

        # Session expires when the browser session ends.

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
    # Only Job Applicants may access this dashboard.
    if not request.user.groups.filter(
        name="Job Applicant"
    ).exists():
        if request.user.groups.filter(
            name="System Administrator"
        ).exists():
            return redirect("admin_dashboard")

        return redirect("home")

    # Retrieve the applicant's profile.
    try:
        applicant = request.user.job_applicant
    except JobApplicant.DoesNotExist:
        messages.error(
            request,
            "Your applicant profile could not be found."
        )
        return redirect("home")

    # Process the applicant's feedback submission.
    if request.method == "POST" and "feedback_type" in request.POST:
        category = request.POST.get(
            "feedback_type", ""
        ).strip()

        feedback_text = request.POST.get(
            "feedback", ""
        ).strip()

        rating_value = request.POST.get(
            "rating", ""
        ).strip()

        valid_categories = [
            choice[0]
            for choice in Testimonial.Category.choices
        ]

        if category not in valid_categories:
            messages.error(
                request,
                "Please select a valid feedback category."
            )

        elif not feedback_text:
            messages.error(
                request,
                "Please enter your feedback."
            )

        else:
            try:
                rating = int(rating_value)
            except (TypeError, ValueError):
                rating = 0

            if rating < 1 or rating > 5:
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

    # Show only feedback submitted by the logged-in applicant.
    feedback_submissions = Testimonial.objects.filter(
        user=request.user
    ).order_by("-created_at")

    # Dashboard summary information.
    jobs_analyzed = AnalysisResult.objects.filter(
        job__applicant=applicant
    ).count()

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

    context = {
        "jobs_analyzed": jobs_analyzed,
        "high_risk_count": high_risk_count,
        "reports_count": reports_count,
        "pending_feedback_count": pending_feedback_count,
        "feedback_submissions": feedback_submissions,
    }

    return render(
        request,
        "core/Applicantdashboard.html",
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

    pending_employer_verifications = Employer.objects.filter(
        verification_records__isnull=True
    ).distinct().count()

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
    }

    return render(
        request,
        "core/AdminDashboard.html",
        context,
    )

# =========================================================
# REVIEW, APPROVE AND REJECT USER FEEDBACK
# =========================================================

# =========================================================
# REVIEW, APPROVE, KEEP PRIVATE AND UNDO FEEDBACK DECISIONS
# =========================================================

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