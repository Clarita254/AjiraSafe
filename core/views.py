from django.shortcuts import render, redirect

from django.contrib.auth.models import User, Group

from django.contrib.auth import login, authenticate , logout

from django.contrib import messages

from django.core.validators import validate_email

from django.core.exceptions import ValidationError

from django.contrib.auth.decorators import login_required

from django.http import HttpResponseForbidden

from django.urls import reverse

from django.views.csrf import csrf_failure as django_csrf_failure

from django.views.decorators.cache import never_cache


from .models import Testimonial, JobApplicant


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
# =========================================================

def testimonials(request):

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
        "core/testimonials.html",
        {
            "testimonials": testimonial_data
        }
    )


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

@login_required
def dashboard(request):

    return render(
        request,
        "core/Applicantdashboard.html"
    )


# =========================================================
# SYSTEM ADMINISTRATOR DASHBOARD
# =========================================================

@login_required
def admin_dashboard(request):

    return render(
        request,
        "core/AdminDashboard.html"
    )

    # =========================================================
# LOG OUT
# =========================================================

@never_cache
def logout_view(request):

    logout(request)

    return redirect("login")