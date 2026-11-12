from django.urls import path
from . import views


urlpatterns = [
    path("", views.home, name="home"),
    path("about/", views.about, name="about"),
    path("services/", views.services, name="services"),
    path("signup/", views.signup, name="signup"),
    path("login/", views.login_view, name="login"),
    path("forgot-password/", views.forgot_password, name="forgot_password"),
    path(
        "testimonials/",
        views.testimonials,
        name="testimonials"
    ),
    path(
    "dashboard/",
    views.dashboard,
    name="dashboard"
    ),
    path(
    "feedback/<int:testimonial_id>/<str:action>/",
    views.testimonial_action,
    name="testimonial_action",
   ),
   path("admin-dashboard/", views.admin_dashboard, name="admin_dashboard"),

path(
    "administrator/employers/<int:employer_id>/verify/",
    views.employer_verification_action,
    name="employer_verification_action",
),
    path(
        "administrator/reports/<int:report_id>/update/",
        views.suspicious_job_report_action,
        name="suspicious_job_report_action",
    ),

path(
    "administrator/employers/<int:employer_id>/reopen/",
    views.reopen_employer_verification,
    name="reopen_employer_verification",
),


path(
    "administrator/assessment-results/<int:result_id>/lime/",
    views.admin_lime_explanation,
    name="admin_lime_explanation",
),

path(
    "assessment-results/<int:result_id>/",
    views.assessment_results,
    name="assessment_results",
),
   path("logout/", views.logout_view, name="logout"),

   path(
    "administrator/reports/<int:report_id>/evidence/",
    views.suspicious_job_report_evidence,
    name="suspicious_job_report_evidence",
),

path(
    "administrator/users/<int:user_id>/manage/",
    views.manage_user_account,
    name="manage_user_account"
),
]