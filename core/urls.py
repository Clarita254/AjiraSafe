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
path("admin-dashboard/", views.admin_dashboard, name="admin_dashboard"),

path("logout/", views.logout_view, name="logout"),
]