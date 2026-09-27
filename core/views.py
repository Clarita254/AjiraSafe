from django.shortcuts import render

from .models import Testimonial
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

def about(request):
    return render(request, "core/about.html")



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

def services(request):
    return render(request, "core/services.html")


def signup(request):
    return render(request, "core/signup.html")


def login_view(request):
    return render(request, "core/login.html")


def forgot_password(request):
    return render(request, "core/forgot_password.html")