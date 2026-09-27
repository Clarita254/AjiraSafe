from django.contrib import admin
from .models import Testimonial


@admin.register(Testimonial)
class TestimonialAdmin(admin.ModelAdmin):

    list_display = (
        "user",
        "role",
        "rating",
        "is_approved",
        "created_at",
    )

    list_filter = (
        "is_approved",
        "rating",
        "created_at",
    )

    search_fields = (
        "user__username",
        "user__first_name",
        "user__last_name",
        "feedback",
    )

    list_editable = (
        "is_approved",
    )

    ordering = (
        "-created_at",
    )