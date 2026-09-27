from django.db import models
from django.contrib.auth.models import User


class Testimonial(models.Model):

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="testimonials"
    )

    feedback = models.TextField()

    role = models.CharField(
        max_length=100,
        blank=True
    )

    rating = models.PositiveSmallIntegerField(
        default=5
    )

    is_approved = models.BooleanField(
        default=False
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.username} - {self.rating}/5"