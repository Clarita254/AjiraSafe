from django.contrib.auth.models import Group, User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from core.models import JobApplicant, Testimonial


class DashboardAuthenticationTests(SimpleTestCase):
    def test_applicant_dashboard_redirects_anonymous_users_to_login(self):
        response = self.client.get(reverse("dashboard"))

        self.assertRedirects(
            response,
            f"{reverse('login')}?next={reverse('dashboard')}",
            fetch_redirect_response=False,
        )

    def test_admin_dashboard_redirects_anonymous_users_to_login(self):
        response = self.client.get(reverse("admin_dashboard"))

        self.assertRedirects(
            response,
            f"{reverse('login')}?next={reverse('admin_dashboard')}",
            fetch_redirect_response=False,
        )


class PublicTestimonialsTests(TestCase):
    def test_home_shows_three_latest_approved_testimonials(self):
        approved_feedback = []
        for index in range(5):
            user = User.objects.create_user(
                username=f"testimonial-user-{index}",
                first_name=f"Testimonial User {index}",
            )
            feedback = f"Approved testimonial {index}"
            approved_feedback.append(feedback)
            Testimonial.objects.create(
                user=user,
                feedback=feedback,
                is_approved=True,
            )

        hidden_user = User.objects.create_user(
            username="unapproved-testimonial-user",
            first_name="Unapproved User",
        )
        Testimonial.objects.create(
            user=hidden_user,
            feedback="This testimonial is not approved",
            is_approved=False,
        )

        home_response = self.client.get(reverse("home"))
        full_page_response = self.client.get(reverse("testimonials"))

        self.assertEqual(len(home_response.context["testimonials"]), 3)
        self.assertSetEqual(
            {
                testimonial["feedback"]
                for testimonial in home_response.context["testimonials"]
            },
            set(approved_feedback[-3:]),
        )
        self.assertEqual(len(full_page_response.context["testimonials"]), 5)
        self.assertNotContains(home_response, "This testimonial is not approved")
        self.assertNotContains(
            full_page_response,
            "This testimonial is not approved",
        )


class AssessmentFormValidationTests(TestCase):
    def setUp(self):
        applicant_group = Group.objects.create(name="Job Applicant")
        user = User.objects.create_user(
            username="assessment-applicant",
            password="test-password",
        )
        user.groups.add(applicant_group)
        JobApplicant.objects.create(user=user, phone_number="1234567890")
        self.client.force_login(user)

    def test_invalid_assessment_stays_on_form_with_submitted_values(self):
        response = self.client.post(
            f"{reverse('dashboard')}?tab=overview",
            {
                "assessment_submit": "1",
                "title": "Senior Engineer",
                "company_name": "Example Employer",
                "location": "Nairobi",
                "job_url": "https://example.com/job",
                "website": "https://example.com",
                "company_reg_number": "REG-123",
                "contact_email": "jobs@example.com",
                "contact_phone": "+254 700 000 000",
                "telecommuting": "yes",
                "has_company_logo": "no",
                "has_questions": "yes",
                "salary_range": "KES 100,000",
                "employment_type": "Full-time",
                "required_experience": "3-5 years",
                "required_education": "Bachelor's degree",
                "industry": "Technology",
                "department": "Engineering",
                "function": "Software Engineering",
                "description": "Too short",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["active_tab"], "assess-job")
        for value in (
            "Senior Engineer",
            "Example Employer",
            "Nairobi",
            "https://example.com/job",
            "https://example.com",
            "REG-123",
            "jobs@example.com",
            "+254 700 000 000",
            "KES 100,000",
            "Technology",
            "Engineering",
            "Software Engineering",
        ):
            self.assertContains(response, f'value="{value}"')

        self.assertContains(
            response,
            'value="Bachelor&#x27;s degree"',
        )

        for selected_value in ("yes", "no", "Full-time", "3-5 years"):
            self.assertContains(
                response,
                f'value="{selected_value}" selected',
            )

        self.assertContains(response, ">Too short</textarea>")
        self.assertContains(response, "Job description must contain at least 50 characters.")
        self.assertContains(
            response,
            'const requestedTab = "assess\\u002Djob";',
        )
