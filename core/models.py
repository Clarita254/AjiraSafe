
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator


# =========================================================
# EXISTING MODEL: JOB APPLICANT
# =========================================================

class JobApplicant(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="job_applicant"
    )
    phone_number = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.user.get_full_name() or self.user.username


# =========================================================
# SYSTEM ADMINISTRATOR PROFILE
# =========================================================

class SystemAdministrator(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="system_administrator"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.user.get_full_name() or self.user.username


# =========================================================
# EMPLOYER
# =========================================================

class Employer(models.Model):
    employer_id = models.AutoField(primary_key=True)

    company_name = models.CharField(max_length=255)
    email_domain = models.CharField(max_length=150, unique=True)

    website = models.URLField(max_length=255, blank=True)
    company_reg_number = models.CharField(max_length=255, blank=True)
    contact_email = models.EmailField(max_length=150, blank=True)
    contact_phone = models.CharField(max_length=20, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.company_name


# =========================================================
# JOB ADVERTISEMENT
# =========================================================

class JobAdvertisement(models.Model):

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        ANALYZED = "analyzed", "Analyzed"
        FLAGGED = "flagged", "Flagged"

    job_id = models.AutoField(primary_key=True)

    applicant = models.ForeignKey(
        JobApplicant,
        on_delete=models.PROTECT,
        related_name="job_advertisements"
    )
    employer = models.ForeignKey(
        Employer,
        on_delete=models.PROTECT,
        related_name="job_advertisements"
    )

    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    location = models.CharField(max_length=150, blank=True)
    department = models.CharField(max_length=150, blank=True)
    salary_range = models.CharField(max_length=150, blank=True)
    company_profile = models.TextField(blank=True)
    requirements = models.TextField(blank=True)
    benefits = models.TextField(blank=True)

    telecommuting = models.BooleanField(default=False)
    has_company_logo = models.BooleanField(default=False)
    has_questions = models.BooleanField(default=False)

    employment_type = models.CharField(max_length=100, blank=True)
    required_experience = models.CharField(max_length=100, blank=True)
    required_education = models.CharField(max_length=100, blank=True)
    industry = models.CharField(max_length=150, blank=True)
    function = models.CharField(max_length=150, blank=True)

    submission_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )
    updated_at = models.DateTimeField(auto_now=True)




# =========================================================
# EMPLOYER VERIFICATION RECORD
# =========================================================

class EmployerVerificationRecord(models.Model):

    class VerificationStatus(models.TextChoices):
        VERIFIED = "verified", "Verified"
        UNVERIFIED = "unverified", "Unverified"
        SUSPICIOUS = "suspicious", "Suspicious"

    verification_id = models.AutoField(primary_key=True)

    employer = models.ForeignKey(
        Employer,
        on_delete=models.PROTECT,
        related_name="verification_records"
    )
    verified_by = models.ForeignKey(
        SystemAdministrator,
        on_delete=models.PROTECT,
        related_name="verification_records"
    )

    verification_status = models.CharField(
        max_length=20,
        choices=VerificationStatus.choices,
        default=VerificationStatus.UNVERIFIED
    )
    verification_date = models.DateTimeField()
    remarks = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-verification_date"]

    def __str__(self):
        return (
            f"{self.employer.company_name} - "
            f"{self.verification_status}"
        )


# =========================================================
# SUSPICIOUS JOB REPORT
# =========================================================

class SuspiciousJobReport(models.Model):

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        REVIEWED = "reviewed", "Reviewed"
        ACTION_TAKEN = "action_taken", "Action Taken"
        DISMISSED = "dismissed", "Dismissed"

    report_id = models.AutoField(primary_key=True)

    job = models.ForeignKey(
        JobAdvertisement,
        on_delete=models.PROTECT,
        related_name="suspicious_reports"
    )
    applicant = models.ForeignKey(
        JobApplicant,
        on_delete=models.PROTECT,
        related_name="suspicious_reports"
    )

    reason = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    report_date = models.DateTimeField(auto_now_add=True)

    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )

    reviewed_by = models.ForeignKey(
        SystemAdministrator,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="reviewed_suspicious_reports"
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)
    admin_comment = models.TextField(blank=True)

    class Meta:
        ordering = ["-report_date"]

    def __str__(self):
        return f"Report #{self.report_id} - {self.status}"




# =========================================================
# MACHINE LEARNING MODEL METADATA
# =========================================================

class MLModel(models.Model):
    model_id = models.AutoField(primary_key=True)

    model_name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=50)
    model_version = models.CharField(max_length=50)
    trained_on = models.DateField()

    accuracy = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        null=True,
        blank=True,
        validators=[
            MinValueValidator(0),
            MaxValueValidator(100)
        ]
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["model_name", "model_version"],
                name="unique_ml_model_version"
            )
        ]

    def __str__(self):
        return f"{self.model_name} v{self.model_version}"


# =========================================================
# ANALYSIS RESULT
# =========================================================

class AnalysisResult(models.Model):

    class Prediction(models.TextChoices):
        LEGITIMATE = "legitimate", "Legitimate"
        FRAUDULENT = "fraudulent", "Fraudulent"

    result_id = models.AutoField(primary_key=True)

    job = models.ForeignKey(
        JobAdvertisement,
        on_delete=models.PROTECT,
        related_name="analysis_results"
    )
    model = models.ForeignKey(
        MLModel,
        on_delete=models.PROTECT,
        related_name="analysis_results"
    )

    prediction = models.CharField(
        max_length=20,
        choices=Prediction.choices
    )

    prediction_probability = models.DecimalField(
        max_digits=5,
        decimal_places=4,
        null=True,
        blank=True,
        validators=[
            MinValueValidator(0),
            MaxValueValidator(1)
        ]
    )

    analysis_date = models.DateTimeField(auto_now_add=True)
    processed_text_length = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-analysis_date"]

    def __str__(self):
        return f"Analysis #{self.result_id} - {self.prediction}"


# =========================================================
# RISK ASSESSMENT
# =========================================================

class RiskAssessment(models.Model):

    class RiskLevel(models.TextChoices):
        LOW = "low", "Low"
        MEDIUM = "medium", "Medium"
        HIGH = "high", "High"
        VERY_HIGH = "very_high", "Very High"

    risk_id = models.AutoField(primary_key=True)

    result = models.OneToOneField(
        AnalysisResult,
        on_delete=models.CASCADE,
        related_name="risk_assessment"
    )

    risk_score = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        validators=[
            MinValueValidator(0),
            MaxValueValidator(100)
        ]
    )

    risk_level = models.CharField(
        max_length=20,
        choices=RiskLevel.choices
    )

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Risk assessment #{self.risk_id} - {self.risk_level}"


# =========================================================
# EXPLANATION
# =========================================================

class Explanation(models.Model):
    explanation_id = models.AutoField(primary_key=True)

    result = models.ForeignKey(
        AnalysisResult,
        on_delete=models.CASCADE,
        related_name="explanations"
    )

    explanation_text = models.TextField()
    key_contribution_features = models.TextField(blank=True)
    method = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Explanation #{self.explanation_id} ({self.method})"


# =========================================================
# EXISTING MODEL: TESTIMONIAL
# =========================================================


class Testimonial(models.Model):

    class Status(models.TextChoices):
        PENDING = "pending", "Awaiting Review"
        UNDER_REVIEW = "under_review", "Under Review"
        APPROVED = "approved", "Approved"

    class Category(models.TextChoices):
        USABILITY = "usability", "Website usability"
        ANALYSIS = "analysis", "Job analysis or risk explanation"
        EMPLOYER_VERIFICATION = (
            "employer_verification",
            "Employer verification",
        )
        TECHNICAL_ISSUE = "technical_issue", "Technical issue"
        SUGGESTION = "suggestion", "Suggestion for improvement"
        OTHER = "other", "Other"

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="testimonials",
    )
    feedback = models.TextField()

    category = models.CharField(
        max_length=30,
        choices=Category.choices,
        default=Category.OTHER,
    )

    rating = models.PositiveSmallIntegerField(default=5)
    role = models.CharField(max_length=100, blank=True)

    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )

    is_approved = models.BooleanField(default=False)

    reviewed_by = models.ForeignKey(
        "SystemAdministrator",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="reviewed_testimonials",
    )

    reviewed_at = models.DateTimeField(null=True, blank=True)
    admin_comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user.username} - {self.rating}/5"


        # =========================================================
# TESTIMONIAL DECISION AUDIT LOG
# =========================================================

class TestimonialDecisionLog(models.Model):
    testimonial = models.ForeignKey(
        Testimonial,
        on_delete=models.CASCADE,
        related_name="decision_logs"
    )

    administrator = models.ForeignKey(
        "SystemAdministrator",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="testimonial_decision_logs"
    )

    action = models.CharField(max_length=30)

    previous_status = models.CharField(max_length=20)

    previous_is_approved = models.BooleanField()

    new_status = models.CharField(max_length=20)

    new_is_approved = models.BooleanField()

    reason = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return (
            f"Feedback #{self.testimonial_id} - "
            f"{self.action} - {self.created_at}"
        )