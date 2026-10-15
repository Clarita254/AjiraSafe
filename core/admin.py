
from django.contrib import admin
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




@admin.register(Testimonial)
class TestimonialAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "role",
        "rating",
        "status",
        "is_approved",
        "reviewed_by",
        "created_at",
    )
    list_filter = (
        "status",
        "is_approved",
        "rating",
        "created_at",
    )
    search_fields = (
        "user__username",
        "user__email",
        "user__first_name",
        "user__last_name",
        "feedback",
        "admin_comment",
    )
    readonly_fields = (
        "created_at",
        "reviewed_at",
    )
    ordering = ("-created_at",)



@admin.register(JobApplicant)
class JobApplicantAdmin(admin.ModelAdmin):
    list_display = ("user", "phone_number", "created_at", "updated_at")
    search_fields = ("user__username", "user__email", "phone_number")
    ordering = ("-created_at",)


@admin.register(SystemAdministrator)
class SystemAdministratorAdmin(admin.ModelAdmin):
    list_display = ("user", "created_at", "updated_at")
    search_fields = ("user__username", "user__email")
    ordering = ("-created_at",)


@admin.register(Employer)
class EmployerAdmin(admin.ModelAdmin):
    list_display = (
        "company_name",
        "email_domain",
        "contact_email",
        "created_at",
    )
    search_fields = ("company_name", "email_domain", "contact_email")
    ordering = ("company_name",)


@admin.register(JobAdvertisement)
class JobAdvertisementAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "employer",
        "applicant",
        "status",
        "submission_date",
    )
    list_filter = ("status", "employment_type", "submission_date")
    search_fields = (
        "title",
        "employer__company_name",
        "description",
        "location",
    )
    ordering = ("-submission_date",)


@admin.register(EmployerVerificationRecord)
class EmployerVerificationRecordAdmin(admin.ModelAdmin):
    list_display = (
        "employer",
        "verification_status",
        "verified_by",
        "verification_date",
    )
    list_filter = ("verification_status", "verification_date")
    search_fields = ("employer__company_name", "remarks")
    ordering = ("-verification_date",)


@admin.register(SuspiciousJobReport)
class SuspiciousJobReportAdmin(admin.ModelAdmin):
    list_display = (
        "report_id",
        "job",
        "applicant",
        "reason",
        "status",
        "report_date",
    )
    list_filter = ("status", "report_date")
    search_fields = (
        "reason",
        "description",
        "job__title",
        "applicant__user__username",
    )
    ordering = ("-report_date",)





@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = (
        "model_name",
        "model_type",
        "model_version",
        "trained_on",
        "accuracy",
    )
    search_fields = ("model_name", "model_type", "model_version")
    ordering = ("-trained_on",)


@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = (
        "result_id",
        "job",
        "model",
        "prediction",
        "analysis_date",
    )
    list_filter = ("prediction", "analysis_date")
    search_fields = ("job__title", "model__model_name")
    ordering = ("-analysis_date",)


@admin.register(RiskAssessment)
class RiskAssessmentAdmin(admin.ModelAdmin):
    list_display = ("risk_id", "result", "risk_score", "risk_level", "created_at")
    list_filter = ("risk_level", "created_at")
    ordering = ("-created_at",)


@admin.register(Explanation)
class ExplanationAdmin(admin.ModelAdmin):
    list_display = ("explanation_id", "result", "method", "created_at")
    list_filter = ("method", "created_at")
    search_fields = ("explanation_text", "key_contribution_features")
    ordering = ("-created_at",)
