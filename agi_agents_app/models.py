from django.db import models

class Agent(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    category = models.CharField(max_length=50, null=True, blank=True)
    industry = models.CharField(max_length=50, null=True, blank=True)
    pricing = models.CharField(max_length=20, null=True, blank=True)
    accessory_model = models.CharField(max_length=20, null=True, blank=True)
    website_url = models.URLField(max_length=200, null=True, blank=True)
    email = models.EmailField(max_length=150, null=True, blank=True)
    date_added = models.DateTimeField(auto_now_add=True)
    tagline = models.CharField(max_length=255, null=True, blank=True)
    likes = models.PositiveIntegerField(default=0, null=True, blank=True)
    overview = models.TextField(null=True, blank=True)
    key_features = models.TextField(null=True, blank=True)
    use_cases = models.TextField(null=True, blank=True)
    created_by = models.CharField(max_length=255, null=True, blank=True)
    access = models.CharField(max_length=50, null=True, blank=True)
    tags = models.TextField(null=True, blank=True)
    preview_image = models.URLField(max_length=500, null=True, blank=True)
    logo = models.URLField(max_length=500, null=True, blank=True)
    demo_video = models.URLField(max_length=500, null=True, blank=True)
    is_approved = models.BooleanField(default=False)  # New field

    def __str__(self):
        return self.name

