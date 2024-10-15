from django import forms
from .models import Agent


class AgentForm(forms.ModelForm):
    class Meta:
        model = Agent
        fields = [
            'name', 
            'description', 
            'category', 
            'industry', 
            'pricing', 
            'accessory_model', 
            'website_url', 
            'email',
            'tagline',
            'likes',
            'overview',
            'key_features',
            'use_cases',
            'created_by',
            'access',
            'tags',
            'preview_image',
            'logo',
            'demo_video'
        ]


from django import forms

class AgentUpdateForm(forms.Form):
    name = forms.CharField(max_length=255, required=False)
    description = forms.CharField(widget=forms.Textarea, required=False)
    category = forms.CharField(max_length=255, required=False)
    industry = forms.CharField(max_length=255, required=False)
    pricing = forms.CharField(max_length=255, required=False)
    accessory_model = forms.CharField(max_length=255, required=False)
    website_url = forms.URLField(required=False)
    email = forms.EmailField(required=False)
    tagline = forms.CharField(max_length=255, required=False)
    likes = forms.IntegerField(required=False)
    overview = forms.CharField(widget=forms.Textarea, required=False)
    key_features = forms.CharField(required=False)  # Comma-separated values
    use_cases = forms.CharField(required=False)  # Comma-separated values
    created_by = forms.CharField(max_length=255, required=False)
    access = forms.CharField(max_length=255, required=False)
    tags = forms.CharField(required=False)  # Comma-separated values
    preview_image = forms.URLField(required=False)
    logo = forms.URLField(required=False)
    demo_video = forms.URLField(required=False)
    is_approved = forms.BooleanField(required=False)  # Keep existing approval status if needed
