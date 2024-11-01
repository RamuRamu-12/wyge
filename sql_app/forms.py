from django import forms


class UploadFileForm(forms.Form):
    api_key = forms.CharField(max_length=255, label="OpenAI API Key")
    user = forms.CharField(max_length=255, label="Database User")
    password = forms.CharField(widget=forms.PasswordInput, label="Database Password")
    host = forms.CharField(max_length=255, label="Database Host")
    database = forms.CharField(max_length=255, label="Database Name")

    # Single file input for uploading Excel files
    file = forms.FileField(widget=forms.FileInput, label="Upload Excel File", required=False)
