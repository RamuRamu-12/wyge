from django.urls import path
from .views import generate_content, send_email,upload_file

urlpatterns = [
    path('generate-content/', generate_content, name='generate_content'),
    path('send-email/', send_email, name='send_email'),
    path('upload_file/', upload_file, name='upload_file'),
]
