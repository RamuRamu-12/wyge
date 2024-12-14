from django.urls import path
from .views import generate_linkedin_post, upload_file, post_on_linkedin, send_email

urlpatterns = [
    path('generate_linkedin_post/', generate_linkedin_post, name='generate_linkedin_post'),
    path('upload_file/', upload_file, name='upload_file'),
    path('post_on_linkedin/', post_on_linkedin, name='post_on_linkedin'),
    path('send_email/', send_email, name='send_email'),
]
