from django.urls import path
from .views import generate_data, send_email

urlpatterns = [
    path('generate-data/', generate_data, name='generate_data'),
    path('send-email/', send_email, name='send_email'),
]
