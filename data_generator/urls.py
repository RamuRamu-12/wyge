from django.urls import path
from .views import excel_to_synthetic,text_to_synthetic,fill_missing_data

urlpatterns = [
    path('excel_to_synthetic', excel_to_synthetic, name='excel_to_synthetic'),
    path('text_to_synthetic', text_to_synthetic, name='text_to_synthetic'),
    path('filling_missing_data', fill_missing_data, name='filling_missing_data'),
]
