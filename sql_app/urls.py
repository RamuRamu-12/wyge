from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_excel, name='upload'),
    path('query/', views.execute_query_view, name='query'),
]
