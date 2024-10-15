# agents/urls.py

from django.urls import path
from .views import agent_list, agent_detail, agent_create, modify_agent,add_email,delete,create_submission_view,get_submission_view,update_submission_view,delete_submission_view

urlpatterns = [
    path('agent_list', agent_list, name='agent-list'),
    path('agents_detail/<int:id>', agent_detail, name='agent-detail'),
    path('agents_create', agent_create, name='agent-create'),
    path('agent/<int:agent_id>/modify/', modify_agent , name='modify_agent'),
    path('agent/<int:agent_id>/delete/', delete , name='delete_agent'),
    path('add_email', add_email, name='add_email'),
    path('submission/create/',create_submission_view, name='create_submission'),
    path('submission/<int:submission_id>/',get_submission_view, name='get_submission'),
    path('submission/<int:submission_id>/update/',update_submission_view, name='update_submission'),
    path('submission/<int:submission_id>/delete/',delete_submission_view, name='delete_submission'),


]
