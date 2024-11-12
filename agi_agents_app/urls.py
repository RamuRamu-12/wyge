# agents/urls.py

from django.urls import path
from .views import registerPage, get_user_details, loginPage, googlelogin, agent_list, agent_detail, agent_create, \
    modify_agent, add_email, reject_agent, view_agents_by_status, create_submission_view, get_submission_view, \
    update_submission_view, delete_submission_view,get_all_submissions_view

urlpatterns = [
    path('register', registerPage, name='register'),
    path('get_user_data',get_user_details , name='get_user_details'),
    path('login', loginPage, name='login'),
    path('google_login', googlelogin, name='google_login'),

    path('agent_list', agent_list, name='agent-list'),
    path('agents_detail/<int:id>', agent_detail, name='agent-detail'),
    path('agents_create', agent_create, name='agent-create'),
    path('agent/<int:agent_id>/modify/', modify_agent , name='modify_agent'),
    path('agent/<int:agent_id>/delete/', reject_agent , name='delete_agent'),
    path('agent_status/', view_agents_by_status , name='view agent by status'),
    path('add_email', add_email, name='add_email'),
    path('submission/create/',create_submission_view, name='create_submission'),
    path('submission/<int:submission_id>/',get_submission_view, name='get_submission'),
    path('submission/<int:submission_id>/update/',update_submission_view, name='update_submission'),
    path('submission/<int:submission_id>/delete/',delete_submission_view, name='delete_submission'),
    path('submission/all/',get_all_submissions_view, name='delete_submission'),



]
