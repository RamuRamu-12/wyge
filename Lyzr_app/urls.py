
from django.urls import path
from . import views

urlpatterns = [
    path('environment/create', views.create_environment, name='create_environment'),
    path('environment/<int:environment_id>', views.read_environment, name='read_environment'),
    path('environment/update/<int:environment_id>', views.update_environment, name='update_environment'),
    path('environment/delete/<int:environment_id>', views.delete_environment, name='delete_environment'),
    path('environments', views.read_all_environments, name='read_all_environments'),
    path('agent/create', views.create_agent, name='create_agent'),
    path('agent/<int:agent_id>', views.read_agent, name='read_agent'),
    path('agent/update/<int:agent_id>', views.update_agent, name='update_agent'),
    path('agent/delete/<int:agent_id>', views.delete_agent, name='delete_agent'),
    path('agents/', views.read_all_agents, name='read_all_agents'),
    path('openai/env', views.create_openai_environment_api, name='openai_env_creation'),
    path('openai/run', views.run_openai_environment, name='openai_env_creation'),

]
