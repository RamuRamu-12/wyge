
from django.urls import path
from . import views,views_for_hana_rag

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
    path('send_email', views.send_email, name='sending email'),
    #path('openai/run_excel', views.run_openai_with_excel, name='openai_env_excel'),

    #HANA URLS
    path('processing_files', views_for_hana_rag.processing_files, name='processing_files'),
    path('query_making', views_for_hana_rag.query_system, name='query system'),

    #Dynamic agents
    path('dyn_create-agent', views.create_dynamic_agent, name='create_agent'),
    path('dyn_agents/<int:agent_id>', views.read_dynamic_agent, name='read_agent'),
    path('dyn_agents/<int:agent_id>/update', views.update_dynamic_agent, name='update_agent'),
    path('dyn_agents/<int:agent_id>/delete', views.delete_dynamic_agent, name='delete_agent'),
    path('dyn_agents/', views.read_all_dynamic_agents, name='read_all_agents'),
    #path('create-openai-environment/', create_openai_environment_api, name='create_openai_environment'),
    path('run-agent-environment', views.run_agent_environment, name='run_agent_environment'),




]
