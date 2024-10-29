# Environment Creation
import os
from datetime import datetime

import requests
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from vyzeai.agents.react_agent import Agent
from vyzeai.models.openai import ChatOpenAI
from vyzeai.tools.prebuilt_tools import execute_query

from .database import PostgreSQLDB

db = PostgreSQLDB(dbname='uibmogli', user='uibmogli', password='8ogImHfL_1G249lXtM3k2EAIWTRDH2mX')


# Create environment
@api_view(['POST'])
def create_environment(request):
    try:
        data = request.data
        name = data.get('name')
        # model_vendor = data.get('model_vendor')
        api_key = data.get('api_key')
        model = data.get('model')
        temperature = data.get('temperature', 0.5)
        # top_p = data.get('top_p', 0.9)

        environment_id = db.create_environment(name, api_key, model, temperature)
        return Response({"environment_id": environment_id}, status=201)
    except Exception as e:
        return Response({"error": str(e)}, status=400)


# Read environment by ID
# Read environment by ID
@api_view(['GET'])
def read_environment(request, environment_id):
    try:
        environment = db.read_environment(environment_id)
        if environment:
            response_data = {

                "features": [],  # Assuming no features are provided, keeping it empty
                "llm_config": {
                    # "provider": environment[2],  # model_vendor
                    "model": environment[3],  # model
                    "config": {
                        "temperature": environment[4],  # temperature
                        # "top_p": environment[6],  # top_p
                    }
                },
                "env": {
                    "Environment_name": environment[1],  # name
                    "OPENAI_API_KEY": environment[2],  # API_KEY
                },

            }

            return Response(response_data, status=200)

        return Response({"error": "Environment not found"}, status=404)
    except Exception as e:
        return Response({"error": str(e)}, status=400)


# Update environment by ID
@api_view(['POST'])
def update_environment(request, environment_id):
    try:
        data = request.data
        name = data.get('name')
        # model_vendor = data.get('model_vendor')
        api_key = data.get('api_key')
        model = data.get('model')
        temperature = data.get('temperature')
        # top_p = data.get('top_p')

        updated_rows = db.update_environment(
            environment_id,
            name,
            api_key,
            model,
            temperature

        )

        if updated_rows:
            return Response({"message": f"Environment with ID {environment_id} updated successfully."}, status=200)
        return Response({"message": f"Environment with ID {environment_id} updated successfully."}, status=200)
    except Exception as e:
        return Response({"error": str(e)}, status=400)


# Delete environment by ID
@api_view(['GET'])
def delete_environment(request, environment_id):
    try:
        deleted_rows = db.delete_environment(environment_id)
        if deleted_rows:
            return Response({"message": f"Environment with ID {environment_id} deleted successfully."}, status=200)
        return Response({"error": f"Environment with ID {environment_id} not found."}, status=404)
    except Exception as e:
        return Response({"error": str(e)}, status=400)


# Read all environments
@api_view(['GET'])
def read_all_environments(request):
    try:
        environments = db.read_all_environments()
        if environments:
            environment_list = []
            for environment in environments:
                environment_list.append({
                    "id": environment[0],
                    "name": environment[1],
                    # "model_vendor": environment[2],
                    "api_key": environment[2],
                    "model": environment[3],
                    "temperature": environment[4],
                    # "top_p": environment[6],

                })
            return Response(environment_list, status=200)
        return Response({"message": "No environments found."}, status=404)
    except Exception as e:
        return Response({"error": str(e)}, status=400)


from rest_framework.decorators import api_view
from rest_framework.response import Response


# Create Agent
@api_view(['POST'])
def create_agent(request):
    try:
        data = request.data
        name = data.get('name')
        system_prompt = data.get('system_prompt')
        agent_description = data.get('agent_description')
        tools = data.get('tools')
        upload_attachment = data.get('upload_attachment', False)  # Default value set to False
        env_id = data.get('env_id')

        if not env_id:
            return Response({"error": "Environment ID is required"}, status=400)

        # Create the agent in the database
        agent_id = db.create_agent(name, system_prompt, agent_description, tools, upload_attachment, env_id)

        return Response({"agent_id": agent_id}, status=201)
    except Exception as e:
        return Response({"error": str(e)}, status=400)


# Read Agent by ID
@api_view(['GET'])
def read_agent(request, agent_id):
    try:
        agent = db.read_agent(agent_id)
        if agent:
            return Response({
                "id": agent[0],
                "name": agent[1],
                "system_prompt": agent[2],
                "agent_description": agent[3],
                "tools": agent[4],
                "Additional_Features": {
                    "upload_attachment": agent[5],  # Include upload_excel

                },
                "env_id": agent[6]
            }, status=200)
        return Response({"error": "Agent not found"}, status=404)
    except Exception as e:
        return Response({"error": str(e)}, status=400)


# Update Agent by ID
@api_view(['POST'])
def update_agent(request, agent_id):
    try:
        data = request.data
        name = data.get('name')
        system_prompt = data.get('system_prompt')
        agent_description = data.get('agent_description')
        tools = data.get('tools')
        upload_attachment = data.get('upload_attachment')  # Handle upload_excel update

        env_id = data.get('env_id')

        # Update agent in the database
        db.update_agent(agent_id, name, system_prompt, agent_description, tools, upload_attachment, env_id)

        return Response({"message": f"Agent with ID {agent_id} updated successfully."}, status=200)
    except Exception as e:
        return Response({"error": str(e)}, status=400)


# Delete Agent by ID
@api_view(['GET'])
def delete_agent(request, agent_id):
    try:
        db.delete_agent(agent_id)
        return Response({"message": f"Agent with ID {agent_id} deleted successfully."}, status=204)
    except Exception as e:
        return Response({"error": str(e)}, status=400)


# Read all Agents
from rest_framework.decorators import api_view
from rest_framework.response import Response
import logging

# Set up logging
logger = logging.getLogger(__name__)


@api_view(['GET'])
def read_all_agents(request):
    try:
        # Fetch all agents from the database
        agents = db.get_all_agents()

        # Check if any agents are returned
        if not agents:
            return Response({"message": "No agents found"}, status=404)

        # Structure the agents' data for JSON response
        agents_data = [
            {
                "id": agent[0],
                "name": agent[1],
                "system_prompt": agent[2],
                "agent_description": agent[3],
                "tools": agent[4],
                "Additional_Features": {
                    "upload_attachment": agent[5],  # Include upload_excel

                },
                "env_id": agent[6]
            }
            for agent in agents
        ]

        return Response({"agents": agents_data}, status=200)

    except Exception as e:
        # Log the error for further investigation
        logger.error(f"Error fetching agents: {str(e)}")

        # Return a user-friendly error message
        return Response({"error": "An error occurred while fetching agents"}, status=500)


# Creation of the openai environment.
from rest_framework.decorators import api_view
from vyzeai.agents.prebuilt_agents import ResearchAgent, VideoAudioBlogAgent, YTBlogAgent, BlogAgent, LinkedInAgent, \
    VideoAgent, EmailAgent

from rest_framework.response import Response
from rest_framework import status
import requests
import pandas as pd

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import requests


# Main API to create an OpenAI environment
@api_view(['POST'])
def create_openai_environment_api(request):
    try:
        agent_id = request.data.get('agent_id')

        if not agent_id:
            return Response({"error": "Agent ID is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve the agent details from the database
        agent = db.read_agent(agent_id)
        if not agent:
            return Response({"error": "Agent not found"}, status=status.HTTP_404_NOT_FOUND)

        # Proceed with environment creation logic
        agent_details = {
            "name": agent[1],
            "system_prompt": agent[2],
            "agent_description": agent[3],
            "tools": agent[4],
            "env_id": agent[6]
        }

        # Retrieve API key from the environment table
        env_details = db.read_environment(agent_details['env_id'])
        if not env_details or not env_details[2]:
            return Response({"error": "API key not found in environment table"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        openai_api_key = env_details[2]

        # Create the OpenAI environment
        environment_response = create_openai_environment(agent_details, openai_api_key)

        if environment_response.get("success"):
            return Response({"message": "OpenAI environment created successfully.",
                             "details": environment_response},
                            status=status.HTTP_201_CREATED)
        else:
            return Response({"error": "Failed to create OpenAI environment."},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# Helper function to create OpenAI environment
def create_openai_environment(agent_details, openai_api_key):
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",  # Ensure this model exists or use a valid one
            "messages": [
                {"role": "system", "content": agent_details['system_prompt']},
                {"role": "user", "content": agent_details['agent_description']}
            ],
            "max_tokens": 1500
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            return {"success": True, "response": response.json()}
        else:
            return {"success": False, "error": response.text}

    except Exception as e:
        return {"success": False, "error": str(e)}


import base64


def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        return None  # Handle the case where the image path is invalid or the image doesn't exist


# Helper function to delete all images in a directory
def delete_images_in_directory(directory: str) -> None:
    """Delete all image files in the specified directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    for filename in os.listdir(directory):
        _, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            os.remove(os.path.join(directory, filename))


# Helper function to get all images in a directory
def get_images_in_directory(directory: str) -> list:
    """Return all image files in the specified directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_files = []
    for filename in os.listdir(directory):
        _, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            image_files.append(os.path.join(directory, filename))
    return image_files


import re
def extract_num_rows_from_prompt(prompt):
    """
    Extracts the number of rows from the user prompt.
    Assumes the number of rows is mentioned as "Generate X rows" or similar in the prompt.
    """
    match = re.search(r'(\d+)\s+rows', prompt, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def differentiate_url(url):
    """
    Differentiates between a general website URL and a YouTube URL.

    Parameters:
    - url (str): The URL to check.

    Returns:
    - str: "YouTube" if the URL is a YouTube link, otherwise "Website".
    """
    youtube_patterns = [
        r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/",  # matches youtube.com or youtu.be
    ]
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return "YouTube"
    return "Website"


import markdown
def markdown_to_html(md_text):
    html_text = markdown.markdown(md_text)
    return html_text


from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
import os
import base64
import shutil
import tempfile
from django.core.files.storage import default_storage
from django.conf import settings
from vyzeai.tools.raw_functions import excel_to_sql
from .plots import tools as plot_tools
from .generator import generate_synthetic_data, generate_data_from_text, fill_missing_data_in_chunk
from django.http import HttpResponse


@api_view(['POST'])
def run_openai_environment(request):
    try:
        agent_id = request.data.get('agent_id')
        user_prompt = request.data.get('prompt', '')
        url = request.data.get('url', '')
        file = request.FILES.get('file')  # File if attached

        # Retrieve agent details
        agent = db.read_agent(agent_id)

        # Retrieve the API key from the environment table using env_id
        env_details = db.read_environment(agent[6])  # agent[6] is env_id
        openai_api_key = env_details[2]

        # Process based on input type (prompt, URL, file)
        result = None
        response_data = {}

        # Delete existing image plots (optional)
        delete_images_in_directory(settings.BASE_DIR)

        # Define the list of tool IDs that should be treated as 'blog_post'
        BLOG_TOOL_IDS = ['blog_post', 'mail_blog', 'audio_blog', 'video_blog', 'youtube_blog']


        # Tool handling conditions
        if file and 'text_to_sql' in agent[4]:
            operation_type = 'text_to_sql'
            result = handle_excel_file_based_on_type(file, openai_api_key, user_prompt,operation_type)
            if isinstance(result, dict):
                if 'image_path' in result:
                    response_data["image_base64"] = image_to_base64(result["image_path"])
                if 'content' in result:
                    response_data["content"] = markdown_to_html(result["content"])

        if file and 'graph_to_sql' in agent[4]:
            operation_type_graph = 'graph_to_sql'
            result = handle_excel_file_based_on_type(file, openai_api_key, user_prompt,operation_type_graph)
            if isinstance(result, dict):
                if 'image_path' in result:
                    response_data["image_base64"] = image_to_base64(result["image_path"])
                if 'content' in result:
                    response_data["content"] = markdown_to_html(result["content"])

        if file and 'forecast_to_sql' in agent[4]:
            operation_type_forecast = 'forecast_to_sql'
            result = handle_excel_file_based_on_type(file, openai_api_key, user_prompt,operation_type_forecast)
            if isinstance(result, dict):
                if 'image_path' in result:
                    response_data["image_base64"] = image_to_base64(result["image_path"])
                if 'content' in result:
                    response_data["content"] = markdown_to_html(result["content"])

        elif url and user_prompt:
            url_type = differentiate_url(url)


            # Handling based on URL type
            if url_type == "YouTube":
                if any(tool_id in agent[4] for tool_id in BLOG_TOOL_IDS):
                    result = generate_blog_from_yt_url(user_prompt, url, 'blog_post', openai_api_key)
                elif 'linkedin_post' in agent[4]:
                    result = generate_blog_from_yt_url(user_prompt, url, 'linkedin_post', openai_api_key)
            else:  # General Website URL
                if any(tool_id in agent[4] for tool_id in BLOG_TOOL_IDS):
                    result = generate_blog_from_url(user_prompt, url, 'blog_post', openai_api_key)
                elif 'linkedin_post' in agent[4]:
                    result = generate_blog_from_url(user_prompt, url, 'linkedin_post', openai_api_key)

            if isinstance(result, dict):
                response_data["content"] = markdown_to_html(result.get("content", ""))
                if "image_path" in result and result["image_path"]:
                    response_data["image_base64"] = image_to_base64(result["image_path"])

        elif file and user_prompt:
            if any(tool_id in agent[4] for tool_id in BLOG_TOOL_IDS):
                result = generate_blog_from_file(user_prompt, file, 'blog_post', openai_api_key)
            elif 'linkedin_post' in agent[4]:
                result = generate_blog_from_file(user_prompt, file, 'linkedin_post', openai_api_key)

            if isinstance(result, dict):
                response_data["content"] = markdown_to_html(result.get("content", ""))
                if "image_path" in result and result["image_path"]:
                    response_data["image_base64"] = image_to_base64(result["image_path"])


        # Synthetic data handling cases
        if file and 'synthetic_data_new_data' in agent[4]:
            result = handle_synthetic_data_for_new_data(file, user_prompt, openai_api_key)
            response_data["csv_file"] = result  # Assume result contains CSV file path

        elif file and 'synthetic_data_extended_data' in agent[4]:
            result = handle_synthetic_data_generation(file, user_prompt, openai_api_key)
            response_data["csv_file"] = result

        elif file and 'synthetic_data_missing_data' in agent[4]:
            result = handle_fill_missing_data(file, openai_api_key)
            response_data["csv_file"] = result

        # Construct response
        if response_data:
            return Response(response_data, status=status.HTTP_200_OK)
        else:
            return Response({"error": "No valid tool found for the given input."}, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# # Helper function to process the prompt through OpenAI
# def send_prompt_to_openai(api_key, agent, user_prompt):
#     try:
#         oi_url = "https://api.openai.com/v1/chat/completions"
#         headers = {
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "model": "gpt-4o-mini",  # Use appropriate model
#             "messages": [
#                 {"role": "system", "content": agent[2]},  # System prompt from agent details
#                 {"role": "user", "content": user_prompt}  # User's input prompt
#             ],
#             "max_tokens": 1500
#         }
#         response = requests.post(oi_url, headers=headers, json=payload)
#         if response.status_code == 200:
#             response_data = response.json()
#             content = response_data['choices'][0]['message']['content']
#             total_tokens = response_data['usage']['total_tokens']
#             return {
#                 "success": True,
#                 "content": content,
#                 "total_tokens": total_tokens
#             }
#         else:
#             return {"success": False, "error": response.text}
#
#     except Exception as e:
#         return {"success": False, "error": str(e)}
#



# Generate content from URL (for blog or LinkedIn post)
def generate_blog_from_url(prompt, url, option, api_key):
    try:
        if option == 'blog_post':
            print(datetime.now())
            research_agent = ResearchAgent(api_key)
            blog_agent = BlogAgent(api_key)
            context = research_agent.research(prompt, url)

            print(datetime.now())
            blog, doc_file, image = blog_agent.generate_blog(prompt, url, context)
            # content = contents[0][0]
            # image_path = contents[-1][-1][0]
            return {"content": blog, "image_path": image}
        elif option == 'linkedin_post':
            linkedin_agent = LinkedInAgent(api_key)
            research_agent = ResearchAgent(api_key)
            context = research_agent.research(prompt, url)
            content, image_path = linkedin_agent.generate_linkedin_post(context)
            return {"content": content, "image_path": image_path}
    except Exception as e:
        return {"error": str(e)}


# Generate content from URL (for blog or LinkedIn post)
def generate_blog_from_yt_url(prompt, url, option, api_key):
    try:
        if option == 'blog_post':
            yt_agent = YTBlogAgent(api_key)
            print(url,datetime.now())

            blog, doc_file, image = yt_agent.generate_blog(url)
            print("result", datetime.now())
            # content = contents[0][1]
            # image_path = contents[-1][-1][0]
            return {"content": blog, "image_path": image}
        elif option == 'linkedin_post':
            linkedin_agent = LinkedInAgent(api_key)
            yt_agent = YTBlogAgent(api_key)
            context = yt_agent.extract_transcript(url)
            content, image_path = linkedin_agent.generate_linkedin_post(context)
            return {"content": content, "image_path": image_path}
    except Exception as e:
        return {"error": str(e)}


# Generate content from file (for blog or LinkedIn post)
def generate_blog_from_file(prompt, file, option, api_key):
    try:
        file_path = save_file(file)

        if option == 'blog_post':
            va_agent = VideoAudioBlogAgent(api_key)
            blog, doc_file, image = va_agent.generate_blog(file_path)
            # content = contents[0][0]
            # image_path = contents[-1][-1][0]
            return {"content": blog, "image_path": image}
        elif option == 'linkedin_post':
            linkedin_agent = LinkedInAgent(api_key)
            va_agent = VideoAudioBlogAgent(api_key)
            context = va_agent.extract_text(file_path)
            content, image_path = linkedin_agent.generate_linkedin_post(context)
            return {"content": content, "image_path": image_path}

    except Exception as e:
        return {"error": str(e)}



def handle_synthetic_data_for_new_data(uploaded_file, user_prompt, openai_api_key):
    """
    Function to handle synthetic data generation.

    Parameters:
    - uploaded_file: The empty Excel or CSV file containing column names
    - user_prompt: The user's prompt, which should contain the number of rows
    - openai_api_key: The OpenAI API key to be used for generating synthetic data

    Returns:
    - A CSV string with generated synthetic data or an error message
    """
    try:
        # Determine file type and extract column names
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        print(file_extension)
        if file_extension == ".xlsx":
            df = pd.read_excel(uploaded_file)
        elif file_extension == ".csv":
            df = pd.read_csv(uploaded_file)
        else:
            return {"error": "Unsupported file format. Please upload an Excel or CSV file."}

        column_names = df.columns.tolist()

        # Check if the necessary information is provided
        if not user_prompt or not column_names:
            return {"error": "Missing user prompt or column names"}

        # Extract the number of rows from the prompt
        num_rows = extract_num_rows_from_prompt(user_prompt)
        if num_rows is None:
            return {"error": "Number of rows not found in the prompt"}

        # Generate synthetic data using the column names and the number of rows
        generated_df = generate_data_from_text(openai_api_key, user_prompt, column_names, num_rows=num_rows)

        # Convert the generated DataFrame to CSV format
        combined_csv = generated_df.to_csv(index=False)

        return {
            "data": combined_csv
        }

    except Exception as e:
        return {"error": str(e)}


def handle_synthetic_data_generation(file, user_prompt, openai_api_key):
    """
    Function to handle synthetic data generation and merging with original data.

    Parameters:
    - file: The uploaded Excel or CSV file containing original data
    - user_prompt: The user's prompt, which should contain the number of rows
    - openai_api_key: The OpenAI API key to be used for generating synthetic data

    Returns:
    - A dictionary with message and combined CSV data, or an error message
    """
    try:
        # Determine file type and read into a DataFrame
        file_extension = os.path.splitext(file.name)[1].lower()
        print(file_extension)
        if file_extension == ".xlsx":
            original_df = pd.read_excel(file)
        elif file_extension == ".csv":
            original_df = pd.read_csv(file)
            print(original_df)
        else:
            return {"error": "Unsupported file format. Please upload an Excel or CSV file."}

        # Check if the DataFrame has more than 300 rows
        if len(original_df) > 300:
            original_df = original_df.head(300)  # Take the first 300 rows

        # Create a temporary file for the data
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file_name = temp_file.name
            # Save the truncated or original data to a temporary location
            if file_extension == ".xlsx":
                original_df.to_excel(temp_file_name, index=False)
            elif file_extension == ".csv":
                original_df.to_csv(temp_file_name, index=False)

        # Extract the number of rows from the prompt
        num_rows = extract_num_rows_from_prompt(user_prompt)
        if num_rows is None:
            return {"error": "Number of rows not found in the prompt"}

        # Generate synthetic data using the temporary file path
        generated_df = generate_synthetic_data(openai_api_key, temp_file_name, num_rows)

        # Combine the original and synthetic data
        combined_df = pd.concat([original_df, generated_df], ignore_index=True)

        # Convert to CSV for download
        combined_csv = combined_df.to_csv(index=False)

        return {
            "data": combined_csv
        }

    except Exception as e:
        return {"error": str(e)}


def handle_fill_missing_data(file, openai_api_key):
    """
    Function to handle filling missing data in chunks and returning as CSV.

    Parameters:
    - file: The uploaded Excel or CSV file containing original data
    - openai_api_key: The OpenAI API key to be used for generating missing data

    Returns:
    - A dictionary with a message and combined CSV data, or an error message
    """
    try:
        # Determine file type and read into a DataFrame
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == ".xlsx":
            original_df = pd.read_excel(file)
        elif file_extension == ".csv":
            original_df = pd.read_csv(file)
        else:
            return {"error": "Unsupported file format. Please upload an Excel or CSV file."}

        # Check if the DataFrame has more than 300 rows
        if len(original_df) > 300:
            original_df = original_df.head(300)  # Limit to the first 300 rows

        # Create a temporary file for the data
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file_name = temp_file.name
            # Save the truncated or original data to a temporary location
            if file_extension == ".xlsx":
                original_df.to_excel(temp_file_name, index=False)
            elif file_extension == ".csv":
                original_df.to_csv(temp_file_name, index=False)

        # Fill missing data using the temporary file path
        filled_df = fill_missing_data_in_chunk(openai_api_key, temp_file_name)

        # Convert the filled DataFrame to CSV format for output
        combined_csv = filled_df.to_csv(index=False)

        return {
            "data": combined_csv
        }

    except Exception as e:
        return {"error": str(e)}


import os
from django.core.files.storage import default_storage
from django.http import HttpResponse
from vyzeai.models.openai import ChatOpenAI
from vyzeai.agents.react_agent import Agent
from vyzeai.tools.prebuilt_tools import execute_query, execute_code, install_library
from vyzeai.tools.raw_functions import excel_to_sql, get_metadata
from .system_prompt1 import reAct_prompt
from .system_prompt2 import plot_prompt
from .system_prompt3 import forecasting_prompt

# Shared database credentials
USER = 'uibcedotbqcywunfl752'
PASSWORD = 'LrdjP9dvLV0GP8PWRDmvREDB9IxmGu'
HOST = 'by80v7itmu1gw3kjmblq-postgresql.services.clever-cloud.com:50013'
DATABASE = 'by80v7itmu1gw3kjmblq'


def handle_excel_file_based_on_type(file, openai_api_key, user_prompt, operation_type):
    """
    Processes an uploaded Excel file by storing it in the database, converting it to SQL,
    and generating output based on the operation type (text, graph, forecast).

    Parameters:
    - file: File object of the uploaded Excel file
    - openai_api_key: OpenAI API key for model access
    - user_prompt: Query to be processed by the AI agent
    - operation_type: Specifies the operation type ('text_to_sql', 'graph_to_sql', 'forecast_to_sql')

    Returns:
    - HttpResponse with HTML content containing the generated text and image (if available)
    """
    table_name = file.name.split('.')[0]

    try:
        print(datetime.now())
        # Save the uploaded file to a storage path
        file_path = os.path.join(settings.MEDIA_ROOT, file.name)
        with default_storage.open(file_path, 'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)
        print(datetime.now())
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(datetime.now())
        # Convert the Excel file to SQL table
        excel_to_sql(file_path, table_name, USER, PASSWORD, HOST, DATABASE)
        print(datetime.now())

        print("#########################################")

        # Initialize the LLM model with appropriate prompt based on operation type
        llm = ChatOpenAI(memory=True, api_key=openai_api_key)

        # Choose the appropriate tools and prompt based on the operation type
        if operation_type == 'text_to_sql':
            print(datetime.now())
            tools = [execute_query()]
            prompt = reAct_prompt
            print(datetime.now())
        elif operation_type == 'graph_to_sql':
            print(datetime.now())
            tools = [execute_query(), execute_code(), install_library()]
            prompt = plot_prompt
            print(datetime.now())
        elif operation_type == 'forecast_to_sql':
            print(datetime.now())
            tools = [execute_query(), execute_code(), install_library()]
            prompt = forecasting_prompt
            print(datetime.now())
        else:
            return HttpResponse("Invalid operation type specified.", status=400)

        # Set up agent
        agent = Agent(llm, tools, react_prompt=prompt)
        metadata = get_metadata(HOST, USER, PASSWORD, DATABASE, [table_name])
        command = f"""
            user = '{USER}'
            password = '{PASSWORD}'
            host = '{HOST}'
            database = '{DATABASE}'
            tables related to user are: {table_name}
            Metadata of the tables: {metadata}
            User query: {user_prompt}
        """

        # Get agent response
        response = agent(command)
        response = response.split('**Answer**:')[-1]

        # Fetch images from the directory
        images = get_images_in_directory(settings.BASE_DIR)
        print(images)

        # Prepare the result dictionary
        result = {"content": response}
        if images:
            result["image_path"] = images[0]  # Only add image_path if images are available

        return result

    except Exception as e:
        return {"error": str(e)}


# Helper function to save uploaded file
def save_file(file):
    directory = 'temp'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())
    return file_path
