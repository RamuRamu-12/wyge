#Environment Creation
import requests
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .database import PostgreSQLDB

db = PostgreSQLDB(dbname='uibmogli', user='uibmogli', password='8ogImHfL_1G249lXtM3k2EAIWTRDH2mX')


# Create environment
@api_view(['POST'])
def create_environment(request):
    try:
        data = request.data
        name = data.get('name')
        model_vendor = data.get('model_vendor')
        api_key = data.get('api_key')
        model = data.get('model')
        temperature = data.get('temperature', 0.5)
        top_p = data.get('top_p', 0.9)

        environment_id = db.create_environment(name, model_vendor, api_key, model, temperature, top_p)
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
                    "provider": environment[2],  # model_vendor
                    "model": environment[4],  # model
                    "config": {
                        "temperature": environment[5],  # temperature
                        "top_p": environment[6],  # top_p
                    }
                },
                "env": {
                    "Environment_name": environment[1],  # name
                    "OPENAI_API_KEY":environment[3],  #API_KEY
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
        model_vendor = data.get('model_vendor')
        api_key = data.get('api_key')
        model = data.get('model')
        temperature = data.get('temperature')
        top_p = data.get('top_p')

        updated_rows = db.update_environment(
            environment_id,
            name,
            model_vendor,
            api_key,
            model,
            temperature,
            top_p,
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
                    "model_vendor": environment[2],
                    "api_key": environment[3],
                    "model": environment[4],
                    "temperature": environment[5],
                    "top_p": environment[6],

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
            "Additional_Features":{
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
            "Additional_Features":{
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




#Creation of the openai environment.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

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

        # Create the OpenAI environment with the retrieved agent details
        agent_details = {
            "name": agent[1],
            "system_prompt": agent[2],
            "agent_description": agent[3],
            "tools":agent[4],
            "env_id": agent[6]
        }

        # Call the function to create the OpenAI environment
        environment_response = create_openai_environment(agent_details)

        # Check the response from the OpenAI environment creation
        if environment_response.get("success"):
            return Response({"message": "OpenAI environment created successfully.", "details": environment_response},
                            status=status.HTTP_201_CREATED)
        else:
            return Response({"error": "Failed to create OpenAI environment."},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# helper Function to create OpenAI environment
import requests
from decimal import Decimal

def create_openai_environment(agent_details):
    try:
        # Fetch the OpenAI API key from the environments table based on env_id
        env_id = agent_details["env_id"]
        env_details = db.read_environment(env_id)  # Returns a tuple

        if not env_details:
            raise ValueError("Environment details not found.")

        # Extract API key from the tuple (index 3)
        openai_api_key = env_details[3]

        if not openai_api_key:
            raise ValueError("OpenAI API key not found for the specified environment")

        # Use the OpenAI API key to create the environment
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",  # Ensure this model exists or use a valid one like "gpt-4"
            "messages": [
                {"role": "system", "content": agent_details['system_prompt']},
                {"role": "user", "content": agent_details['agent_description']}
            ],
            "max_tokens": 1500
        }

        # Make the POST request to OpenAI API using correct 'requests' module
        response = requests.post(url, headers=headers, json=payload)

        # Check response status
        if response.status_code == 200:
            return {"success": True, "response": response.json()}
        else:
            print(f"OpenAI API Error: {response.status_code} - {response.text}")
            return {"success": False, "error": response.text}

    except Exception as e:
        print(f"Error creating OpenAI environment: {e}")
        return {"success": False, "error": str(e)}


#Run Openai Environment
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import requests

@api_view(['POST'])
def run_openai_environment(request):
    try:
        # Retrieve the environment details using agent_id
        agent_id = request.data.get('agent_id')
        if not agent_id:
            return Response({"error": "Agent ID is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve agent and environment details
        agent = db.read_agent(agent_id)
        if not agent:
            return Response({"error": "Agent not found"}, status=status.HTTP_404_NOT_FOUND)

        # Extract prompt from user input
        user_prompt = request.data.get('prompt')
        if not user_prompt:
            return Response({"error": "User prompt is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Get environment details
        env_id = agent[6]
        env_details = db.read_environment(env_id)
        if not env_details:
            return Response({"error": "Environment details not found"}, status=status.HTTP_404_NOT_FOUND)

        # Extract the OpenAI API key
        openai_api_key = env_details[3]
        if not openai_api_key:
            return Response({"error": "OpenAI API key not found for the specified environment"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Use the OpenAI API key to send the user's prompt to the model
        model_response = send_prompt_to_openai(openai_api_key, agent, user_prompt)

        # Check response from OpenAI
        if model_response.get("success"):
            return Response({
                "content": model_response['content'],
                "total_tokens": model_response['total_tokens']
            }, status=status.HTTP_200_OK)
        else:
            return Response({"error": model_response['error']}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


#Helper function to run the prompt through openai
def send_prompt_to_openai(api_key, agent, user_prompt):
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Prepare the message format required by the OpenAI API
        payload = {
            "model": "gpt-4o-mini",  # Use the model stored in the environment
            "messages": [
                {"role": "system", "content": agent[2]},  # system_prompt from agent details
                {"role": "user", "content": user_prompt}  # user input prompt
            ],
            "max_tokens": 1500  # Modify as per the requirement
        }

        # Make the POST request to OpenAI API
        response = requests.post(url, headers=headers, json=payload)

        # Check if the response is successful
        if response.status_code == 200:
            response_data = response.json()
            # Extract only the content and number of tokens used
            content = response_data['choices'][0]['message']['content']
            total_tokens = response_data['usage']['total_tokens']
            return {
                "success": True,
                "content": content,
                "total_tokens": total_tokens
            }
        else:
            return {"success": False, "error": response.text}

    except Exception as e:
        return {"success": False, "error": str(e)}



#Uploading an excel file
import pandas as pd
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import parser_classes, api_view
from rest_framework.response import Response
from rest_framework import status
import requests

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])  # Allow multipart form data
def run_openai_with_excel(request):
    try:
        # Retrieve the environment details using agent_id
        agent_id = request.data.get('agent_id')
        if not agent_id:
            return Response({"error": "Agent ID is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve agent and environment details
        agent = db.read_agent(agent_id)
        if not agent:
            return Response({"error": "Agent not found"}, status=status.HTTP_404_NOT_FOUND)

        # Retrieve the uploaded Excel file
        excel_file = request.FILES.get('file')
        if not excel_file:
            return Response({"error": "Excel file is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve the user query (analysis type question)
        user_query = request.data.get('query')
        if not user_query:
            return Response({"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Read the Excel file into a pandas DataFrame
        try:
            df = pd.read_excel(excel_file)
        except Exception as e:
            return Response({"error": f"Error reading Excel file: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # You can process the DataFrame as needed before sending it to OpenAI
        data_summary = df.describe().to_string()  # Summary of the data

        # Get environment details
        env_id = agent[6]
        env_details = db.read_environment(env_id)
        if not env_details:
            return Response({"error": "Environment details not found"}, status=status.HTTP_404_NOT_FOUND)

        # Extract the OpenAI API key
        openai_api_key = env_details[3]
        if not openai_api_key:
            return Response({"error": "OpenAI API key not found for the specified environment"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Send the Excel data summary and user query to OpenAI for processing
        model_response = send_query_to_openai(openai_api_key, agent, data_summary, user_query)

        # Check response from OpenAI
        if model_response.get("success"):
            return Response({
                "content": model_response['content'],
                "total_tokens": model_response['total_tokens']
            }, status=status.HTTP_200_OK)
        else:
            return Response({"error": model_response['error']}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# Helper function to send Excel data and user query to OpenAI API
def send_query_to_openai(api_key, agent, data_summary, user_query):
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Prepare the message format required by the OpenAI API
        payload = {
            "model": "gpt-4o-mini",  # Use the model stored in the environment
            "messages": [
                {"role": "system", "content": agent[2]},  # system_prompt from agent details
                {"role": "user", "content": f"Here's the summary of the Excel data: {data_summary}. Now, based on this data, {user_query}"}
            ],
            "max_tokens": 1500  # Modify as per the requirement
        }

        # Make the POST request to OpenAI API
        response = requests.post(url, headers=headers, json=payload)

        # Check if the response is successful
        if response.status_code == 200:
            response_data = response.json()
            # Extract only the content and number of tokens used
            content = response_data['choices'][0]['message']['content']
            total_tokens = response_data['usage']['total_tokens']
            return {
                "success": True,
                "content": content,
                "total_tokens": total_tokens
            }
        else:
            return {"success": False, "error": response.text}

    except Exception as e:
        return {"success": False, "error": str(e)}


#Website_url
from rest_framework.parsers import JSONParser
from rest_framework.decorators import parser_classes, api_view
from rest_framework.response import Response
from rest_framework import status
import requests
from bs4 import BeautifulSoup  # For parsing HTML content from the website


@api_view(['POST'])
def run_openai_with_url(request):
    try:
        # Retrieve the environment details using agent_id
        agent_id = request.data.get('agent_id')
        if not agent_id:
            return Response({"error": "Agent ID is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve agent and environment details
        agent = db.read_agent(agent_id)
        if not agent:
            return Response({"error": "Agent not found"}, status=status.HTTP_404_NOT_FOUND)

        # Retrieve the website URL
        website_url = request.data.get('website_url')
        if not website_url:
            return Response({"error": "Website URL is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Fetch content from the website
        try:
            page_content = get_website_content(website_url)
            print(page_content)
        except Exception as e:
            return Response({"error": f"Error fetching website content: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # Get environment details
        env_id = agent[6]
        env_details = db.read_environment(env_id)
        if not env_details:
            return Response({"error": "Environment details not found"}, status=status.HTTP_404_NOT_FOUND)

        # Extract the OpenAI API key
        openai_api_key = env_details[3]
        if not openai_api_key:
            return Response({"error": "OpenAI API key not found for the specified environment"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Send the webpage content to OpenAI for summarization
        model_response = send_summary_request_to_openai(openai_api_key, agent, page_content)

        # Check response from OpenAI
        if model_response.get("success"):
            return Response({
                "content": model_response['content'],
                "total_tokens": model_response['total_tokens']
            }, status=status.HTTP_200_OK)
        else:
            return Response({"error": model_response['error']}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# Helper function to send webpage content to OpenAI for summarization
def send_summary_request_to_openai(api_key, agent, page_content):
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Prepare the message format required by the OpenAI API
        payload = {
            "model": "gpt-4o-mini",  # Use the model stored in the environment
            "messages": [
                {"role": "system", "content": agent[2]},  # system_prompt from agent details
                {"role": "user", "content": f"Summarize the following webpage content: {page_content}"}
            ],
            "max_tokens": 1500  # Modify as per the requirement
        }

        # Make the POST request to OpenAI API
        response = requests.post(url, headers=headers, json=payload)

        # Check if the response is successful
        if response.status_code == 200:
            response_data = response.json()
            # Extract only the content and number of tokens used
            content = response_data['choices'][0]['message']['content']
            total_tokens = response_data['usage']['total_tokens']
            return {
                "success": True,
                "content": content,
                "total_tokens": total_tokens
            }
        else:
            return {"success": False, "error": response.text}

    except Exception as e:
        return {"success": False, "error": str(e)}


# Helper function to fetch the website content using BeautifulSoup
def get_website_content(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve content from URL: {url}")

        # Parse the HTML content of the website
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text content from the HTML (you can modify this to extract specific content)
        page_text = soup.get_text(separator=' ', strip=True)

        # Return a portion or the entire content as needed
        return page_text[:5000]  # Return the first 5000 characters as a summary, you can adjust this as needed

    except Exception as e:
        raise Exception(f"Error retrieving website content: {str(e)}")
