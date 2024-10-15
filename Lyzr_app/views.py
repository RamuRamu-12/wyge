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

        # Set default values for the boolean fields
        upload_excel = data.get('upload_excel', False)
        read_website = data.get('read_website', False)

        # Ensure upload_excel and read_website are treated as booleans
        upload_excel = True if upload_excel == True else False
        read_website = True if read_website == True else False

        environment_id = db.create_environment(name, model_vendor, api_key, model, temperature, top_p, upload_excel, read_website)
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
                "Additional_Features":{
                    "upload_excel": environment[7],  # Boolean field
                    "read_website": environment[8]   # Boolean field
                }
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

        # Set the boolean fields based on the request, default is None (no change)
        upload_excel = data.get('upload_excel')  # Optional, will update if True or False is provided
        read_website = data.get('read_website')  # Optional, will update if True or False is provided

        # No need for explicit condition checks for True/False now, we pass the value directly if provided.
        updated_rows = db.update_environment(
            environment_id,
            name,
            model_vendor,
            api_key,
            model,
            temperature,
            top_p,
            upload_excel,  # Now updates to both True and False values are allowed
            read_website   # Now updates to both True and False values are allowed
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
                    "upload_excel": environment[7],  # Boolean field
                    "read_website": environment[8],  # Boolean field
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
        tools = data.get('tools')  # Handle tools from the request
        env_id = data.get('env_id')

        if not env_id:
            return Response({"error": "Environment ID is required"}, status=400)

        # Create the agent in the database
        agent_id = db.create_agent(name, system_prompt, agent_description, tools, env_id)

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
                "tools": agent[4],  # Include tools in the response
                "env_id": agent[5]
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
        tools = data.get('tools')  # Handle tools update
        env_id = data.get('env_id')

        # Update agent in the database
        db.update_agent(agent_id, name, system_prompt, agent_description, tools, env_id)

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
@api_view(['GET'])
def read_all_agents(request):
    try:
        # Fetch all agents from the database
        agents = db.get_all_agents()

        # Structure the agents' data for JSON response
        agents_data = [
            {
                "id": agent[0],
                "name": agent[1],
                "system_prompt": agent[2],
                "agent_description": agent[3],
                "tools": agent[4],  # Include tools in the response
                "env_id": agent[5]
            }
            for agent in agents
        ]

        return Response({"agents": agents_data}, status=200)
    except Exception as e:
        return Response({"error": str(e)}, status=400)


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
            "env_id": agent[4]
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
            "max_tokens": 150
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
        env_id = agent[4]
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
            "max_tokens": 150  # Modify as per the requirement
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
