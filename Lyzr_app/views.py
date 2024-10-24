#Environment Creation
import os

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
        #model_vendor = data.get('model_vendor')
        api_key = data.get('api_key')
        model = data.get('model')
        temperature = data.get('temperature', 0.5)
        #top_p = data.get('top_p', 0.9)

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
                    #"provider": environment[2],  # model_vendor
                    "model": environment[3],  # model
                    "config": {
                        "temperature": environment[4],  # temperature
                        #"top_p": environment[6],  # top_p
                    }
                },
                "env": {
                    "Environment_name": environment[1],  # name
                    "OPENAI_API_KEY":environment[2],  #API_KEY
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
        #model_vendor = data.get('model_vendor')
        api_key = data.get('api_key')
        model = data.get('model')
        temperature = data.get('temperature')
        #top_p = data.get('top_p')

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
                    #"model_vendor": environment[2],
                    "api_key": environment[2],
                    "model": environment[3],
                    "temperature": environment[4],
                    #"top_p": environment[6],

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
from vyzeai.agents.prebuilt_agents import ResearchAgent, VideoAudioBlogAgent, YTBlogAgent, BlogAgent,LinkedInAgent, VideoAgent, EmailAgent

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
            return Response({"error": "API key not found in environment table"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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



reAct_prompt = """
You are an SQL Agent and can  execute SQL queries using the provided action. if user asks to forecast, you also do forecasting using moving average with plot and also provide stats.

You must strictly follow the cycle of *Thought -> Action -> PAUSE -> Observation -> Thought -> Action -> PAUSE -> Observation -> Thought -> -> -> -> Answer. Each message in conversation should contain only one role at a time, followed by **PAUSE*.

### Rules:
1. *Thought*: Reflect on how to solve the problem. Describe the SQL query that will be executed without running it yet.
2. *Action*: Execute the SQL query. 
3. *Observation*: After receiving the result from the SQL query, report the outcome and whether further adjustments are needed. Do not provide the final answer yet. 
4. *Answer*: Provide the final answer once the task is fully complete. 

### Important Guidelines:
- Do not combine multiple steps (e.g., Thought + Action or Observation + Answer) in a single message. 
- Each role must be distinctly addressed to uphold clarity and prevent confusion. 
- If steps are combined or skipped, it may lead to miscommunication and errors in the final message.

### Example Session:

## Example Actions:
- *execute_query*: e.g., execute_query('SELECT * FROM table_name). Runs a SQL query. 

## Agent Flow (agent responds step by step):
*user*: Retrieve all users from the database where age is greater than 30.

*assistant*: Thought: I need to execute a SQL query to retrieve all users where the age is greater than 30 from the 'users' table. PAUSE

*assistant*: Action: SELECT * FROM users WHERE age > 30; PAUSE

*assistant*: Observation: The query executed successfully and returned 12 rows of data. PAUSE

*assistant*: Thought: Now I can provide final answer. PAUSE

*assistant*: Answer: Here are the users where age is greater than 30.

---

Now it's your turn:

- Execute one step at a time (Thought or Action or Observation or Answer).
- Entire flow is not required for simple user queries.
- User can *see only the Final Answer*. So provide clear Answer at the end.

Additional Handling for Special Requests:
- Forecasting: If the user asks for a forecast (e.g., sales forecast or trend prediction), include statistical insights in the final answer and generate a plot to visualize the forecast.
    - Save the plot in present directory and include the file path in the final Answer.
    - Report statistics (e.g., averages, trends) along with the final answer.

** Final Answer should be verbose**
"""

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
from .plots import  tools as plot_tools
from .generator import generate_synthetic_data, generate_data_from_text, fill_missing_data_in_chunk

@api_view(['POST'])
def run_openai_environment(request):
    try:
        agent_id = request.data.get('agent_id')
        user_prompt = request.data.get('prompt', '')
        url = request.data.get('url', '')
        yt_url = request.data.get('yt_url', '')
        file = request.FILES.get('file')  # File if attached
        num_rows = request.data.get('num_rows', 10)  # Number of synthetic rows to generate
        column_names = request.data.getlist('columns')
        print(column_names)

        # Ensure num_rows is an integer
        try:
            num_rows = int(num_rows)
        except ValueError:
            return Response({"error": "Invalid value for num_rows. It must be an integer."}, status=status.HTTP_400_BAD_REQUEST)

        if not agent_id:
            return Response({"error": "Agent ID is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve agent details
        agent = db.read_agent(agent_id)
        if not agent:
            return Response({"error": "Agent not found"}, status=status.HTTP_404_NOT_FOUND)

        # Retrieve the API key from the environment table using env_id
        env_details = db.read_environment(agent[6])  # agent[6] is env_id
        if not env_details or not env_details[2]:
            return Response({"error": "API key not found in environment table"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        openai_api_key = env_details[2]

        # Process based on input type (prompt, URL, file)
        result = None

        # Delete existing image plots (optional)
        delete_images_in_directory(settings.MEDIA_ROOT)

        # If file is uploaded and tool type is text-to-SQL
        if file and 'text-to-sql' in agent[4]:
            # Store the Excel file in the database
            table_name = file.name.split('.')[0]
            try:
                file_path = os.path.join(settings.MEDIA_ROOT, file.name)
                with default_storage.open(file_path, 'wb+') as f:
                    for chunk in file.chunks():
                        f.write(chunk)

                # Convert the Excel file to a SQL table
                excel_to_sql(file_path, table_name, "uibcedotbqcywunfl752", "LrdjP9dvLV0GP8PWRDmvREDB9IxmGu", "by80v7itmu1gw3kjmblq-postgresql.services.clever-cloud.com:50013", "by80v7itmu1gw3kjmblq")

                # After storing, generate a graph or text based on the query
                llm = ChatOpenAI(memory=True, api_key=openai_api_key)
                query_tool = execute_query()
                tools = [query_tool] + plot_tools

                agent = Agent(llm, tools, react_prompt=reAct_prompt)

            # Construct the command to pass to the agent
                command = f"""
                    user = 'uibcedotbqcywunfl752'
                    password = 'LrdjP9dvLV0GP8PWRDmvREDB9IxmGu'
                    host = 'by80v7itmu1gw3kjmblq-postgresql.services.clever-cloud.com:50013'
                    database = 'by80v7itmu1gw3kjmblq'
                    tables related to user are {table_name}
                    User query: {user_prompt}
                    """

            # Get agent response
                result = agent(command)
                result = result.split('**Answer**:')[-1]

                # Check if the prompt is related to plotting a graph
                plot_keywords = ['plot', 'graph', 'chart', 'visualize']
                if any(keyword in user_prompt.lower() for keyword in plot_keywords):
                    # If the prompt asks for a graph, find the latest graph in the images directory
                    images = get_images_in_directory(settings.MEDIA_ROOT)

                    if images:
                        latest_image = images[-1]  # Get the latest image
                        image_path = os.path.join(settings.MEDIA_ROOT, latest_image)
                        image_base64 = image_to_base64(image_path)

                        # Return the image in base64 format
                        return JsonResponse({
                            "message": "Graph generated successfully.",
                            "image_base64": image_base64
                        }, status=200)
                    else:
                        return JsonResponse({"error": "No graph generated."}, status=500)

                return JsonResponse({"result": result}, status=200)

            except Exception as e:
                return JsonResponse({"error": str(e)}, status=500)


        # If only a prompt is provided
        elif user_prompt and not (url or yt_url or file or column_names):
            model_response = send_prompt_to_openai(openai_api_key, agent, user_prompt)
            if model_response.get("success"):
                return Response({
                    "message": "Prompt processed successfully.",
                    "content": model_response['content'],
                    "total_tokens": model_response['total_tokens']
                }, status=status.HTTP_200_OK)
            else:
                return Response({"error": model_response['error']}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


        # If URL + prompt is provided
        elif (url or yt_url) and user_prompt:
            if 'blog_post' in agent[4]:
                result = generate_blog_from_url(user_prompt, url or yt_url, 'blog_post', openai_api_key)
            elif 'linkedin_post' in agent[4]:
                result = generate_blog_from_url(user_prompt, url or yt_url, 'linkedin_post', openai_api_key)


        # If file + prompt is provided for blog or LinkedIn post
        elif file and user_prompt:
            if 'blog_post' in agent[4]:
                result = generate_blog_from_file(user_prompt, file, 'blog_post', openai_api_key)
            elif 'linkedin_post' in agent[4]:
                result = generate_blog_from_file(user_prompt, file, 'linkedin_post', openai_api_key)


        # If file is provided for synthetic data generation for the new data
        # For synthetic data generation for new data
        elif 'Synthetic_data_new_data' in agent[4]:
            try:
                # Generate synthetic data from the text input and column names
                generated_df = generate_data_from_text(openai_api_key, user_prompt, column_names, num_rows=num_rows)
                print(generated_df)
                # Convert to CSV for download
                combined_csv = generated_df.to_csv(index=False)
                print(combined_csv)

                return JsonResponse({
                    "message": "Synthetic data generated successfully.",
                    "data": combined_csv
                })

            except Exception as e:
                return Response({"error": str(e)}, status=500)

        # Synthetic data generation for the extended data
        elif file and 'Synthetic_data_extended_data' in agent[4]:
            try:
                # Read the uploaded Excel file into a DataFrame
                original_df = pd.read_excel(file)

                # Check if the DataFrame has more than 300 rows
                if len(original_df) > 300:
                    original_df = original_df.head(300)  # Take the first 300 rows

                # Create a temporary file for the Excel data
                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
                    # Save the truncated or original file to a temporary location
                    temp_file_name = temp_file.name
                    original_df.to_excel(temp_file_name, index=False)

                # Generate synthetic data using the file path
                generated_df = generate_synthetic_data(openai_api_key, temp_file_name, num_rows=num_rows)

                # Combine the original and synthetic data
                combined_df = pd.concat([original_df, generated_df], ignore_index=True)

                # Convert to CSV for download
                combined_csv = combined_df.to_csv(index=False)

                return JsonResponse({
                    "message": "Synthetic data generated successfully.",
                    "data": combined_csv
                })

            except Exception as e:
                return Response({"error": str(e)}, status=500)

        # Synthetic data generation for missing data
        elif file and 'Synthetic_data_missing_data' in agent[4]:
            try:
                # Read the uploaded Excel file into a DataFrame
                original_df = pd.read_excel(file)

                # Check if the DataFrame has more than 300 rows
                if len(original_df) > 300:
                    original_df = original_df.head(300)  # Take the first 300 rows

                # Create a temporary file for the Excel data
                with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
                    # Save the truncated or original file to a temporary location
                    temp_file_name = temp_file.name
                    original_df.to_excel(temp_file_name, index=False)

                # Fill missing data using the file path
                filled_df = fill_missing_data_in_chunk(openai_api_key, temp_file_name)
                print(filled_df)

                # Convert to CSV for download
                combined_csv = filled_df.to_csv(index=False)

                return JsonResponse({
                    "message": "Synthetic data generated successfully.",
                    "data": combined_csv
                })

            except Exception as e:
                return Response({"error": str(e)}, status=500)


        if result:
            image_base64 = image_to_base64(result.get('image_path'))
            return Response({
                "message": "Content generated successfully.",
                "content": result['content'],
                "image_base64": image_base64 or " "
            }, status=status.HTTP_200_OK)

        else:
            return Response({"error": "No valid tool found for the given input."}, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# Helper function to process the prompt through OpenAI
def send_prompt_to_openai(api_key, agent, user_prompt):
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",  # Use appropriate model
            "messages": [
                {"role": "system", "content": agent[2]},  # System prompt from agent details
                {"role": "user", "content": user_prompt}  # User's input prompt
            ],
            "max_tokens": 1500
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            response_data = response.json()
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


# Generate content from URL (for blog or LinkedIn post)
def generate_blog_from_url(prompt, url, option, api_key):
    try:
        if option == 'blog_post':
            research_agent = ResearchAgent(api_key)
            blog_agent = BlogAgent(api_key)
            context = research_agent.research(prompt, url)
            blog,doc_file,image = blog_agent.generate_blog(prompt, url, context)
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


# Generate content from file (for blog or LinkedIn post)
def generate_blog_from_file(prompt, file, option, api_key):
    try:
        file_path = save_file(file)

        if option == 'blog_post':
            va_agent = VideoAudioBlogAgent(api_key)
            blog,doc_file,image = va_agent.generate_blog(file_path)
            # content = contents[0][0]
            # image_path = contents[-1][-1][0]
            return {"content": blog, "image_path":image}
        elif option == 'linkedin_post':
            va_agent = VideoAudioBlogAgent(api_key)
            linkedin_agent = LinkedInAgent(api_key)
            context = va_agent.extract_text(file_path)
            content, image_path = linkedin_agent.generate_linkedin_post(context)
            return {"content": content, "image_path": image_path}
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

