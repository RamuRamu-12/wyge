#Environment Creation
import os

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
from vyzeai.agents.prebuilt_agents import ResearchAgent, VideoAudioBlogAgent, YTBlogAgent, BlogAgent,LinkedInAgent, VideoAgent, EmailAgent

from rest_framework.response import Response
from rest_framework import status
import requests
import pandas as pd


# Main API that handles both tool execution and OpenAI environment creation
@api_view(['POST'])
def create_openai_environment_api(request):
    try:
        agent_id = request.data.get('agent_id')
        option = request.data.get('option')

        if not agent_id:
            return Response({"error": "Agent ID is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve the agent details from the database
        agent = db.read_agent(agent_id)
        if not agent:
            return Response({"error": "Agent not found"}, status=status.HTTP_404_NOT_FOUND)

        # Ensure that the selected tool is present in the agent's tools
        tools = agent[4]  # Assuming the tools column is at index 4
        if option not in tools:
            return Response({"error": f"{option} is not available for this agent."}, status=status.HTTP_400_BAD_REQUEST)

        # Proceed with environment creation logic
        agent_details = {
            "name": agent[1],
            "system_prompt": agent[2],
            "agent_description": agent[3],
            "tools": agent[4],
            "env_id": agent[6]
        }

        # Call the function to create the OpenAI environment
        environment_response = create_openai_environment(agent_details)

        if environment_response.get("success"):
            # If environment creation is successful, handle tool execution
            result = handle_agent_tool(request, agent, option)
            return Response({"message": "OpenAI environment created successfully.",
                             "details": environment_response, "result": result},
                            status=status.HTTP_201_CREATED)
        else:
            return Response({"error": "Failed to create OpenAI environment."},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


# Function to handle different tools (Blog post, LinkedIn post, Synthetic data, etc.)
def handle_agent_tool(request, agent, option):
    try:
        # Fetch the environment details based on the agent's env_id
        env_id = agent[6]
        if not env_id:
            raise ValueError("Environment ID not found in agent details")

        # Retrieve the API key from the environment table
        env_details = db.read_environment(env_id)
        if not env_details:
            raise ValueError("Environment details not found")

        # Extract API key (assuming it's at index 3 in the env_details tuple)
        api_key = env_details[3]
        if not api_key:
            raise ValueError("OpenAI API key not found in the environment")

        # Blog post option
        if option == 'blog_post':
            options = request.data.get('post_type')
            topic = request.data.get('topic', '')
            url = request.data.get('url', '')
            yt_url = request.data.get('yt_url', '')
            file_path = request.data.get('file_path', '')

            content, image_path = None, None
            if options == 'website url to blog':
                research_agent = ResearchAgent(api_key)
                blog_agent = BlogAgent(api_key)
                context = research_agent.research(topic, url)
                contents = blog_agent.generate_blog(topic, url, context)
                content = contents[0][0]
                print("**********************************************************************************************************")
                print(contents)
                image_path = contents[-1][-1][0]
            elif options == 'youtube url to blog':
                yt_agent = YTBlogAgent(api_key)
                contents = yt_agent.generate_blog(yt_url)

                content = contents[0][0]
                image_path = contents[-1][-1][0]
            elif options == 'video/audio to blog':
                va_agent = VideoAudioBlogAgent(api_key)
                contents = va_agent.generate_blog(file_path)
                content = contents[0][0]
                image_path = contents[-1][-1][0]
            return {"content": content, "image_path": image_path}

        # LinkedIn post option
        elif option == 'linkedin_post':
            post_type = request.data.get('post_type')
            linkedin_agent = LinkedInAgent(api_key)

            content, image_path = None, None
            if post_type == 'website':
                topic = request.data.get('topic')
                url = request.data.get('url')
                research_agent = ResearchAgent(api_key)
                context = research_agent.research(topic, url)
                content, image_path = linkedin_agent.generate_linkedin_post(context)
            elif post_type == 'youtube':
                yt_url = request.data.get('yt_url')
                yt_agent = YTBlogAgent(api_key)
                context = yt_agent.extract_transcript(yt_url)
                content, image_path = linkedin_agent.generate_linkedin_post(context)
            elif post_type == 'video/audio':
                file_path = request.data.get('file_path')
                va_agent = VideoAudioBlogAgent(api_key)
                context = va_agent.extract_text(file_path)
                content, image_path = linkedin_agent.generate_linkedin_post(context)
            return {"content": content, "image_path": image_path}

        # Synthetic data option
        elif option == 'synthetic_data':
            if 'uploaded_file' in request.FILES:
                uploaded_file = request.FILES['uploaded_file']
                num_rows = int(request.POST.get('num_rows', 10))  # Default to 10 if not provided
                file_path = save_file(uploaded_file)
                generated_df = generate_synthetic_data(api_key, file_path, num_rows=num_rows)
                original_df = pd.read_excel(file_path)
                combined_df = pd.concat([original_df, generated_df], ignore_index=True)
                combined_csv = combined_df.to_csv(index=False)
                return {"message": "Synthetic data generated successfully.", "data": combined_csv}
            return {"error": "File not provided."}

        # Text-to-SQL option (implement your logic here)
        elif option == 'text_to_sql':
            # Placeholder for text-to-SQL logic
            return {"message": "Text to SQL feature is being implemented."}

        else:
            return {"error": "Invalid option selected."}

    except Exception as e:
        return {"error": str(e)}


# Helper function to save uploaded file
def save_file(file):
    # Define the directory path
    directory = 'temp'

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the file to the 'temp' directory
    file_path = os.path.join(directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.read())

    return file_path


# Helper function to create OpenAI environment
def create_openai_environment(agent_details):
    try:
        env_id = agent_details["env_id"]
        env_details = db.read_environment(env_id)  # Returns a tuple

        if not env_details:
            raise ValueError("Environment details not found.")

        openai_api_key = env_details[3]
        if not openai_api_key:
            raise ValueError("OpenAI API key not found for the specified environment")

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
            print(f"OpenAI API Error: {response.status_code} - {response.text}")
            return {"success": False, "error": response.text}

    except Exception as e:
        print(f"Error creating OpenAI environment: {e}")
        return {"success": False, "error": str(e)}



from vyzeai.models.openai import ChatOpenAI
def generate_synthetic_data(api_key, file_path, num_rows=10, chunk_size=50):
    """Generate synthetic data."""

    llm = ChatOpenAI(api_key)

    # Load the original data and get the column structure
    data = pd.read_excel(file_path).tail(30)
    sample_str = data.to_csv(index=False, header=False)
    expected_columns = data.columns  # The expected number of columns

    sysp = "You are a synthetic data generator. Your output should only be CSV format without any additional text and code fences."

    generated_rows = []
    rows_generated = 0

    while rows_generated < num_rows:

        if generated_rows:
            # If we have already generated rows, use the last 10 rows for context
            current_sample_str = "\n".join([",".join(row) for row in generated_rows[-10:]])
        else:
            # Use the original sample data if no rows have been generated yet
            current_sample_str = sample_str

        # Calculate how many more rows are needed
        rows_to_generate = min(chunk_size, num_rows - rows_generated)

        prompt = (
            f"Generate {rows_to_generate} more rows of synthetic data following this pattern:\n\n{current_sample_str}\n"
            "\nEnsure the synthetic data does not contain column names or old data. "
            "\nExpected Output: synthetic data as comma-separated values (',').")

        # Generate the synthetic data using the language model
        generated_data = llm.run(prompt, system_message=sysp)

        # Parse the generated data into rows
        rows = [row.split(",") for row in generated_data.strip().split("\n") if row]

        # Ensure that each generated row matches the expected column count
        cleaned_rows = []
        for row in rows:
            if len(row) == len(expected_columns):
                cleaned_rows.append(row)  # Accept rows with the correct number of columns
            elif len(row) < len(expected_columns):
                # If the row has fewer columns, append empty strings to match the column count
                cleaned_rows.append(row + [''] * (len(expected_columns) - len(row)))
            elif len(row) > len(expected_columns):
                # If the row has more columns, truncate the extra columns
                cleaned_rows.append(row[:len(expected_columns)])

        # Add the cleaned rows to the generated rows
        rows_needed = num_rows - rows_generated
        generated_rows.extend(cleaned_rows[:rows_needed])

        rows_generated += len(cleaned_rows[:rows_needed])

    # Create the DataFrame using the original column names
    generated_df = pd.DataFrame(generated_rows, columns=expected_columns)

    return generated_df









#Run Openai Environment
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import requests
import os
import pandas as pd

@api_view(['POST'])
def run_openai_environment(request):
    try:
        # Retrieve the agent_id and user prompt from request
        agent_id = request.data.get('agent_id')
        user_prompt = request.data.get('prompt')
        option = request.data.get('option')

        if not agent_id:
            return Response({"error": "Agent ID is required"}, status=status.HTTP_400_BAD_REQUEST)
        if not user_prompt:
            return Response({"error": "User prompt is required"}, status=status.HTTP_400_BAD_REQUEST)
        if not option:
            return Response({"error": "Option is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve agent details
        agent = db.read_agent(agent_id)
        if not agent:
            return Response({"error": "Agent not found"}, status=status.HTTP_404_NOT_FOUND)

        # Check if the option (tool) exists for the agent
        tools = agent[4]  # Assuming the tools column is at index 4
        if option not in tools:
            return Response({"error": f"{option} is not available for this agent."}, status=status.HTTP_400_BAD_REQUEST)

        # Create the OpenAI environment
        agent_details = {
            "name": agent[1],
            "system_prompt": agent[2],
            "agent_description": agent[3],
            "tools": agent[4],
            "env_id": agent[6]
        }
        environment_response = create_openai_environment(agent_details)

        if not environment_response.get("success"):
            return Response({"error": "Failed to create OpenAI environment."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Call the function to handle the selected tool
        result = handle_agent_tool(request, agent, option)

        # Send the user's prompt to OpenAI
        env_id = agent_details["env_id"]
        env_details = db.read_environment(env_id)  # Returns a tuple

        if not env_details:
            raise ValueError("Environment details not found.")

        openai_api_key = env_details[3]
        if not openai_api_key:
            raise ValueError("OpenAI API key not found for the specified environment")

        model_response = send_prompt_to_openai(openai_api_key, agent, user_prompt)

        if model_response.get("success"):
            return Response({
                "message": "OpenAI environment run successful.",
                #"result": result,
                "content": model_response['content'],
                "total_tokens": model_response['total_tokens']
            }, status=status.HTTP_200_OK)
        else:
            return Response({"error": model_response['error']}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)




# Function to send the prompt to OpenAI (same as defined earlier)
def send_prompt_to_openai(api_key, agent, user_prompt):
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": agent[2]},
                {"role": "user", "content": user_prompt}
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
        print(data_summary)

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

