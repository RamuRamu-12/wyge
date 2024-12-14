
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
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from wyge.models.openai import ChatOpenAI
from wyge.agents.react_agent import Agent
from wyge.tools.prebuilt_tools import execute_query
from wyge.tools.raw_functions import file_to_sql
from .plots import  tools as plot_tools
import pandas as pd


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


# Function-based view for file upload and conversion to SQL
@csrf_exempt
@require_http_methods(["POST"])
def upload_excel(request):
    """
    Handle the upload of Excel files and store them in MySQL.
    """
    user = request.POST.get('user')
    password = request.POST.get('password')
    host = request.POST.get('host')
    database = request.POST.get('database')
    uploaded_files = request.FILES.getlist('files')

    if not uploaded_files:
        return JsonResponse({"error": "No files uploaded."}, status=400)

    if not (user and password and host and database):
        return JsonResponse({"error": "Missing database credentials."}, status=400)

    # Delete existing image plots (optional)
    delete_images_in_directory(settings.MEDIA_ROOT)

    try:
        for uploaded_file in uploaded_files:
            # Validate the file extension
            if not uploaded_file.name.endswith(('.xlsx', '.xls')):
                return JsonResponse({"error": f"Invalid file format: {uploaded_file.name}"}, status=400)

            # Store the uploaded file
            file_name = uploaded_file.name
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            with default_storage.open(file_path, 'wb+') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            # Convert the Excel file to a SQL table
            table_name = file_name.split('.')[0]
            file_to_sql(file_path, table_name, user, password, host, database)

        return JsonResponse({"success": "Files uploaded and converted successfully."}, status=201)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# Function-based view to execute queries
@csrf_exempt
@require_http_methods(["POST"])
def execute_query_view(request):
    """
    Handle user queries and interact with the database via the OpenAI agent.
    """
    api_key = request.POST.get('api_key')
    user = request.POST.get('user')
    password = request.POST.get('password')
    host = request.POST.get('host')
    database = request.POST.get('database')
    query = request.POST.get('query')
    tables = request.POST.getlist('tables', [])

    if not (user and password and host and database and query):
        return JsonResponse({"error": "Missing required parameters."}, status=400)

    try:

        llm = ChatOpenAI(memory=True, api_key=api_key)
        query_tool = execute_query()
        tools = [query_tool] + plot_tools

        agent = Agent(llm, tools, react_prompt=reAct_prompt)

        # Construct the command to pass to the agent
        command = f"""
        user = '{user}'
        password = '{password}'
        host = '{host}'
        database = '{database}'
        tables related to user are {tables}
        User query: {query}
        """

        # Get agent response
        response = agent(command)
        response = response.split('**Answer**:')[-1]
        return JsonResponse({"result": response}, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# Function-based view to retrieve image plots
@csrf_exempt
@require_http_methods(["GET"])
def get_images(request):
    """
    Return a list of all image file paths in the media directory.
    """
    try:
        image_files = get_images_in_directory(settings.MEDIA_ROOT)
        return JsonResponse({"images": image_files}, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# Render the upload page with the file upload form
def upload_page(request):
    """
    Render the file upload page.
    """
    return render(request, 'api/upload.html')
