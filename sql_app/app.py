def delete_images_in_current_directory() -> None:
    """
    Deletes all image files in the current working directory.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    current_directory = os.getcwd()

    for filename in os.listdir(current_directory):

        _, extension = os.path.splitext(filename)
        

        if extension.lower() in image_extensions:
            file_path = os.path.join(current_directory, filename)
            try:
                os.remove(file_path)
                # print(f"Deleted: {file_path}")
            except OSError as e:
                print(f"Error: {e} - {file_path}")

def get_images_in_current_directory() -> list:
    """
    Returns a list of all image files in the current working directory.

    Returns:
        list: A list of image file paths in the current directory.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    current_directory = os.getcwd()

    image_files = []

    for filename in os.listdir(current_directory):

        _, extension = os.path.splitext(filename)
        
        
        if extension.lower() in image_extensions:
            file_path = os.path.join(current_directory, filename)
            image_files.append(file_path)

    return image_files


import os
import streamlit as st
from vyzeai.models.openai import ChatOpenAI
from vyzeai.agents.react_agent import Agent
from vyzeai.tools.prebuilt_tools import execute_query
from vyzeai.tools.raw_functions import excel_to_sql
from plots import tools as plot_tools

st.set_page_config(page_title="Excel to SQL Chat App")
st.title("Excel to SQL Chat App")

# Initialize session state variables if they don't exist
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = None

if 'llm' not in st.session_state:
    st.session_state['llm'] = None

if 'agent' not in st.session_state:
    st.session_state['agent'] = None

if 'tables' not in st.session_state:
    st.session_state['tables'] = []

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Input fields for OpenAI API key and MySQL credentials
api_key = st.text_input("Enter OpenAI API key", type='password', key='api_key')
user = st.text_input("MySQL Username", 'uibcedotbqcywunfl752', key='user')
password = st.text_input("MySQL Password", 'LrdjP9dvLV0GP8PWRDmvREDB9IxmGu', type="password", key='password')
host = st.text_input("MySQL Host", 'by80v7itmu1gw3kjmblq-postgresql.services.clever-cloud.com:50013', key='host')
database = st.text_input("Database Name", 'by80v7itmu1gw3kjmblq', key='database')

uploaded_files = st.file_uploader("Upload Excel Files", type=["xlsx"], key='files', accept_multiple_files=True)

if st.session_state.api_key:

    # Initialize the LLM and Agent if they are not already initialized
    if st.session_state['llm'] is None:
        # Ensure ChatOpenAI is properly initialized with memory
        st.session_state['llm'] = ChatOpenAI(memory=True, api_key=st.session_state.api_key)

    if st.session_state['agent'] is None:
        query_tool = execute_query()
        tools = [query_tool] + plot_tools
        with open('system_prompt.py') as f:
            sys_p = f.read()
        st.session_state['agent'] = Agent(st.session_state['llm'], tools, react_prompt=sys_p)

    # Handling file uploads and storing data
    if st.button("Store Data", key='files_uploaded'):
        delete_images_in_current_directory()
        if uploaded_files:
            for uploaded_file in uploaded_files:
                excel_file_path = uploaded_file.name
                table_name = uploaded_file.name.split('.')[0]
                st.session_state.tables.append(table_name)

                with open(excel_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                response = excel_to_sql(excel_file_path, table_name, user, password, host, database)
                st.success(response)
        else:
            st.error("Please upload at least one Excel file.")

    # Handling user queries
    st.subheader("Chat with your Data")
    query = st.text_area("Enter your message", height=95)

    if st.button("Execute Query"):
        st.session_state.messages.append({"role": "user", "content": query})

        command = f"""
            user = '{user}'
            password = '{password}'
            host = '{host}'
            database = '{database}'

            tables related to user are {st.session_state.tables}
            User query: {query}
            """

        # Execute the query using the agent
        response = st.session_state['agent'](command)
        response = response.split('**Answer**:')[-1]
        st.session_state.messages.append({"role": "assistant", "content": response})

        ai_message = st.chat_message('ai')
        ai_message.write(response)

        plots = get_images_in_current_directory()
        for plot in plots:
            st.image(plot)

    # Sidebar Chat Log
    with st.sidebar:
        st.header("Chat Log")
        for message in st.session_state.messages:
            with st.expander(message['role']):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    # Display agent's memory (optional)
    if hasattr(st.session_state['llm'], 'chat_memory') and st.session_state['llm'].chat_memory is not None:
        st.write(st.session_state['llm'].chat_memory.get_memory())
    else:
        st.write("No memory available or memory has not been initialized properly.")
