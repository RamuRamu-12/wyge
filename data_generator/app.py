import streamlit as st
import pandas as pd
from generator import generate_synthetic_data, generate_data_from_text, fill_missing_data_in_chunk
import io

st.title("Synthetic Data Generator")

api_key = st.text_input("Enter your OpenAI API key:", type="password")

# Creating two tabs: "Excel to Synthetic" and "Text to Synthetic"
tab1, tab2, tab3 = st.tabs(["Excel to Synthetic", "Text to Synthetic", "Fill missing data"])

# Tab 1: Excel to Synthetic Data Generation
with tab1:
    st.header("Generate Synthetic Data from Excel File")

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    num_rows = st.number_input("Number of rows to generate from Excel", min_value=1, step=1, value=10)

    if st.button("Generate from Excel"):
        if uploaded_file is not None and api_key:
            with open("temp.xlsx", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Generate synthetic data using the Excel file
            generated_df = generate_synthetic_data(api_key, "temp.xlsx", num_rows=num_rows)

            # Read original data from Excel
            original_df = pd.read_excel("temp.xlsx")

            # Display generated data
            st.write("Generated Synthetic Data from Excel:")
            st.dataframe(generated_df)

            # Combine original and synthetic data
            combined_df = pd.concat([original_df, generated_df], ignore_index=True)

            # Download the combined data as an Excel file
            st.write("Download:")
            output_text = io.BytesIO()
            with pd.ExcelWriter(output_text, engine='xlsxwriter') as writer:
                combined_df.to_excel(writer, index=False)
            st.download_button(
                label="Download",
                data=output_text,
                file_name="generated_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.error("Please upload an Excel file and enter your API key.")

# Tab 2: Text to Synthetic Data Generation
with tab2:
    st.header("Generate Synthetic Data from Text")

    text_input = st.text_area("Enter a description to generate synthetic data from (e.g., 'create data of customers on Amazon')")

    # Dynamic input for column names
    st.subheader("Specify Column Names")

    # Create a placeholder for the dynamic column names
    column_container = st.container()

    # Add buttons to add/remove column names dynamically
    columns_list = []

    if 'columns_list' not in st.session_state:
        st.session_state['columns_list'] = []

    # Function to add a column name input
    def add_column():
        st.session_state['columns_list'].append('')

    # Function to remove a column name input
    def remove_column():
        if st.session_state['columns_list']:
            st.session_state['columns_list'].pop()

    # Display current column names and allow modification
    for i, column in enumerate(st.session_state['columns_list']):
        column_name = column_container.text_input(f"Column {i + 1} Name", key=f"column_{i}")
        st.session_state['columns_list'][i] = column_name

    # Buttons to add/remove columns
    st.button("Add Column", on_click=add_column)
    st.button("Remove Last Column", on_click=remove_column)

    # Number of rows to generate
    num_rows_text = st.number_input("Number of rows to generate", min_value=1, step=1, value=10)

    # Generate synthetic data button
    if st.button("Generate from Text"):
        if text_input and api_key and st.session_state['columns_list']:
            # Pass the dynamically created columns to the function
            column_names = st.session_state['columns_list']

            # Generate synthetic data using the provided columns and text
            generated_text_df = generate_data_from_text(api_key, text_input, column_names, num_rows=num_rows_text)

            # Display generated data
            st.write("Generated Synthetic Data from Text:")
            st.dataframe(generated_text_df)

            # Download generated data as an Excel file
            output_text = io.BytesIO()
            with pd.ExcelWriter(output_text, engine='xlsxwriter') as writer:
                generated_text_df.to_excel(writer, index=False)
            output_text.seek(0)
            st.download_button(
                label="Download",
                data=output_text,
                file_name="generated_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.error("Please enter a description, column names, and your API key.")

with tab3:
    st.header("Generate Missing Data from Excel File")

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"], key='excel_file')

    if st.button("Fill Missing Data"):
        if uploaded_file is not None and api_key:
            with open("temp.xlsx", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Generate synthetic data using the Excel file
            generated_df = fill_missing_data_in_chunk(api_key, "temp.xlsx")

            # Read original data from Excel
            original_df = pd.read_excel("temp.xlsx")

            # Display generated data
            st.write("Generated Synthetic Data from Excel:")
            st.dataframe(generated_df)

            # Combine original and synthetic data
            # combined_df = pd.concat([original_df, generated_df], ignore_index=True)

            # Download the combined data as an Excel file
            st.write("Download Combined Data:")
            output_text = io.BytesIO()
            with pd.ExcelWriter(output_text, engine='xlsxwriter') as writer:
                generated_df.to_excel(writer, index=False)
            st.download_button(
                label="Download",
                data=output_text,
                file_name="generated_excel.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.error("Please upload an Excel file and enter your API key.")





# import streamlit as st
# import pandas as pd
# from generator import generate_synthetic_data

# st.title("Synthetic Data Generator")

# api_key = st.text_input("Enter your OpenAI API key:", type="password")

# uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

# num_rows = st.number_input("Number of rows to generate", min_value=1, step=1, value=10)

# if st.button("Generate Data"):
#     if uploaded_file is not None and api_key:
    
#         with open("temp.xlsx", "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
    
#         generated_df = generate_synthetic_data(api_key, "temp.xlsx", num_rows=num_rows)

    
#         original_df = pd.read_excel("temp.xlsx")

    
#         st.write("Generated Synthetic Data:")
#         st.dataframe(generated_df)

    
#         combined_df = pd.concat([original_df, generated_df], ignore_index=True)

    
#         # combined_csv = combined_df.to_csv(index=False).encode('utf-8')
#         combined_excel = combined_df.to_excel(index=False)
#         st.download_button(
#             label="Download Combined Data as CSV",
#             data=combined_excel,
#             file_name="combined_data.csv",
#             mime="text/csv",
#         )
#     else:
#         st.error("Please upload an Excel file and enter your API key.")
