# # util.py
#
# import pandas as pd
# from vyzeai.models.openai import ChatOpenAI
#
#
# def generate_synthetic_data(api_key, file_path, num_rows=10, chunk_size=50):
#     """Generate synthetic data."""
#
#     llm = ChatOpenAI(api_key)
#
#     data = pd.read_excel(file_path).tail(30)
#     sample_str = data.to_csv(index=False, header=False)
#
#     sysp = "You are a synthetic data generator. Your output should only be CSV format without any additional text and code fences."
#
#     generated_rows = []
#     rows_generated = 0
#
#     while rows_generated < num_rows:
#
#         if generated_rows:
#             current_sample_str = "\n".join([",".join(row) for row in generated_rows[-10:]])
#         else:
#             current_sample_str = sample_str
#
#         rows_to_generate = min(chunk_size, num_rows - rows_generated)
#
#         prompt = (
#             f"Generate {rows_to_generate} more rows of synthetic data following this pattern:\n\n{current_sample_str}\n"
#             "\nEnsure the synthetic data does not contain column names or old data. "
#             "\nExpected Output: synthetic data as comma-separated values (',').")
#         #                  "\nFor dates and time, maintain sequence. "
#
#         generated_data = llm.run(prompt, system_message=sysp)
#
#         rows = [row.split(",") for row in generated_data.strip().split("\n") if row]
#
#         rows_needed = num_rows - rows_generated
#         generated_rows.extend(rows[:rows_needed])
#
#         rows_generated += len(rows[:rows_needed])
#
#     generated_df = pd.DataFrame(generated_rows, columns=data.columns)
#
#     return generated_df


import pandas as pd
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
