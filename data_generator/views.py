import io
import pandas as pd
from rest_framework.response import Response
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import FileResponse, JsonResponse
import tempfile
from .generator import generate_synthetic_data, generate_data_from_text, fill_missing_data_in_chunk


# Excel to Synthetic Data Generation
@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def excel_to_synthetic(request):
    api_key = request.data.get('api_key')
    num_rows = int(request.data.get('num_rows', 10))
    file = request.FILES.get('file')

    if not api_key or not file:
        return Response({"error": "API key and Excel file are required."}, status=400)

    try:
        # Read the uploaded Excel file into a DataFrame
        original_df = pd.read_excel(file)

        # Create a temporary file for the Excel data
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
            # Save the uploaded file to a temporary location
            temp_file_name = temp_file.name
            original_df.to_excel(temp_file_name, index=False)

        # Generate synthetic data using the file path (as the function expects a file path)
        generated_df = generate_synthetic_data(api_key, temp_file_name, num_rows=num_rows)

        # Combine the original and synthetic data
        combined_df = pd.concat([original_df, generated_df], ignore_index=True)

        print(combined_df)

        # Convert to CSV for download
        combined_csv = combined_df.to_csv(index=False)

        return JsonResponse({
            "message": "Synthetic data generated successfully.",
            "data": combined_csv
        })

        # # Return the generated Excel file as a download
        # return FileResponse(combined_csv, as_attachment=True, filename='generated_data.xlsx', content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        return Response({"error": str(e)}, status=500)


# Text to Synthetic Data Generation
@api_view(['POST'])
def text_to_synthetic(request):
    api_key = request.data.get('api_key')
    text_input = request.data.get('text_input')
    column_names = request.data.getlist('columns')
    num_rows = int(request.data.get('num_rows', 10))

    if not api_key or not text_input or not column_names:
        return Response({"error": "API key, text input, and column names are required."}, status=400)

    try:
        # Generate synthetic data from the text input and column names
        generated_df = generate_data_from_text(api_key, text_input, column_names, num_rows=num_rows)

        # Convert to CSV for download
        combined_csv = generated_df.to_csv(index=False)

        return JsonResponse({
            "message": "Synthetic data generated successfully.",
            "data": combined_csv
        })

        # # Return the generated Excel file as a download
        # return FileResponse(output, as_attachment=True, filename='generated_data.xlsx', content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        return Response({"error": str(e)}, status=500)


# Fill Missing Data in Excel
@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def fill_missing_data(request):
    api_key = request.data.get('api_key')
    file = request.FILES.get('file')

    if not api_key or not file:
        return Response({"error": "API key and Excel file are required."}, status=400)

    try:
        # Read the uploaded Excel file into a DataFrame
        original_df = pd.read_excel(file)

        # Create a temporary file for the Excel data
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
            # Save the uploaded file to a temporary location
            temp_file_name = temp_file.name
            original_df.to_excel(temp_file_name, index=False)

        # Fill missing data using the file path (as the function expects a file path)
        filled_df = fill_missing_data_in_chunk(api_key, temp_file_name)

        # Convert to CSV for download
        combined_csv = filled_df.to_csv(index=False)

        return JsonResponse({
            "message": "Synthetic data generated successfully.",
            "data": combined_csv
        })

        # # Return the filled Excel file as a download
        # return FileResponse(output, as_attachment=True, filename='filled_data.xlsx', content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        return Response({"error": str(e)}, status=500)
