import os
import pandas as pd
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.utils import json

from .generator import generate_synthetic_data
from django.core.mail import EmailMessage


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


@csrf_exempt
def generate_data(request):
    """
    API for generating synthetic data.
    """
    if request.method == 'POST':
        # Check if 'uploaded_file' is in request.FILES
        if 'uploaded_file' in request.FILES:
            uploaded_file = request.FILES['uploaded_file']

            # Get other form-data parameters
            api_key = request.POST.get('api_key')
            num_rows = int(request.POST.get('num_rows', 10))  # Default to 10 if not provided

            # Save the uploaded file
            file_path = save_file(uploaded_file)

            # Call the function to generate synthetic data
            generated_df = generate_synthetic_data(api_key, file_path, num_rows=num_rows)

            # Load original data for combination
            original_df = pd.read_excel(file_path)

            # Combine the original and generated data
            combined_df = pd.concat([original_df, generated_df], ignore_index=True)

            # Convert to CSV for download
            combined_csv = combined_df.to_csv(index=False)

            return JsonResponse({
                "message": "Synthetic data generated successfully.",
                "data": combined_csv
            })

        return JsonResponse({"error": "File not provided."}, status=400)

    return JsonResponse({"message": "Use POST method."}, status=405)


@csrf_exempt
def send_email(request):
    """
    API to send the generated synthetic data to an email address.
    """
    if request.method == 'POST':
        # Load JSON payload from the request
        data = json.loads(request.body)

        api_key = data.get('api_key')
        email = data.get('email')
        combined_data = data.get('combined_data')

        # Send email with the combined data as an attachment
        try:
            email_message = EmailMessage(
                "Synthetic Data Report",
                "Please find the attached synthetic data report.",
                to=[email]
            )
            email_message.attach("synthetic_data.csv", combined_data, 'text/csv')
            email_message.send()

            return JsonResponse({"message": "Email sent successfully."})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"message": "Use POST method."}, status=405)
