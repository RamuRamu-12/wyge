from django.conf import settings
from django.core.mail import send_mail
from django.shortcuts import render
from django.http import HttpResponseNotFound, JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .database import PostgreSQLDB  
from .forms import AgentForm, AgentUpdateForm, CreateUserForm

# Initialize database connection
db = PostgreSQLDB(dbname='test', user='test_owner', password='tcWI7unQ6REA')

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from django.contrib.auth import authenticate, login
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User

@csrf_exempt
def loginPage(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        try:
            # Check if the user with the provided email exists
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return JsonResponse({"status": "error", "message": "Invalid email or password"}, status=401)

        # Authenticate the user using the username (retrieved from the email) and password
        user = authenticate(request, username=user.username, password=password)

        if user is not None:
            # Log the user in
            login(request, user)

            # Prepare the user details to return as a response
            user_details = {
                'username': user.username,
                'email': user.email,
                # Add other fields if needed
            }

            return JsonResponse({"status": "success", "user": user_details})
        else:
            return JsonResponse({"status": "error", "message": "Invalid email or password"}, status=401)

    return JsonResponse({"status": "error", "message": "Login failed"}, status=400)


@csrf_exempt
def registerPage(request):
    if request.method == 'POST':
        form = CreateUserForm(request.POST)
        try:
            if form.is_valid():
                form.save()
                user_name = form.cleaned_data.get('username')
                password1 = form.cleaned_data.get('password1')
                first_name = request.POST.get('first_name')
                last_name = request.POST.get('last_name')
                address = request.POST.get('address')
                email = form.cleaned_data.get('email')
                mobile = request.POST.get('mobile')
                
                # Adding user to the custom database
                db.add_user(user_name, password1, first_name, last_name, address, email, mobile)
                return JsonResponse({"status": "success", "message": "Registration successful"})
            else:
                # Return form errors if the form is not valid
                return JsonResponse({"status": "error", "errors": form.errors}, status=400)
        except Exception as e:
            # Return an error message in case of an exception
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    # Return an error response if the request method is not POST
    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)


@csrf_exempt
def get_user_details(request):
    if request.method == 'POST':
        email = request.POST.get("email")
        if not email:
            return JsonResponse({"status": "error", "message": "Email not provided."}, status=400)

        user_data = db.get_user_data(email)
        if not user_data:
            return JsonResponse({"status": "error", "message": "User not found."}, status=404)

        return JsonResponse({"status": "success", "userinfo": list(user_data)}, safe=False)

    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)


@csrf_exempt
def googlelogin(request):
    if request.method == 'POST':
        try:
            username = request.POST.get("username")
            user_id = request.POST.get("id")
            email = request.POST.get("email")

            # Generate a password
            password = "auto@" + user_id

            # Check if the user already exists
            users = db.get_users()
            print(users)
            if email in [user[0] for user in users]:
                user_details = db.get_user_data(email)
                return JsonResponse({"status": "success", "user_details": user_details})
            else:
                # Create a new user
                form = CreateUserForm({
                    'username': username,
                    'email': email,
                    'password1': password,
                    'password2': password
                })
                if form.is_valid():
                    form.save()
                    db.add_user(
                        user_name=username,
                        password=password,
                        first_name="",  # Can get these from the request if available
                        last_name="",
                        address="",
                        email=email,
                        mobile=""
                    )
                    user_details = db.get_user_data(email)
                    return JsonResponse({"status": "success", "user_details": user_details})
                else:
                    return JsonResponse({"status": "error", "errors": form.errors}, status=400)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)


@csrf_exempt
def agent_list(request):
    # Get the search query, filters, and sorting options from the GET request
    search_query = request.GET.get('search', '')  # Get search query, default to an empty string
    category_filter = request.GET.getlist('category')  # Get list of selected categories
    industry_filter = request.GET.getlist('industry')  # Get list of selected industries
    pricing_filter = request.GET.getlist('pricing')  # Get list of selected pricing models
    accessory_filter = request.GET.getlist('accessory_model')  # Get list of selected accessories
    sort_option = request.GET.get('sort', 'date_added')  # Get sorting option, default to 'date_added'
    
    try:
        

        # Fetch agents based on filters, search, and sorting
        agents = db.get_filtered_agents(
            search_query=search_query,
            category_filter=category_filter,
            industry_filter=industry_filter,
            pricing_filter=pricing_filter,
            accessory_filter=accessory_filter,
            sort_option=sort_option,
            is_approved=True  # Ensure only approved agents are displayed
        )

        # Return the agents as JSON response
        return JsonResponse({'agents': agents}, safe=False)
    except Exception as e:
        # Handle errors and return error response
        return JsonResponse({'error': str(e)}, status=500)


# 2. Agent Detail View
@csrf_exempt
def agent_detail(request, id):
    try:
        agent = db.get_agent_by_id(id)  # Assuming this returns a tuple with all agent fields in order

        # Check if the agent exists and includes `is_approved` and `status`
        if agent:
            is_approved = agent[20]  # Assuming `is_approved` is at index 20
            status = agent[21]  # Assuming `status` is at index 21

            # Proceed only if the agent is approved
            if is_approved or (status == 'active'):  # Check `status` or `is_approved`
                agent_data = {
                    "id": agent[0],
                    "name": agent[1],
                    "description": agent[2],
                    "email": agent[8],
                    "overview": agent[11],
                    "key_features": agent[12],
                    "use_cases": agent[13],
                    "tag": agent[16],
                    "tagline": agent[9],
                    "details": {
                        "created_by": agent[14],
                        "category": agent[3],
                        "industry": agent[4],
                        "pricing": agent[5],
                        "access": agent[15],
                        "date_added": agent[20].strftime('%Y-%m-%d'),
                        "status": status  # Include status in details
                    },
                    "website_url": agent[7],
                    "preview_image": agent[17],
                    "demo_video": agent[19],
                    "logo": agent[18]
                }
                return JsonResponse({'agent': agent_data})
            else:
                return HttpResponseNotFound(JsonResponse({'error': 'Agent not approved or inactive'}))
        else:
            return HttpResponseNotFound(JsonResponse({'error': 'Agent not found'}))
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

#Agent Creation.
@csrf_exempt
def agent_create(request):
    if request.method == 'POST':
        form = AgentForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            try:
                # Convert any array fields into proper arrays (handling empty strings)
                key_features = data.get('key_features', '')
                use_cases = data.get('use_cases', '')
                tags = data.get('tags', '')

                # Handle empty strings by converting them to empty arrays
                key_features_array = key_features.split(',') if key_features else []
                use_cases_array = use_cases.split(',') if use_cases else []
                tags_array = tags.split(',') if tags else []

                # Debug: Print the data before adding to the database
                print("Data before creating agent:", data)

                # Add agent to the database via PostgreSQLDB
                new_agent_id = db.add_agent(
                    name=data.get('name'),
                    description=data.get('description'),
                    category=data.get('category'),
                    industry=data.get('industry'),
                    pricing=data.get('pricing'),
                    accessory_model=data.get('accessory_model'),
                    website_url=data.get('website_url'),
                    email=data.get('email'),
                    tagline=data.get('tagline'),
                    likes=data.get('likes', 0),
                    overview=data.get('overview'),
                    key_features=key_features_array,
                    use_cases=use_cases_array,
                    created_by=data.get('created_by'),
                    access=data.get('access'),
                    tags=tags_array,
                    preview_image=data.get('preview_image'),
                    logo=data.get('logo'),
                    demo_video=data.get('demo_video'),
                    is_approved=False,  # New agents are not approved by default
                    status=data.get('status', 'pending')  # Set default status as 'pending' if not provided
                )

                # Debug: Print the new agent ID after creation
                print("New agent ID:", new_agent_id)

                if new_agent_id:
                    return JsonResponse({
                        'message': 'Agent created successfully and sent for approval',
                        'agent_id': new_agent_id
                    })
                else:
                    print("Failed to add agent to the database")  # Debug statement
                    return JsonResponse({'error': 'Failed to add agent to the database'}, status=500)

            except Exception as e:
                print("Error occurred:", str(e))  # Debug: Print the error
                return JsonResponse({'error': str(e)}, status=500)
        else:
            print("Form validation errors:", form.errors)  # Debug: Print form errors
            return JsonResponse({'errors': form.errors}, status=400)
    else:
        return JsonResponse({'message': 'Only POST requests are allowed'}, status=405)


# API for modifying agent details by admin
@csrf_exempt
def modify_agent(request, agent_id):
    if request.method == 'POST':
        form = AgentUpdateForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            try:
                # Fetch the existing agent by ID
                agent = db.get_agent_by_id(agent_id)
                if not agent:
                    return JsonResponse({'error': 'Agent not found'}, status=404)

                # Handle key features, use cases, and tags arrays
                key_features = data.get('key_features', agent[12])
                use_cases = data.get('use_cases', agent[13])
                tags = data.get('tags', agent[16])

                key_features_array = key_features.split(',') if key_features else agent[12]
                use_cases_array = use_cases.split(',') if use_cases else agent[13]
                tags_array = tags.split(',') if tags else agent[16]

                # Update the agent's details in the database
                db.update_agent(
                    agent_id=agent_id,
                    name=data.get('name', agent[1]),
                    description=data.get('description', agent[2]),
                    category=data.get('category', agent[3]),
                    industry=data.get('industry', agent[4]),
                    pricing=data.get('pricing', agent[5]),
                    accessory_model=data.get('accessory_model', agent[6]),
                    website_url=data.get('website_url', agent[7]),
                    email=data.get('email', agent[8]),
                    tagline=data.get('tagline', agent[9]),
                    likes=data.get('likes', agent[10]),
                    overview=data.get('overview', agent[11]),
                    key_features=key_features_array,
                    use_cases=use_cases_array,
                    created_by=data.get('created_by', agent[14]),
                    access=data.get('access', agent[15]),
                    tags=tags_array,
                    preview_image=data.get('preview_image', agent[17]),
                    logo=data.get('logo', agent[18]),
                    demo_video=data.get('demo_video', agent[19]),
                    is_approved=True,  # Ensure approval status
                    status='approved'  # Set status to 'approved' after modification
                )

                return JsonResponse({'message': 'Agent updated and approved successfully'})

            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)

        else:
            return JsonResponse({'errors': form.errors}, status=400)

    return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)


#Deleting the agent if we dont need
@csrf_exempt
def reject_agent(request, agent_id):
    if request.method == 'GET':  # Using GET instead of DELETE for marking as rejected
        try:
            # Check if the agent exists
            agent = db.get_agent_by_id(agent_id)
            print(agent)
            if agent:
                # Agent exists, update the status to 'rejected'
                db.update_agent(agent_id=agent_id, status='rejected')
                return JsonResponse({'message': 'Agent status updated to rejected successfully'}, status=200)  # OK status
            else:
                # Agent does not exist
                return JsonResponse({'error': 'Agent not found'}, status=404)  # Not found status

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)  # Internal server error status

    return JsonResponse({'error': 'Invalid HTTP method'}, status=405)


#Viewing the agents based on the status type
@csrf_exempt
def view_agents_by_status(request):
    if request.method == 'GET':
        # Get the 'status' parameter from the query string
        status = request.GET.get('status')

        # Check if status parameter is provided and valid
        if status not in ['pending', 'approved', 'rejected']:
            return JsonResponse(
                {'error': 'Invalid status parameter. Choose from "pending", "approved", or "rejected".'}, status=400)

        try:
            # Retrieve agents based on the status
            agents = db.get_agents_by_status(status)  # Assuming this function retrieves agents by status

            # Transform the data into a list of dictionaries for JSON response
            agents_data = [
                {
                    "id": agent[0],
                    "name": agent[1],
                    "description": agent[2],
                    "category": agent[3],
                    "industry": agent[4],
                    "pricing": agent[5],
                    "website_url": agent[7],
                    "email": agent[8],
                    "tagline": agent[9],
                    "likes": agent[10],
                    "overview": agent[11],
                    "key_features": agent[12],
                    "use_cases": agent[13],
                    "created_by": agent[14],
                    "access": agent[15],
                    "tags": agent[16],
                    "preview_image": agent[17],
                    "logo": agent[18],
                    "demo_video": agent[19],
                    "date_added": agent[20].strftime('%Y-%m-%d'),
                    "is_approved": agent[20],
                    "status": agent[21]  # Assuming status is at index 21
                }
                for agent in agents
            ]

            # Return the agents data
            return JsonResponse({'agents': agents_data}, status=200)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Only GET requests are allowed'}, status=405)


# Adding the email.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def add_email(request):
    if request.method == 'POST':
        try:
            email = request.POST.get('email')

            # Check if the email field is present
            if not email:
                return JsonResponse({'error': 'Email is required'}, status=400)

            # Insert the email into the database (Assuming db.insert_email is a valid function)
            email_id = db.insert_email(email)

            if email_id:
                return JsonResponse({'message': 'Email added successfully', 'email_id': email_id}, status=201)
            else:
                return JsonResponse({'error': 'Failed to add email'}, status=500)

        except Exception as e:
            return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid HTTP method'}, status=405)

    



#Hackathon page code
@csrf_exempt
def create_submission_view(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        github_profile = request.POST.get('github_profile')
        linkedin_profile = request.POST.get('linkedin_profile')
        country = request.POST.get('country')
        affiliation_status = request.POST.get('affiliation_status')

        # Create submission
        submission_id = db.create_submission(first_name, last_name, email, github_profile, linkedin_profile, country, affiliation_status)

        return JsonResponse({'submission_id': submission_id}, status=201)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

@csrf_exempt
def get_submission_view(request, submission_id):
    if request.method == 'GET':
        submission = db.get_submission(submission_id)
        if submission:
            return JsonResponse({
                "id": submission[0],
                "first_name": submission[1],
                "last_name": submission[2],
                "email": submission[3],
                "github_profile": submission[4],
                "linkedin_profile": submission[5],
                "country": submission[6],
                "affiliation_status": submission[7],
                "created_at": submission[8]
            })
        else:
            return JsonResponse({"error": "Submission not found"}, status=404)

@csrf_exempt
def update_submission_view(request, submission_id):
    if request.method == 'POST':
        try:
            first_name = request.POST.get('first_name')
            last_name = request.POST.get('last_name')
            email = request.POST.get('email')
            github_profile = request.POST.get('github_profile')
            linkedin_profile = request.POST.get('linkedin_profile')
            country = request.POST.get('country')
            affiliation_status = request.POST.get('affiliation_status')

            db.update_submission(
                submission_id, first_name, last_name, email, github_profile, linkedin_profile, country, affiliation_status
            )
            return JsonResponse({"status": "Submission updated successfully"})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

@csrf_exempt
def delete_submission_view(request, submission_id):
    if request.method == 'DELETE':
        try:
            db.delete_submission(submission_id)
            return JsonResponse({"status": "Submission deleted successfully"})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)


@csrf_exempt
def get_all_submissions_view(request):
    if request.method == 'GET':
        try:
            submissions = db.get_all_submissions()

            # Format the submissions data to a list of dictionaries
            submissions_data = [
                {
                    "id": submission[0],
                    "first_name": submission[1],
                    "last_name": submission[2],
                    "email": submission[3],
                    "github_profile": submission[4],
                    "linkedin_profile": submission[5],
                    "country": submission[6],
                    "affiliation_status": submission[7],
                    "created_at": submission[8]  # Adjust index based on your table structure
                }
                for submission in submissions
            ]

            return JsonResponse({"submissions": submissions_data}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)


#Hackathon partners code:
@csrf_exempt
def create_partner_view(request):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        company_name = request.POST.get('company_name')
        sponsorship_level = request.POST.get('sponsorship_level')

        # Create partner entry
        try:
            partner_id = db.create_partner(first_name, last_name, email, company_name, sponsorship_level)
            return JsonResponse({'partner_id': partner_id}, status=201)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def get_partner_details_view(request, partner_id):
    if request.method == 'GET':
        partner = db.get_partner_details(partner_id)
        if partner:
            return JsonResponse({
                "id": partner[0],
                "first_name": partner[1],
                "last_name": partner[2],
                "email": partner[3],
                "company_name": partner[4],
                "sponsorship_level": partner[5],
            })
        else:
            return JsonResponse({"error": "Partner not found"}, status=404)

    return JsonResponse({"error": "Invalid request method"}, status=400)


@csrf_exempt
def update_partner_view(request, partner_id):
    if request.method == 'POST':
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        company_name = request.POST.get('company_name')
        sponsorship_level = request.POST.get('sponsorship_level')

        # Update partner entry
        try:
            db.update_partner(
                partner_id, first_name, last_name, email, company_name, sponsorship_level
            )
            return JsonResponse({"status": "Partner updated successfully"})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=400)


@csrf_exempt
def delete_partner_view(request, partner_id):
    if request.method == 'DELETE':
        try:
            db.delete_partner(partner_id)
            return JsonResponse({"status": "Partner deleted successfully"})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=400)


@csrf_exempt
def get_all_partners_view(request):
    if request.method == 'GET':
        try:
            partners = db.get_all_partners()
            # Format the partners data to a list of dictionaries
            partners_data = [
                {
                    "id": partner[0],  # Assuming partner[0] is the id
                    "first_name": partner[1],
                    "last_name": partner[2],
                    "email": partner[3],
                    "company_name": partner[4],
                    "sponsorship_level": partner[5],
                }
                for partner in partners
            ]

            return JsonResponse({"partners": partners_data}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=400)