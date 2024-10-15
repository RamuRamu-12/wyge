import os
import json

token = os.getenv('token')
refresh_token = os.getenv('refresh_token')
token_uri = os.getenv('token_uri')
client_id_mail = os.getenv('client_id_mail')
client_secret = os.getenv('client_secret')
scopes = os.getenv('scopes')
universe_domain = os.getenv('universe_domain')
account = os.getenv('account')
expiry = os.getenv('expiry')

token_info = {
    "token": token,
    "refresh_token": refresh_token,
    "token_uri": token_uri,
    "client_id": client_id_mail,
    "client_secret": client_secret,
    "scopes": ["https://www.googleapis.com/auth/gmail.send"],
    "universe_domain": universe_domain,
    "account": account,
    "expiry": expiry
}

# with open('token.json', 'w') as f:
#     json.dump(token_info, f)
# print(os.path.exists("token.json"))


import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from vyzeai.agents.prebuilt_agents import ResearchAgent, VideoAudioBlogAgent, YTBlogAgent, BlogAgent, VideoAgent, EmailAgent

upload_dir = 'media/'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Helper function to save uploaded file
def save_file(file):
    file_path = f"media/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.read())
    return file_path

# API to handle file uploads
@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if file:
            file_path = save_file(file)
            return JsonResponse({'file_path': file_path})
        else:
            return JsonResponse({'error': 'No file uploaded'}, status=400)


@csrf_exempt
def generate_content(request):
    """
    API for generating content from the provided source (website, YouTube, etc.)
    """
    if request.method == 'POST':
        data = json.loads(request.body)

        api_key = data.get('api_key')
        option = data.get('option')
        topic = data.get('topic', '')
        url = data.get('url', '')
        yt_url = data.get('yt_url', '')
        file_path = data.get('file_path', '')

        content = None
        image_path = None

        try:
            if option == 'website url to blog':
                research_agent = ResearchAgent(api_key)
                blog_agent = BlogAgent(api_key)
                context = research_agent.research(topic, url)
                contents = blog_agent.generate_blog(topic, url, context)
                content = contents[0][1]
                image_path = contents[-1][-1][0]

            elif option == 'youtube url to blog':
                yt_agent = YTBlogAgent(api_key)
                contents = yt_agent.generate_blog(yt_url)
                content = contents[0][1]
                image_path = contents[-1][-1][0]

            elif option == 'video/audio to blog':
                va_agent = VideoAudioBlogAgent(api_key)
                contents = va_agent.generate_blog(file_path)
                content = contents[0][1]
                image_path = contents[-1][-1][0]

            elif option == 'website url to video':
                research_agent = ResearchAgent(api_key)
                video_agent = VideoAgent(api_key)
                context = research_agent.research(topic, url)
                content = video_agent.generate_video(topic, context)[0]

            return JsonResponse({
                "content": content,
                "image_path": image_path
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"message": "Use POST method"}, status=405)


@csrf_exempt
def send_email(request):
    """
    API for sending the generated content via email.
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        api_key = data.get('api_key')
        to_mail = data.get('to_mail')
        content = data.get('content')
        token_info = data.get('token_info', './token.json')  #C:\Users\rammohan\PycharmProjects\linkedin\DIGIOTAI\Email_Blog\token.json
        #C:\Users\rammohan\PycharmProjects\linkedin\DIGIOTAI\Email_Blog\Email_Blog_app\views.py
        try:
            email_agent = EmailAgent(api_key)
            ack = email_agent.send_email(
                to_mail,
                'Your content',
                'Thank you for using our product.',
                content,
                token_json_file_path=token_info
            )


            return JsonResponse({"ack": ack[0]})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"message": "Use POST method"}, status=405)

