import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from vyzeai.agents.prebuilt_agents import ResearchAgent, LinkedInAgent, VideoAudioBlogAgent, YTBlogAgent
import json

upload_dir = 'media/'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# Helper function to save uploaded file
def save_file(file):
    file_path = f"media/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.read())
    return file_path

# API to handle LinkedIn post generation
@csrf_exempt
def generate_linkedin_post(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        api_key = data.get('api_key')
        post_type = data.get('post_type')
        linkedin_agent = LinkedInAgent(api_key)

        content, image_path = None, None
        print("execution started")
        try:
            if post_type == 'website':
                topic = data.get('topic')
                url = data.get('url')

                research_agent = ResearchAgent(api_key)
                context = research_agent.research(topic, url)
                print("Context from ResearchAgent:", context)  # Debug print
                content, image_path = linkedin_agent.generate_linkedin_post(context)

            elif post_type == 'youtube':
                yt_url = data.get('yt_url')
                yt_agent = YTBlogAgent(api_key)
                context = yt_agent.extract_transcript(yt_url)
                print("Context from YTBlogAgent:", context)  # Debug print
                content, image_path = linkedin_agent.generate_linkedin_post(context)

            elif post_type == 'video/audio':
                file_path = data.get('file_path')
                va_agent = VideoAudioBlogAgent(api_key)
                context = va_agent.extract_text(file_path)
                print("Context from VideoAudioBlogAgent:", context)  # Debug print
                content, image_path = linkedin_agent.generate_linkedin_post(context)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

        return JsonResponse({
            'content': content,
            'image_path': image_path
        })


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

# API to post content on LinkedIn
@csrf_exempt
def post_on_linkedin(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        api_key = data.get('api_key')
        token = data.get('token')
        content = data.get('content')
        image_path = data.get('image_path')

        linkedin_agent = LinkedInAgent(api_key)
        ack = linkedin_agent.post_content_on_linkedin(token, content, image_path)
        
        return JsonResponse({'ack': ack})

# Separate API to send email
@csrf_exempt
def send_email(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        recipient_email = data.get('email')
        subject = data.get('subject')
        message = data.get('message')
        
        # Logic to send email (e.g., using Django's Email module)
        from django.core.mail import send_mail
        from django.conf import settings

        send_mail(
            subject,
            message,
            settings.DEFAULT_FROM_EMAIL,
            [recipient_email],
            fail_silently=False,
        )

        return JsonResponse({'status': 'Email sent successfully'})

