import requests
from opencage.geocoder import OpenCageGeocode
import openai
import re
from wyge.models.openai import ChatOpenAI


# Function to fetch latitude and longitude for the destination using OpenCage SDK
def fetch_lat_long(api_key, location):
    geocoder = OpenCageGeocode(api_key)
    results = geocoder.geocode(location)
    if results:
        lat = results[0]['geometry']['lat']
        lon = results[0]['geometry']['lng']
        return lat, lon
    return None, None


# Function to fetch real-time weather data
def fetch_weather(api_key, lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        print(weather_data)
        return {
            "temperature": weather_data["main"]["temp"] - 273.15,  # Convert Kelvin to Celsius
            "condition": weather_data["weather"][0]["description"].capitalize()
        }
    return "Weather data not available."


# TravelPlannerAgent class that generates a travel plan based on user input
class TravelPlannerAgent:
    def __init__(self, weather_api_key, geolocation_api_key, openai_api_key):
        self.weather_api_key = weather_api_key
        self.geolocation_api_key = geolocation_api_key
        self.llm = ChatOpenAI(api_key=openai_api_key)  # Initialize OpenAI LLM

    def parse_user_input(self, prompt):
        # Extract destination and duration (days) from the prompt
        match = re.search(r'travel plan to (\w+)(?: for (\d+) days)?', prompt, re.IGNORECASE)
        destination = match.group(1)
        days = int(match.group(2)) if match.group(2) else 3  # default to 3 days if not specified
        return destination, days

    def gather_destination_info(self, destination):
        # Get latitude and longitude of the destination
        lat, lon = fetch_lat_long(self.geolocation_api_key, destination)
        if lat is None or lon is None:
            return "Unable to retrieve location data."

        # Fetch real-time weather data for the specified destination using latitude and longitude
        weather = fetch_weather(self.weather_api_key, lat, lon)

        return {
            "lat": lat,
            "lon": lon,
            "weather": weather,
        }

    def generate_itinerary(self, destination, days):
        # Gather real-time context data
        context = self.gather_destination_info(destination)
        print("context is ...................")
        print(context)
        if isinstance(context, str):
            return context  # Return error message if location data is unavailable

        # LLM prompt with a structured request
        prompt = (
            # f"Create a comprehensive travel itinerary for {destination} over {days} days. "
            # f"Include local attractions, historical sites, and unique activities. "
            # f"Structure each day with 'Morning,' 'Afternoon,' and 'Evening' sections, "
            # f"and use bullet points to list the activities for each part of the day. "
            # f"Provide brief descriptions for each place, mentioning its significance or unique qualities. "
            # f"Include weather-appropriate recommendations based on current weather conditions: "
            # f"Temperature: {context['weather']['temperature']}°C, Condition: {context['weather']['condition']}. "
            # f"Optimize for cost-effective options where possible.\n\n"
            # f"Format the itinerary in markdown with clear headers, bullet points, and sections. "
            # f"Highlight important tips, and end with a short summary of the trip."

            f"Create a comprehensive travel itinerary for {destination} over {days} days. "
            f"Provide brief description  and importance of {destination}, mentioning its significance or unique qualities. "
            f"Include recommendations for local attractions, historical sites, and unique activities, optimizing for the region's latitude and longitude, and adapting plans based on weather conditions. "
            f" Do not structure each day like morning,afternoon,evening sections, and use bullet points to list activities for each day. "
            f"Suggest suitable accommodations based on the location's latitude and longitude, factoring in weather conditions and accessibility to major attractions. "
            f"Adapt food and accommodation suggestions based on temperature ({context['weather']['temperature']}°C) and condition ({context['weather']['condition']}), "
            f"including ideas like indoor dining spots for rainy days or places with outdoor seating in pleasant weather."
            f"Offer helpful travel tips tailored to the location, such as recommended modes of transport (e.g., metro, bikes, or taxis), nearby transportation hubs, and local etiquette. "
            f"Highlight any important safety tips or local customs that visitors should be aware of to enhance their experience and avoid cultural misunderstandings. "
            f"Provide practical packing tips based on the weather, suggesting items like umbrellas, sunscreen, or specific clothing for comfort. "
            f"Format the itinerary in markdown with clear headers, bullet points, and sections. "
            f"Recommend the famous foods that reflect the destination’s culture and cuisine. "
            # f"End with a short summary of the trip and highlight any essential tips or recommendations for a successful journey."

        )
        print("llm going")
        # Use OpenAI to generate itinerary
        response = self.llm.run(prompt)
        print("______________________________________________________")
        return response

    def generate_travel_plan(self, prompt):
        # Parse prompt and create the travel plan
        destination, days = self.parse_user_input(prompt)
        itinerary = self.generate_itinerary(destination, days)

        travel_plan = f"### Travel Itinerary for {destination} - {days} Days\n\n{itinerary}"
        return travel_plan



