
import re
import openai
import os


class SequentialFlow:
    def __init__(self, agent, model):
        self.agent = agent
        self.model = model

    def generate_prompt(self, user_prompt):
        return f"{self.agent} {user_prompt}"

    def execute(self, user_prompt):
        full_prompt = self.generate_prompt(user_prompt)
        return self.model.generate_image(full_prompt)


class Task:
    def __init__(self, description):
        self.description = description

    def __str__(self):
        return self.description


class InputType:
    def __init__(self, input_description):
        self.input_description = input_description

    def __str__(self):
        return self.input_description


class OutputType:
    def __init__(self, output_description):
        self.output_description = output_description

    def __str__(self):
        return self.output_description


class Agent:
    def __init__(self, expertise, task, input_type, output_type):
        self.expertise = expertise
        self.task = task
        self.input_type = input_type
        self.output_type = output_type

    def __str__(self):
        return f"You are {self.expertise}, your task is to {self.task}. I will give you {self.input_type}, give me a {self.output_type}."


class OpenAIModel:
    def __init__(self, api_key=None, model="dall-e-2"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=openai.api_key)
        openai.api_key = self.api_key
        self.model = model

    def generate_image(self, prompt):
        response = self.client.images.generate(
            prompt=prompt,
            n=1,
            model=self.model,
            size="1024x1024",
            quality="standard"
        )
        return response.data[0].url


class InteriorDesignAgent:
    def parse_prompt(self, prompt):
        # Define regex patterns for parsing
        style_patterns = [
            r"(?:style|design|inspired by|in the style of)\s*[:\-]?\s*(.+?)(?=,|\s*room|\s*color|\s*instructions|$)"
        ]
        room_color_patterns = [
            r"(?:room color|color)\s*[:\-]?\s*(.+?)(?=,|\s*room|\s*instructions|$)"
        ]
        room_type_patterns = [
            r"(?:room type|type)\s*[:\-]?\s*(.+?)(?=,|\s*color|\s*instructions|$)"
        ]
        instructions_patterns = [
            r"(?:instructions|additional instructions)\s*[:\-]?\s*(.+?)(?=$)"
        ]

        # Helper function to find the first match from a list of patterns
        def find_first_match(patterns, text):
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            return None

        # Extract values
        style = find_first_match(style_patterns, prompt) or "Modern"
        room_color = find_first_match(room_color_patterns, prompt) or "Neutral"
        room_type = find_first_match(room_type_patterns, prompt) or "Living Room"
        instructions = find_first_match(instructions_patterns, prompt) or "No specific instructions"

        return style, room_color, room_type, instructions

    def generate_design(self, prompt):
        # Parse the prompt to extract details
        expertise = "Interior Designer"
        task = Task("Image Generation")
        input_type = InputType("Text")
        output_type = OutputType("Image")
        agent = Agent(expertise, task, input_type, output_type)

        style, room_color, room_type, instructions = self.parse_prompt(prompt)
        api_key = os.getenv("OPENAI_API_KEY")
        model = OpenAIModel(api_key=api_key, model="dall-e-2")
        sequential_flow = SequentialFlow(agent, model)

        # Construct the detailed design prompt
        full_prompt = (
            f"Generate a realistic-looking interior design with the following specifications: "
            f"Style: {style}, Room Color: {room_color}, Room Type: {room_type}, Instructions: {instructions}"
        )

        # Use the model to generate the design image URL
        image_url = sequential_flow.execute(full_prompt)
        return image_url


