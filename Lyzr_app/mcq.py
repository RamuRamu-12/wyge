import openai
import re
from vyzeai.models.openai import ChatOpenAI


class MCQGeneratorAgent:
    def __init__(self, openai_api_key):
        self.llm = ChatOpenAI(api_key=openai_api_key)

    def parse_prompt(self, prompt):
        # Comprehensive pattern lists for each component, avoiding look-behind assertions
        topic_patterns = [
            r'(?:mcq[s]*|question[s]*|topic[s]*)(?: on| about| for)?\s*(.+?)(?=\s*with|\s*in|\s*for|,|$)',
            r'(?:generate|create|produce)\s*(.+?)(?=\s*mcq[s]*|\s*question[s]*|,|$)',
            r'topic\s*[:\-]?\s*(.+?)(?=\s*with|\s*in|\s*for|,|$)'
        ]
        count_patterns = [
            r'(\d+)\s*(?:questions|mcqs|items|problems|queries)',  # Match "10 questions" or "5 items"
            r'(?:generate|create|provide|give|produce|make)\s*(\d+)\s*(?:questions|mcqs|items|queries)?',# Match "generate 5 questions"
            r'(?:need|want|require|have|total of|up to)\s*(\d+)\s*(?:mcqs|questions|items|queries)?',# Match "need 3 questions" or "total of 10 items"
            r'(?:I\'d like|I need|give me)\s*(\d+)\s*(?:questions|mcqs|queries|items)?',  # Match "I'd like 8 questions"
            r'(\d+)\s*(?:multiple-choice|mcq|quiz)\s*(?:questions|items|queries)?',# Match "5 multiple-choice questions"
            r'(?:about|around|approximately)\s*(\d+)\s*(?:questions|mcqs|items|queries)?'  # Match "about 6 questions"
        ]
        difficulty_patterns = [
            r'(?:difficulty|level)\s*[:\-]?\s*(easy|medium|hard)',
            r'difficulty\s*is\s*(easy|medium|hard)',
            r'(easy|medium|hard)\s*(?:difficulty|level)?'
        ]
        language_patterns = [
            r'(?:language|in)\s*[:\-]?\s*(\w+)',
            r'language\s*is\s*(\w+)',
            r'\b(\w+)\s*language\b'
        ]

        # Helper function to find the first match from a list of patterns
        def find_first_match(patterns, text):
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            return None

        # Extract values, providing defaults if not specified
        topic = find_first_match(topic_patterns, prompt) or "General Knowledge"
        count = int(find_first_match(count_patterns, prompt) or 5)  # Default to 5 questions
        difficulty = find_first_match(difficulty_patterns, prompt) or "medium"  # Default to medium
        language = find_first_match(language_patterns, prompt) or "English"  # Default to English

        print("*******")
        print("Topic:", topic)
        print("Count:", count)
        print("Difficulty:", difficulty)
        print("Language:", language)
        print("*******")

        return topic, count, difficulty, language

    def generate_question_with_explanation(self, topic, difficulty, language):
        # Build the prompt with specified topic, difficulty, and language
        prompt = (
            f"Generate a multiple-choice question at {difficulty}-level difficulty on the topic of {topic} in {language}. "
            "The question should be well-phrased, clear, and cover important concepts related to the topic. "
            "Provide 4 distinct answer choices labeled A, B, C, D, with only one correct answer. "
            "The correct answer should be indicated clearly, and the explanation should include the reasoning behind the correct answer. "
            "Ensure that the explanation is concise, accurate, and easy to understand."
            "Also, provide a hint for the question to help users arrive at the correct answer. "
            "Finally, tag the question with the key concepts it covers."
        )

        # Use OpenAI API to get the question and explanations
        response = self.llm.run(prompt)
        return response


    def generate_mcq(self, topic, difficulty, language):
        # Generate and parse question data
        question_data = self.generate_question_with_explanation(topic, difficulty, language)
        print("@@@@@Question data will be@@@@@@")
        print(question_data)
        # question, choices, explanations, correct_answer = self.parse_question_data(question_data)

        # Format question data
        return question_data
        # return self.format_mcq(question, choices, explanations, correct_answer, difficulty)

    def generate_mcq_set(self, prompt):
        # Parse user-specified settings from prompt
        topic, count, difficulty, language = self.parse_prompt(prompt)

        if not topic.strip():
            return "Topic not specified. Please provide a valid topic for MCQ generation."

        # Generate specified number of MCQs and format the set
        mcqs = ""
        for i in range(count):
            mcqs += f"### Question {i + 1}\n"
            mcqs += self.generate_mcq(topic, difficulty, language)
            mcqs += "\n\n"

        return mcqs













