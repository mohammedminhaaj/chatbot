from preprocessing import preprocess_text
from intent_matching import get_intent, get_intent_response
from question_answer import process_user_query
from typing import Any
from identity_management import handle_capture_username
from utils.preference import clear_preferences_except_username
from game_search import handle_game_search, extract_genre_platform, get_game_search_message
from game_fact import handle_game_fact

class ChatbotResponse:
    """
    Class to hold chatbot response along with some other characteristics
    """

    def __init__(self, message: str, intent: str | None = None, function_to_execute: Any | None = None) -> None:
        self.message = message
        self.intent = intent
        self.function_to_execute = function_to_execute
    
    def __str__(self) -> str:
        return self.message


def chatbot_response(user_input: str) -> ChatbotResponse:
    """
    Function to process user input and return the bot response
    
    Parameters
    ----------
        user_input: str
            The actual input in str format from the user

    Returns
    -------
        ChatbotResponse: class
            ChatbotResponse class which contains processed text
    """
    try:
        # Pre-process the text to get the user input in the form of tokens
        tokens = preprocess_text(user_input)

        # Get the intent using the tokens
        intent = get_intent(tokens)

        # Check if the intent is change_name
        if intent == "change_name":
            return ChatbotResponse(
                message="Can you please provide your name again?", 
                intent=intent, 
                function_to_execute = lambda : handle_capture_username()
            )
        # Check if the intent is announce_name or general
        elif intent in ["announce_name", "general"]:
            # Return response from the question_answers csv
            return ChatbotResponse(message=process_user_query(tokens), intent=intent)
        
        # Check if intent is game_search, platform_recommendation or genre_exploration
        elif intent in ["game_search", "platform_recommendation", "genre_exploration"]:
            # Clear the preferences except username
            clear_preferences_except_username()

            # Check if the user input has multiple intents
            # Extract the genre and platform if available
            captured_genre, captured_platform = extract_genre_platform(tokens=tokens)

            return ChatbotResponse(
                message=get_game_search_message(captured_genre, captured_platform), 
                intent=intent,
                function_to_execute = lambda : handle_game_search(captured_genre, captured_platform))
        elif intent in ["game_fact", "game_genre_fact", "game_platform_fact"]:
           return ChatbotResponse(
                message="Please wait...", 
                intent=intent, 
                function_to_execute = lambda : handle_game_fact(tokens=tokens, intent=intent)
            ) 
        else:
            return ChatbotResponse(message=get_intent_response(intent), intent=intent)
    except Exception:
        return ChatbotResponse(message="Something went wrong! Could you please restart the program?")