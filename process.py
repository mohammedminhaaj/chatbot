from preprocessing import preprocess_text
from intent_matching import get_intent, get_intent_response
from question_answer import process_user_query
from typing import Any
from utils.helpers import handle_capture_username
from utils.preference import clear_preferences_except_username
from utils.game_search import handle_game_search

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
        return ChatbotResponse(message=process_user_query(tokens), intent=intent)
    elif intent == "game_search":
        clear_preferences_except_username()
        return ChatbotResponse(
            message="Certainly! Could you please tell me what genre are you looking for? Genre can be either Action, Adventure, Puzzle or combination of any.", 
            intent=intent, 
            function_to_execute = lambda : handle_game_search())
    else:
        return ChatbotResponse(message=get_intent_response(intent), intent=intent)