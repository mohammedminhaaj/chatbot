from nltk.tokenize import word_tokenize
from utils import constants
from utils.helpers import get_similar_result

def handle_game_search() -> None:
    """
    Helper function to handle logic when intent is game_search
    
    Parameters
    ----------
        None

    Returns
    -------
        None
    """

    while True:
        genre_input = input(f"{constants.USER}: ")
        genre_input_tokens = word_tokenize(genre_input)
        result, result_response, result_type = get_similar_result(
            tokens=genre_input_tokens,
            file_name="data/game_search.csv",
            query_header_name="keyword",
            result_header_name=["value", "response", "type"],
            threshold=0.7
        )
        
        if result == "not_found" or "genre" not in result_type:
            print(f"{constants.CHATBOT_NAME}: Sorry, I couldn't understand. Could you please provide me with the genre? Genre can be either Action, Adventure, Puzzle or combination of any.")
        else:
            print(f"{constants.CHATBOT_NAME}: {result_response}. Could you please tell me what platform are you currently using? Platform can be either PC, Xbox, PlayStation or combination of any.")
            break

    while True:
        platform_input = input(f"{constants.USER}: ")
        platform_input_tokens = word_tokenize(platform_input)
        result, result_response, result_type = get_similar_result(
            tokens=platform_input_tokens,
            file_name="data/game_search.csv",
            query_header_name="keyword",
            result_header_name=["value", "response", "type"],
            threshold=0.7
        )
        
        if result == "not_found" or "platform" not in result_type:
            print(f"{constants.CHATBOT_NAME}: Sorry, I couldn't understand. Could you please provide me with the platform? Platform can be either PC, Xbox, PlayStation or combination of any.")
        else:
            print(f"{constants.CHATBOT_NAME}: Here is a list of games {result_response if result != 'na' else ''}:\n\t1.Game 1\n\t2.Game 2\n\t3.Game 3")
            break

    
