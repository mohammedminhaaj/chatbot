from nltk.tokenize import word_tokenize
from utils import constants
from utils.helpers import get_similar_result
import pandas as pd
from nltk.corpus import stopwords

# Constant messages for different types
GENRE_MESSAGE = "Certainly! Could you please tell me what genre are you looking for? Currently, we only have Action, Adventure, Puzzle games."
GENRE_REPROMPT = "Sorry, I couldn't understand. Could you please provide me with the genre? Genre can be either Action, Adventure, Puzzle."
PLATFORM_MESSAGE = "Could you please tell me what platform are you currently using? Currently, we only have PC, Xbox and PlayStation games."
PLATFORM_REPROMPT = "Sorry, I couldn't understand. Could you please provide me with the platform? Platform can be either PC, Xbox, PlayStation."

def get_game_search_message(genre: str | None, platform: str | None) -> str:
    """
    Helper function to get the message based on availabilty of genre and platform
    
    Parameters
    ----------
        genre: str
            the genre of the game
        platform: str
            the platform of the game

    Returns
    -------
        str
            message that needs to be displayed
    """
    # If genre and platform both are available
    if genre and platform:
        return "Please wait..."
    
    # If only genre is available
    elif genre and not platform:
        return PLATFORM_MESSAGE
    
    # Any other cases
    else:
        return GENRE_MESSAGE

def extract_genre_platform(tokens: list[str]) -> tuple[str | None]:
    """
    Helper function to extract the genre or platform from a set of tokens
    
    Parameters
    ----------
        tokens: list[str]
            list of extracted tokens from the user input

    Returns
    -------
        tuple[str | None]
            tuple having genre and platform if present
    """
    # Remove stopwords from the token to reduce the list
    tokens = [word.lower() for word in tokens if word.lower() not in stopwords.words("english")]

    # Initialize empty lists
    captured_platforms = []
    captured_genres = []

    # Loop through tokens and check if any of the tokens are present in available platforms or available genres
    for token in tokens:
        if token in constants.AVAILABLE_PLATFORMS:
            captured_platforms.append(token)
        if token in constants.AVAILABLE_GENRES:
            captured_genres.append(token)

    # Concatenate the list to form a comma seperated string
    captured_platform_string = ",".join(captured_platforms) if captured_platforms else None
    captured_genre_string = ",".join(captured_genres) if captured_genres else None
    
    # Return the captured genre or captured platform from the user input tokens
    return (captured_genre_string, captured_platform_string)



def search_game(genre: str, platform: str) -> list[str]:
    """
    Helper function to search games in games.csv by using their genre and platform
    
    Parameters
    ----------
        genre: str
            the genre of the game
        platform: str
            the platform in which the game runs

    Returns
    -------
        list[str]
            List of games in random order
    """

    # Read the csv file to dataframe
    df = pd.read_csv("data/games.csv")

    # Check if the genre is available
    if genre != "na":
        # Split the string into list of keywords
        genre_list = genre.split(",")

        # Filter the games belonging to the given genre
        df = df[df["genre"].apply(lambda x: all(keyword in x.lower() for keyword in genre_list))]

    # Check if the platform is available
    if platform != "na":
        # Split the string into list of keywords
        platform_list = platform.split(",")

        # Filter the games belonging to the given platform
        df = df[df["platform"].apply(lambda x: all(keyword in x.lower() for keyword in platform_list))]

    # Pick only the name and short_description column from the data frame
    result_df = df[["name", "short_description"]]

    # If the results are greater than 5, pick only 5 in random order
    if len(result_df) > 5:
        selected_records = result_df.sample(n=5)
        return [f"({row['name']}) {row['short_description']}" for index, row in selected_records.iterrows()]
    
    # Else, display the results as it is
    else:
        return [f"({row['name']}) {row['short_description']}" for index, row in result_df.iterrows()]
    
def list_games(game_list: list[str], platform_result_response: str) -> None:
    """
    Helper function to display list of games from the game_list or display an alternate message
    
    Parameters
    ----------
        game_list: list[str]
            list containing games along with their short description
        platform_result_response: str
            Response from the game_search.csv which needs to be displayed

    Returns
    -------
        None
    """ 
    # Check if game list is empty or not
    if game_list:
        print(f"{constants.CHATBOT_NAME}: Alright! Here is a list of games {platform_result_response}:")
        for index, game in enumerate(game_list):
            print(f"\n{index+1}. {game}\n")
    else:
        # Print default message
        print(f"{constants.CHATBOT_NAME}: Hmmm, looks like we don't have any games matching your criteria.")  

def handle_game_search(captured_genre: str | None, captured_platform: str | None) -> None:
    """
    Helper function to handle logic when intent is game_search
    
    Parameters
    ----------
        captured_genre: str | None
            genre of the game if it is present in tokens
        captured_platform: str | None
            platform of the game if it is present in tokens

    Returns
    -------
        None
    """

    # Initialize genre result and platform result to None
    genre_result = platform_result = None

    # Initiate a loop if captured genre is None
    while not captured_genre:
        genre_input = input(f"{constants.USER}: ")
        genre_input_tokens = word_tokenize(genre_input)
        # Get genre result, genre result response, genre result type by executing get_similar_result
        genre_result, genre_result_response, genre_result_type = get_similar_result(
            tokens=genre_input_tokens,
            file_name=constants.GAME_SEARCH_PATH,
            query_header_name="keyword",
            result_header_name=["value", "response", "type"],
            threshold=0.6
        )
        
        # Check if genre_result is available
        if genre_result == "not_found" or "genre" not in genre_result_type:
            print(f"{constants.CHATBOT_NAME}: {GENRE_REPROMPT}")
            continue
        
        # Continue execution if platform is not captured before
        print(f"{constants.CHATBOT_NAME}: {genre_result_response}. {PLATFORM_MESSAGE if not captured_platform else ''}")
        break

    # Initiate a loop if platform is not captured
    while not captured_platform:
        platform_input = input(f"{constants.USER}: ")
        platform_input_tokens = word_tokenize(platform_input)

        # Get similar results
        platform_result, platform_result_response, platform_result_type = get_similar_result(
            tokens=platform_input_tokens,
            file_name=constants.GAME_SEARCH_PATH,
            query_header_name="keyword",
            result_header_name=["value", "response", "type"],
            threshold=0.6
        )
        
        # Check if platform is found
        if platform_result == "not_found" or "platform" not in platform_result_type:
            print(f"{constants.CHATBOT_NAME}: {PLATFORM_REPROMPT}")
            continue
        
        break
    
    # Execute search game function to get the game list
    game_list = search_game(genre_result if genre_result else captured_genre, platform_result if platform_result else captured_platform)

    # Display the game list if available
    list_games(game_list, platform_result_response if platform_result and platform_result != 'na' else '')

