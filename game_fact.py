from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from utils.preference import get_preference, update_preferences
from utils import constants
from nltk.tokenize import word_tokenize

def get_game_data(game_name: str, column_name: str) -> str:
    """
    Helper function to get the corresponding game data present in the given column
    
    Parameters
    ----------
        game_name: str
            Name of the game
        column_name: str
            Column name from which the data is picked

    Returns
    -------
        str
            Actual value from the column
    """ 
    # Read the csv into the data frame
    df = pd.read_csv("data/games.csv")

    # Get the row which contains the game name
    matched_row = df.loc[df["name"].str.contains(game_name, case=False, na=False)]
    try:
        # Try to get the column data corresponding to the matched row
        data_to_return = matched_row[column_name].values[0]
    except Exception:
        # If the above fails, return a default message
        data_to_return = "Sorry, I Couldn't find the data which you were looking for"
    
    # Return the data extracted from cell
    return data_to_return

def get_intent_based_game_data(game_name: str, intent: str) -> str:
    """
    Helper function to get the game data based on intent and convert it to a chatbot response.
    
    Parameters
    ----------
        game_name: str
            Name of the game
        intent: str
            The derived intent        

    Returns
    -------
        str
            string that can be used as a chatbot response
    """ 
    match intent:
        case "game_fact":
            data = get_game_data(game_name, "description")
            return f"({game_name}) {data}"
        case "game_genre_fact":
            data = get_game_data(game_name, "genre")
            return f"({game_name}) This game belongs to {data} genre."
        case "game_platform_fact":
            data = get_game_data(game_name, "platform")
            return f"({game_name}) The game is available on {data}."
        case _:
            return "I am sorry, but I am not able to recognize the intent of the statement. Please try rephrasing your request?"


def is_game_in_input(tokens: list[str]) -> bool:
    """
    Helper function to check if there is a game mentioned in the input
    
    A not so accurate method to check if the game is present in the
    user input or not. But it does it's job most of the time.
    
    Parameters
    ----------
        tokens: list[str]
            user input in the form of tokens

    Returns
    -------
        bool
            True if game is present else False
    """ 

    # Custom stop words to eliminate all possible occurences of tokens so that the tokens outside this list can be considered as tokens of games
    CUSTOM_SW = [
        "i", "information", "game", "opinion", "review", "interest", "read", "type",
        "provide", "description", "detail", "insight", "fact", "tell", "give", "how",
        "more", "info", "about", "learn", "understand", "explain", "platform", 
        "genre", "available", "compatible", "play", "show", "platforms", "support", "run",
        "?", ".", "!"
    ]

    # Remove stop words and custom stop words
    tokens = [word.lower() for word in tokens if word.lower() not in stopwords.words("english") and word.lower() not in CUSTOM_SW]

    # Return true if list is not empty else False
    return True if tokens else False


def extract_game(tokens: list[str]) -> tuple[float | str]:
    """
    Helper function to extract game from user input tokens
    
    Parameters
    ----------
        tokens: list[str]
            user input in the form of tokens

    Returns
    -------
        tuple(float | str)
            a tuple having the similarity score and the matched result
    """ 
    # Remove stop words
    tokens = [word.lower() for word in tokens if word not in stopwords.words("english")]

    # Join tokens into a single string
    user_input_str = ' '.join(tokens)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Load the file 
    df = pd.read_csv("data/games.csv")

    # Perform fit and transform on the column name which needs to be queried
    values_tfidf = vectorizer.fit_transform(df["name"])

    # Transform user input
    user_input_tfidf = vectorizer.transform([user_input_str])

    # Calculate cosine similarity between user input and values
    similarities = cosine_similarity(user_input_tfidf, values_tfidf)

    # Find the index of the most similar result
    most_similar_index = similarities.argmax()

    # Get the matched result
    matched_result = df.loc[most_similar_index, "name"]

    return (similarities[0, most_similar_index], matched_result)

def handle_intermediate_confidence(result: str) -> bool:
    """
    Helper function to handle the logic if the confidence falls between 45% - 79%
    
    Parameters
    ----------
        result: str
            The result which we want to verify

    Returns
    -------
        bool
            if the result is a correct guess or not
    """
    # Initialize the variable to break the outer loop
    correct_guess = False

    # Initiate an infinite loop
    while True:
        # Ask the user if it is the correct guess
        print(f"{constants.CHATBOT_NAME}: Did you mean '{result}'? (yes/no)")

        # Get the confirmation
        confirmation_input = input(f"{constants.USER}: ")

        # If the user says 'yes'
        if confirmation_input.strip().lower() == "yes":

            # Set the correct guess to True
            correct_guess = True

            # Break the loop
            break

        # If the user says "no"
        elif confirmation_input.strip().lower() == "no":

            # Break the loop
            break

        # If the user types anything other than yes/no
        else:
            # Print the message and continue the loop
            print(f"{constants.CHATBOT_NAME}: I'm sorry, I didn't understand that.") 

    return correct_guess 

def handle_game_in_context(intent: str) -> None:
    """
    Helper function to store the game in context (preference)
    
    Parameters
    ----------
        intent: str
            The recognized intent of the input

    Returns
    -------
        None
    """ 
    print(f"{constants.CHATBOT_NAME}: Could you please tell me the name of the game you are looking for?")

    # Initiate an infinite loop
    while True:
        # Get the user input
        user_input = input(f"{constants.USER}: ")

        # Tokenize the input
        tokens = word_tokenize(user_input)

        # Check if the game matches any games by using the extract_game helper function
        similarity_score, matched_game = extract_game(tokens=tokens)

        # If the similarity score is less than 45%
        if similarity_score < 0.45:
            # If the game is present in the input
            if is_game_in_input(tokens=tokens):
                print(f"{constants.CHATBOT_NAME}: I'm sorry, we don't have the game which you are looking for? Can you please provide a different game name?")
            # Reprompt
            else:
                print(f"{constants.CHATBOT_NAME}: I'm sorry, I didn't get that. Could you please provide me the name of the game again?")
        
        # If the similarity score is between 45% to 79%
        elif similarity_score >= 0.45 and similarity_score < 0.80:
            # Invoke the helper function to handle this logic
            correct_guess = handle_intermediate_confidence(matched_game)

            # Check if it was the right guess
            if correct_guess:

                # Update the preference to reflect the new game in context
                update_preferences("game_in_context", matched_game)

                # Implicitly confirm and show the results
                print(f"{constants.CHATBOT_NAME}: Thanks for the confirmation. Here are the details you requested for.")

                # Print the details as requested
                print(f"{constants.CHATBOT_NAME}: {get_intent_based_game_data(matched_game, intent)}")

                # Break the loop
                break

            else:
                # Ask the user to provide the name of the game again
                print(f"{constants.CHATBOT_NAME}: Ok, could you please tell me the name of the game again?") 

        # If the similarity score is more than 80%        
        else:
            # Update the preference to reflect the new game in context
            update_preferences("game_in_context", matched_game)

            # Print the details as requested
            print(f"{constants.CHATBOT_NAME}: {get_intent_based_game_data(matched_game, intent)}")

            # Break the loop
            break

def handle_game_fact(tokens: list[str], intent: str) -> None:
    """
    Helper function to handle logic when intent is game_fact
    
    Parameters
    ----------
        tokens: list[str]
            user input in the form of tokens
        intent: str
            The intent recognized

    Returns
    -------
        None
    """
    # Assume the repromt to be true to enter the loop
    reprompt = True

    # Initiate a loop if reprompt is true
    while reprompt:

        # Immediately set the reprompt to false so that it can be set true only if we want to repromt the user
        reprompt = False

        # Get the similarity score and matched game by calling the helper function
        similarity_score, matched_game = extract_game(tokens=tokens)

        # If the similarity score is less than 45%
        if similarity_score < 0.45:
            # Get the game in context
            game_in_context = get_preference("game_in_context")

            # If the game in context is empty
            if game_in_context == "":
                # Handle the logic to store the game in context and display the results accordingly
                handle_game_in_context(intent=intent)
            
            # If the game in context is not empty
            # This can either mean that the user is asking something about the game in context or he might be asking about the game which is not present in the games.csv

            # Check if there is a game name mentioned in the input
            elif is_game_in_input(tokens):
                print(f"{constants.CHATBOT_NAME}: I am sorry! We currently don't have the game you're looking for. Could you please specify another game?")

            # If the game in context is not empty, the user is asking about the game in context
            elif game_in_context != "":
                # Print the requested details
                print(f"{constants.CHATBOT_NAME}: {get_intent_based_game_data(game_in_context, intent)}")
            
            # If it's neither of the case, handle the error
            else:
                # Print a default message and move on
                print(f"{constants.CHATBOT_NAME}: Ugghh! Looks like I am having difficulty processing your request. Could you please rephrase your request again?")
                    
        # if the similarity score falls between 45% to 79%          
        elif similarity_score >= 0.45 and similarity_score < 0.80:
            # Execute the helper function to handle the logic
            correct_guess = handle_intermediate_confidence(matched_game)

            # If the guess is not right
            if not correct_guess:
                # Set the reprompt to true to continue the loop
                reprompt = True
                # Reprompt the user
                print(f"{constants.CHATBOT_NAME}: Ok, could you please tell me the name of the game again?")
                # Get the user input
                reprompt_input = input(f"{constants.USER}: ")
                # Preprocess the userinput
                tokens = word_tokenize(reprompt_input)
                # Continue to execute the loop without moving to the next step
                continue

            # If the guess is right
            # Update the prefererence
            update_preferences("game_in_context", matched_game)
            # Implicitly confirm 
            print(f"{constants.CHATBOT_NAME}: Thanks for the confirmation. Here are the details you requested for.")
            # Print the requested details
            print(f"{constants.CHATBOT_NAME}: {get_intent_based_game_data(matched_game, intent)}")

        # If the similarity is more than 80%
        else:
            # Update the preference to store the game in context
            update_preferences("game_in_context", matched_game)

            # print the requested response
            print(f"{constants.CHATBOT_NAME}: {get_intent_based_game_data(matched_game, intent)}")
