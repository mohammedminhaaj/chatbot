import pandas as pd
import random
from utils.helpers import get_similar_result

def get_intent(tokens: list[str]) -> str:
    """
    Function to get the intent by using the input text tokens
    
    Parameters
    ----------
        tokens: list[str]
            Tokens generated from pre-processing the user input

    Returns
    -------
        str:
            The intent from the user input
    """

    # Using helper function to get the similar intent
    return get_similar_result(
        tokens=tokens,
        file_name="data/intent.csv",
        query_header_name="value",
        result_header_name="intent"
    )

def get_intent_response(intent: str) -> str:
    """
    Function to get the intent response for a specific intent from the intent_response.csv
    
    Parameters
    ----------
        intent: str
            The intent for which the response is needed

    Returns
    -------
        str:
            The response for the given intent
    """

    # Load the intent_response.csv 
    df = pd.read_csv('data/intent_response.csv')
    
    # Get only the responses for the given intent
    intent_response_df = df[df['intent'] == intent]

    # Check if the given intent has any responses
    if not intent_response_df.empty:
        # Return a random response from the list of responses
        return random.choice(intent_response_df['response'].tolist())
    else:
        return "Apologies, I won't be able to answer this as it is out of my scope"
    


