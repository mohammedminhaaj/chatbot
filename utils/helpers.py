from utils import constants
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from utils.preference import get_preference
from typing import Any

string_replacement_dict: dict[str, Any] = {
    "@username": lambda : get_preference("username"),
    "@game_in_context": lambda: get_preference("game_in_context")
}

def get_similar_result(tokens: list[str], file_name: str, query_header_name: str, result_header_name: str | list[str], threshold: float | None = constants.DEFAULT_SIMILARITY_THRESHOLD) -> str | tuple[str]:
    """
    Function to get similar result by using TF-IDF vectorizer and cosine similarity
    
    Parameters
    ----------
        tokens: list[str]
            Tokens generated from pre-processing the user input
        file_name: str
            Name of the file which will be searched for similarity
        query_header_name: str
            Name of column in the file which will be used to match the user input
        result_header_name: str | list[str]
            Name of column or list of columns in the file where the results will be stored
        threshold: float | None
            similarity threshold to be used. When nothing is specified, the threshold defaults to 0.7


    Returns
    -------
        str | tuple[str]:
            Most similar result or tuple of results from the file
    """
    # Join tokens into a single string
    user_input_str = ' '.join(tokens)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Load the file 
    df = pd.read_csv(file_name)

    # Perform fit and transform on the column name which needs to be queried
    values_tfidf = vectorizer.fit_transform(df[query_header_name])

    # Transform user input
    user_input_tfidf = vectorizer.transform([user_input_str])

    # Calculate cosine similarity between user input and values
    similarities = cosine_similarity(user_input_tfidf, values_tfidf)

    # Find the index of the most similar result
    most_similar_index = similarities.argmax()

    # Get the similar index after calculating threshold
    similar_index_after_threshold = most_similar_index if similarities[0, most_similar_index] >= threshold else 0

    # Get the corresponding intent by checking the threshold
    # If the result header name is a string
    if isinstance(result_header_name, str):
        matched_result = df.loc[
            similar_index_after_threshold, 
            result_header_name
        ]
    # If the result header name is a list
    elif isinstance(result_header_name, list):
        matched_result = (
            df.loc[
                similar_index_after_threshold, 
                header_name
            ]
            for header_name in result_header_name
        )
    
    else:
        raise ValueError("Type of result_header_name not supported")

    # Return the matched result
    return matched_result

