from utils.helpers import string_replacement_dict
from utils.helpers import get_similar_result

def process_user_query(tokens: list[str]) -> str:
    """
    Function to get the response from question_answer.csv
    
    Parameters
    ----------
        tokens: list[str]
            Tokens generated from pre-processing the user input

    Returns
    -------
        str:
            The response for the user query
    """

    # Using helper function to get most similar answer
    matched_answer = get_similar_result(
        tokens=tokens,
        file_name="data/question_answer.csv",
        query_header_name="question",
        result_header_name="answer"
    )

    # Check if the matched answer can be further processed
    for key in string_replacement_dict.keys():
        if key in matched_answer:
            matched_answer = matched_answer.replace(key, string_replacement_dict[key]())

    # Return the matched answer
    return matched_answer

