from nltk import word_tokenize, pos_tag, ne_chunk
from utils.preference import update_preferences

def capture_user_name(user_name_input: str) -> str | None:
    """
    Function to capture user's last name from the input

    Parameters
    ----------
    user_name_input : str
        input text from the user

    Returns
    -------
    str | None
        return the last extracted name from the input if found
    """

    # Tokenize the text
    tokens = word_tokenize(user_name_input)

    # Tag the tokens with their part-of-speech (POS) tags
    pos_tags = pos_tag(tokens)

    # Chunk the POS-tagged tokens into named entities (NER)
    chunks = ne_chunk(pos_tags)

    # Initialize empty list
    extracted_names = []

    # Identify the user's name
    for chunk in chunks:
        # Check if the chunk is a class and has a label Person, Geo-Political Entity or Organization
        if not isinstance(chunk, tuple) and (chunk.label() in ["PERSON", "GPE", "ORGANIZATION"]):
            extracted_names.extend([name for name, pos in chunk.leaves()])
        # Else, check if the chunck is a tuple and has a POS proper noun
        elif chunk[1] == "NNP":
            extracted_names.append(chunk[0])

    # If extracted_names list is not empty
    if extracted_names:
        # Return the last name
        update_preferences("username", extracted_names[-1])
        return extracted_names[-1]
    # Return None if the extracted_names list is empty
    else:
        return None 
