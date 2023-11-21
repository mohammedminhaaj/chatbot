import json
from utils import constants

def load_preferences() -> dict[str,str]:
    """
    Function to read preferences.json and return the data in dictionary form
    
    Parameters
    ----------
        None

    Returns
    -------
        dict[str,str]:
            A dictionary containing preferences 
    """

    with open(constants.PREFERENCE_PATH, "r") as f:
        data = json.load(f)
    return data

def get_preference(key: str) -> str | None:
    """
    Function to get the preference if available 
    
    Parameters
    ----------
        key: str
            The key that needs to be fetched

    Returns
    -------
        str | None
            The value of the given key from preferences.json 
    """
    preferences = load_preferences()
    return preferences.get(key)   

def update_preferences(key: str, value: str) -> None:
    """
    Function to update the preference if available or create a new one 
    
    Parameters
    ----------
        key: str
            The key that needs to be updated or created
        value: str
            The value corresponding to the key that will be inserted

    Returns
    -------
        None 
    """
    preferences = load_preferences()
    preferences[key] = value

    with open(constants.PREFERENCE_PATH, "w") as f:
        json.dump(preferences, f)

def clear_preferences() -> None:
    """
    Function to clear preferences. This is done when the user exits the program
    
    Parameters
    ----------
        None

    Returns
    -------
        None 
    """
    preferences = load_preferences()
    for key in preferences.keys():
        preferences[key] = ""

    with open(constants.PREFERENCE_PATH, "w") as f:
        json.dump(preferences, f)
    
def clear_preferences_except_username() -> None:
    """
    Function to clear preferences except username.
    
    Parameters
    ----------
        None

    Returns
    -------
        None 
    """
    preferences = load_preferences()
    for key in preferences.keys():
        if key != "username":
            preferences[key] = ""

    with open(constants.PREFERENCE_PATH, "w") as f:
        json.dump(preferences, f)
