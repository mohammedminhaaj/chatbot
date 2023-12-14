from utils import constants
from process import chatbot_response
from identity_management import handle_capture_username
from utils.preference import clear_preferences

"""
Before executing this program, please download the following packages
by doing nltk.download(<package_name>)

1. punkt
2. stopwords
3. wordnet
4. averaged_perceptron_tagger
5. universal_tagset
6. maxent_ne_chunker
7. words
"""

def main():
    """
    The main function which will be executed when the program runs
    
    Parameters
    ----------
        None

    Returns
    -------
        None
    """

    # Welcome user and capture user's username
    print(f"{constants.CHATBOT_NAME}: Welcome to {constants.STORE_NAME}! My name is {constants.CHATBOT_NAME}")
    print(f"{constants.CHATBOT_NAME}: May I know your name?")

    #Initiate a loop to capture username
    handle_capture_username()
    
    # Initiating IO loop
    while True:

        # Getting the user input
        user_input = input(f"{constants.USER}: ")

        # Getting the chatbot response
        response = chatbot_response(user_input=user_input)

        # Check if user wants to quit chatting
        if response.intent == "goodbye":
            print(f"{constants.CHATBOT_NAME}: {response}")
            # Break the loop if the user quits
            break
        
        # Continue to print the response from chatbot
        print(f"{constants.CHATBOT_NAME}: {response}")
        
        # If chatbot response has additional function to execute
        if response.function_to_execute:
            response.function_to_execute()

if __name__ == "__main__":
    # Clearing all the preferences before starting the program
    clear_preferences()
    # Start execution from here
    main()