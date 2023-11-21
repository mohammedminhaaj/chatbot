from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

# A dictionary for defining mappings for pos_tag tags
posmap: dict[str,str] = {
    "ADJ": "a",
    "ADV": "r",
    "NOUN": "n",
    "VERB": "v"
}

def preprocess_text(text: str) -> list[str]:
    """
    Function to pre-process the text and return tokenized text

    Parameters
    ----------
    text : str
        text that needs to be pre-processed

    Returns
    -------
    list[str]
        List of lemmatized tokens
    """
    # Tokenization
    tokens = word_tokenize(text)

    # # Removing stop words
    # tokens_without_sw = [word for word in tokens if word.lower() not in stopwords.words("english")]

    # POS tagging with pos_tag
    tagged_tokens = pos_tag(tokens, tagset='universal')
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    # Getting lemmatized tokens
    lemmatized_tokens = [lemmatizer.lemmatize(word, posmap.get(tag, "n")) for word, tag in tagged_tokens]

    # Return lemmatized tokens
    return lemmatized_tokens