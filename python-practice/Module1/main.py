print("script started")

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import (
    retrieve, 
    pprint, 
    generate_with_single_input_local, 
    read_dataframe, 
    display_widget
)
import unittests


NEWS_DATA = read_dataframe("news_data_dedup.csv")
pprint(NEWS_DATA[9:11])

def query_news(indices):
    """
    Retrieves elements from a dataset based on specified indices.

    Parameters:
    indices (list of int): A list containing the indices of the desired elements in the dataset.
    
    Returns:
    list: A list of elements from the dataset corresponding to the indices provided in list_of_indices.
    """
     
    output = [NEWS_DATA[index] for index in indices]

    return output


def get_relevant_data(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve and return the top relevant data items based on a given query.

    This function performs the following steps:
    1. Retrieves the indices of the top 'k' relevant items from a dataset based on the provided `query`.
    2. Fetches the corresponding data for these indices from the dataset.

    Parameters:
    - query (str): The search query string used to find relevant items.
    - top_k (int, optional): The number of top items to retrieve. Default is 5.

    Returns:
    - list[dict]: A list of dictionaries containing the data associated 
      with the top relevant items.

    """
    ### START CODE HERE ###

    # Retrieve the indices of the top_k relevant items given the query
    relevant_indices = None
    
    relevant_indices = retrieve(query, top_k)

    # Obtain the data related to the items using the indices from the previous step
    relevant_data = None
    relevant_data = query_news(relevant_indices)

    ### END CODE HERE
    
    return relevant_data


pprint("relevant_data")
query = "Greatest storms in the US"
relevant_data = get_relevant_data(query, top_k = 1)
pprint(relevant_data)
print("relevant_data")


unittests.test_get_relevant_data(get_relevant_data)

def format_relevant_data(relevant_data):
    """
    Formats a list of relevant documents into a structured string for use in a RAG system.

    Parameters:
    relevant_data (list): A list with relevant data.

    Returns:
    str: A formatted string with the relevant documents, structured for use in a Retrieval-Augmented Generation (RAG) system."
    """

    ### START CODE HERE ###

    # Create a list to store the formatted documents
    formatted_documents = []
    
    # Iterates over each relevant document.
    for document in relevant_data:

        # Formats each document into a structured layout string. Remember that each document is in one different line. So you should add a new line character after each document added.
        formatted_document = f"Title: {document['title']}, Description: {document['description']}, Published at: {document['published_at']},URL: {document['url']}"
        
        # Append the formatted document string to the formatted_documents list
        formatted_documents.append(formatted_document)
        "\n".join(formatted_documents)
    
    ### END CODE HERE ###
    
    # Returns the final augmented prompt string.

    return "\n".join(formatted_documents)

example_data = NEWS_DATA[4:8]
print(format_relevant_data(example_data))

unittests.test_format_relevant_data(format_relevant_data)

def generate_final_prompt(query, top_k=5, use_rag=True, prompt=None):
    """
    Generates a final prompt based on a user query, optionally incorporating relevant data using retrieval-augmented generation (RAG).

    Args:
        query (str): The user query for which the prompt is to be generated.
        top_k (int, optional): The number of top relevant data pieces to retrieve and incorporate. Default is 5.
        use_rag (bool, optional): A flag indicating whether to use retrieval-augmented generation (RAG)
                                  by including relevant data in the prompt. Default is True.
        prompt (str, optional): A template string for the prompt. It can contain placeholders {query} and {documents}
                                for formatting with the query and formatted relevant data, respectively.

    Returns:
        str: The generated prompt, either consisting solely of the query or expanded with relevant data
             formatted for additional context.
    """
    # If RAG is not being used, format the prompt with just the query or return the query directly
    if not use_rag:
        return query

    # Retrieve the top_k relevant data pieces based on the query
    relevant_data = get_relevant_data(query, top_k=top_k)

    # Format the retrieved relevant data
    retrieve_data_formatted = format_relevant_data(relevant_data)

    # If no custom prompt is provided, use the default prompt template
    if prompt is None:
        prompt = (
            f"Answer the user query below. There will be provided additional information for you to compose your answer. "
            f"The relevant information provided is from 2024 and it should be added as your overall knowledge to answer the query, "
            f"you should not rely only on this information to answer the query, but add it to your overall knowledge."
            f"Query: {query}\n"
            f"2024 News: {retrieve_data_formatted}"
        )
    else:
        # If a custom prompt is provided, format it with the query and formatted relevant data
        prompt = prompt.format(query=query, documents=retrieve_data_formatted)

    return prompt

print(generate_final_prompt("Tell me about the US GDP in the past 3 years."))

def llm_call(query, top_k = 5, use_rag = True, prompt = None):
    """
    Calls the LLM to generate a response based on a query, optionally using retrieval-augmented generation.

    Args:
        query (str): The user query that will be processed by the language model.
        use_rag (bool, optional): A flag that indicates whether to use retrieval-augmented generation by 
                                  incorporating relevant documents into the prompt. Default is True.

    Returns:
        str: The content of the response generated by the language model.
    """
    

    # Get the prompt with the query + relevant documents
    prompt = generate_final_prompt(query, top_k, use_rag, prompt)

    # Call the LLM
    generated_response = generate_with_single_input_local(prompt)

    # Get the content
    generated_message = generated_response['content']
    
    return generated_message

query = "Tell me about the US GDP in the past 3 years."

print(llm_call(query, use_rag = True))


print(llm_call(query, use_rag = False))
