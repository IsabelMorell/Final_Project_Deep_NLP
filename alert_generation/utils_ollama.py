import math
import time
import ollama
import requests
import concurrent.futures
from tqdm import tqdm
import json
import os

from typing import List, Dict

messages: List[Dict[str, str]] = []


def get_available_models(api_url: str = "http://localhost:11434/api/tags") -> List[str]:
    """
    Returns list of available (downloaded) models
    
    Args:
        api_url (str): url where ollama's api is running
        
    Returns:
        List[str]: list with the names of the available models    
    """
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        return [model_data.get("name") for model_data in data.get("models", [])]

    except requests.exceptions.RequestException:
        log("ERROR fetching model data")
        return []


def load_model(model: str) -> bool:
    """
    Checks if a model is loaded and pulls it otherwise
    
    Args:
        model (str): model to load
        
    Returns:
        bool: indicates if the model was loaded or not
    """
    model_name = model if ':' in model else f"{model}:latest"

    if model_name in get_available_models():
        return True

    else:
        try:
            pull_response = ollama.pull(model_name, stream=True)

            # Initialize tqdm with a custom format
            progress_bar = tqdm(total=100, ncols=120, bar_format='{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}]')

            previous_percent = -1
            for progress in pull_response:
                # Check if 'completed' and 'total' are available in the response
                completed = progress.get('completed', None)
                total = progress.get('total', None)

                # If both values are available, update the progress bar and description
                if completed is not None and total is not None:
                    # Calculate the progress percentage
                    progress_percent = math.floor(completed / total * 100)
                    # Update progress bar
                    progress_bar.n = progress_percent
                    progress_bar.refresh()
                    progress_bar.set_description(f"Pulling {model_name} [{completed}/{total}]")

                # If 'completed' and 'total' are missing, keep the initial description
                else:
                    progress_bar.set_description(f"Pulling {model_name} (Initializing...)")

            progress_bar.close()

            log(f"Waiting for {model_name} to be fully loaded...")
            while True:
                # Fetch the updated model list
                if model in get_available_models():
                    log(f"{model_name} is now loaded.")
                    return True

                # Wait before polling again
                time.sleep(3)

        except Exception as e:
            log(f"ERROR while pulling {model_name}: {e}")
            return False


def delete_history() -> None:
    """Starts a new chat"""
    global messages
    messages = []


def prompt_model(model: str, prompt: str, path: str, timeout: int = 900) -> str:
    """
    Prompts the model, continuing the previous conversation. Requires running 'ollama serve' first
    
    Args:
        model (str): name of the model to converse with
        prompt (str): content to send the model
        path (str): path where the conversation is going to be saved at
        timeout (int): time to wait for the model's answer. Raises an Exception if the model doesn't
            respond within the specified timeout.
        
    Return:
        response (str): model's response to the prompt
    """
    messages.append(
        {
            'role': 'user',
            'content': prompt,
        },
    )

    def chat_with_model():
        return ollama.chat(model=model, messages=messages)['message']['content']

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(chat_with_model)
        try:
            response = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"The chat model took longer than {timeout} seconds to respond")

    messages.append(
        {
            'role': 'assistant',
            'content': response,
        }
    )

    save_conversation_to_json(model, messages, path)

    return response


def save_conversation_to_json(model: str, messages: List[Dict[str, str]], path: str):
    """
    Saves the conversation in a json with the timestamp
    
    Args:
        model (str): name of the model that was prompted
        messages (List[str]): prompt and response of the model
        path (str): path where to save the json    
    """
    try:
        # Name the file
        timestamp = int(time.time())
        model = model.replace(":","-")
        filename = os.path.join(f'{path}/conversations', f'{model}_{timestamp}.json')

        # Make the folder if it doesn't exist
        os.makedirs(f'{path}/conversations', exist_ok=True)

        # Save the conversation in a json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=4)

        print(f"Conversation saved to {filename}")
    except Exception as e:
        print(f"Error saving conversation to JSON: {e}")


logfile: str | None = None


def set_logfile(new_logfile: str):
    """
    Sets the path for the logfile. All logs will be appended to this file.
    
    Args:
        new_logfile (str): Path to the logfile.
    """
    global logfile
    logfile = new_logfile


def log(text: str, end: str = '\n'):
    """
    Logs the provided text to the logfile if set, otherwise prints to stdout.

    Args:
        text (str): The message to log.
        end (str): The end character (default is newline).
    """
    if logfile:
        with open(logfile, 'a+', encoding='utf-8') as f:
            f.write(text)
            f.write(end)
    else:
        try:
            print(text)
        except Exception:
            pass
