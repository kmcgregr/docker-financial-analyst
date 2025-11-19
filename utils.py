"""
Utility Functions
"""

import os
import sys
import json
import urllib.request
from urllib.error import URLError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_model_availability(model_name: str):
    """
    Checks if a model is available in the local Ollama instance.
    
    Args:
        model_name (str): The name of the model to check.
    
    Raises:
        SystemExit: If the Ollama server is not reachable or the model is not found.
    """
    
    ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')
    
    try:
        # Construct the request to the Ollama API
        url = f"{ollama_base_url}/api/tags"
        req = urllib.request.Request(url)
        
        # Make the request and read the response
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                
                # Check if the model is in the list of available models
                for model in data['models']:
                    if model['name'] == model_name:
                        return
                
                # If the loop completes without finding the model
                print(f"Error: Model '{model_name}' not found in local Ollama instance.", file=sys.stderr)
                print(f"Please pull the model using 'ollama pull {model_name}'", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"Error: Failed to get model list from Ollama. Status: {response.status}", file=sys.stderr)
                sys.exit(1)
    
    except URLError as e:
        print(f"Error: Could not connect to Ollama at {ollama_base_url}", file=sys.stderr)
        print("Please ensure the Ollama server is running and accessible.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON response from Ollama.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
