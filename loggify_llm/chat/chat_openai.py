from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import json
import pandas as pd
import base64
import copy
import requests
from tqdm import tqdm
from IPython.display import Image, display
import time

from loggify_llm.mongodb import MongoDBLogger
from loggify_llm.chat.utils import unit_cost, unit_cost_batch_api

load_dotenv()


class ChatOpenAI:
    """
    A class to interact with OpenAI's language model for generating chat responses.

    Attributes:
        llm_model (str): The language model to use, default is "gpt-3.5-turbo-instruct".
        client (OpenAI): The OpenAI client initialized with the API key.

    Methods:
        request(prompt: str, question: str, temperature: float, top_p: float, max_tokens: int, **kwargs):
            Sends a request to the OpenAI API with the provided prompt and question, and returns the response.
    """

    def __init__(self, llm_model: str = "gpt-3.5-turbo-instruct", collection_name=None):
        """
        Initializes the ChatOpenAI class with the specified language model.

        Args:
            llm_model (str): The language model to use. Default is "gpt-3.5-turbo-instruct".
        """
        super().__init__()
        open_api_key = os.environ.get("OPENAI_API_KEY")
        assert open_api_key, "OPENAI_API_KEY environment variable not set"
        self.client = OpenAI(api_key=open_api_key)
        supported_llm_models = list(unit_cost.keys())
        assert (
            llm_model in supported_llm_models
        ), f"`llm_model` should be in {supported_llm_models}"
        self.llm_model = llm_model
        self.mongo_logger = MongoDBLogger(collection_name=collection_name)
        print(f"Initialize LLM model: {llm_model}")

    def request(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 1e-4,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        **kwargs,
    ):
        """
        Sends a request to the OpenAI API with the provided system prompt and user prompt.

        Args:
            system_prompt (str): The system prompt to set the context.
            user_prompt (str): The user's question to get a response for.
            temperature (float): Sampling temperature to control the randomness of the response. Default is 0.1.
            top_p (float): The cumulative probability of token selection. Default is 0.95.
            max_tokens (int): The maximum number of tokens to generate. Default is 1024.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            dict: A dictionary containing the request ID, output, completion tokens, prompt tokens, and total tokens.
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
        allow_request = self.mongo_logger.is_within_daily_quota()
        assert (
            allow_request
        ), f"👾 Error: You have exceeded the daily quota of ${self.mongo_logger.default_quota_by_day}."

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.llm_model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs,
        )
        raw_output = response.choices[0].message.content
        try:
            fine_output = json.loads(raw_output)
        except Exception as e:
            print(f"👾 Warning: Failed to refine the output of LLM because: {e}")
            fine_output = raw_output

        result = {
            "request_id": response.id,
            "llm_model": self.llm_model,
            "input": messages,
            "output": fine_output,
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
            "batch_api": False,
        }

        try:
            self.mongo_logger.insert_one(data=result)
        except Exception as e:
            print(f"👾 Warning: Failed to insert DB because: {e}")

        return result

    def batch_request(
        self,
        system_prompt: str,
        list_user_prompts: list,
        max_tokens: int = 1024,
        temperature: float = 1e-4,
    ):
        """
        Creates and submits a batch of chat completion requests to the API based on a provided system prompt
        and a list of user prompts. Each request is sent with the specified temperature setting.
        Ref:
            - https://cookbook.openai.com/examples/batch_processing
            - https://platform.openai.com/docs/guides/batch/rate-limits

        Parameters:
        - system_prompt (str): The initial prompt provided to the system, which sets the context for the conversation.
        - list_user_prompts (list): A list of user prompts, each of which will be processed as a separate task in the batch.
        - max_tokens (int): The maximum number of tokens to generate. Default is 1024.
        - temperature (float): The sampling temperature for the model. Lower values make the output more deterministic.
        Default is 1e-4.

        Returns:
        - batch_job (object): The created batch job object containing details about the batch processing status.
        """

        # Creating an array of JSON tasks
        self.mongo_logger.use_batch_api = True
        tasks = []

        for index, user_prompt in tqdm(enumerate(list_user_prompts)):
            task = {
                "custom_id": f"task-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    # This is what you would have in your Chat Completions API call
                    "model": self.llm_model,
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": max_tokens,
                },
            }

            tasks.append(task)
        assert len(tasks) < 50000, "A single batch may include up to 50,000 requests"
        batch_fname = "batch_tasks.jsonl"

        # Get the file size in bytes
        batch_fsize = os.path.getsize(batch_fname) / 1024**2

        assert batch_fsize < 100, "A batch input file can be up to 100 MB in size"

        with open(batch_fname, "w") as file:
            for obj in tasks:
                file.write(json.dumps(obj) + "\n")

        # Uploading batch file
        batch_file = self.client.files.create(file=open(batch_fname, "rb"), purpose="batch")

        # Creating the batch job
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h"
        )

        # Delete batch file
        # os.remove(batch_fname)
        # try:
        #     self.mongo_logger.insert_many(data=tasks)
        # except Exception as e:
        #     print(f"👾 Warning: Failed to insert DB because: {e}")

        return batch_fname, batch_job

    def batch_retrieve(self, batch_fname, batch_job, verbose: int = 60):
        """
        Monitors the progress of a batch job until it is complete, retrieves the results from the completed job,
        saves them to a file, and returns the parsed results as a list of dictionaries.

        Parameters:
        - batch_fname (str): Path to batch file jsonl
        - batch_job (object): The batch job object that needs to be monitored and retrieved.

        Returns:
        - results (list): A list of dictionaries containing the results of the batch job.
        """

        # Initialize an empty list to store the dictionaries
        list_input = []

        # Open the JSONL file and read line by line
        with open(batch_fname, "r") as file:
            for line in file:
                # Parse each line (which is a JSON object) into a dictionary
                json_object = json.loads(line.strip())
                # Append the dictionary to the list
                list_input.append(json_object)
        df_input = pd.DataFrame(list_input)

        complete_rate = 0
        while complete_rate < 1:
            batch_job = self.client.batches.retrieve(batch_job.id)
            completed = batch_job.request_counts.completed
            total = batch_job.request_counts.total
            try:
                assert total > 0, "Wait for creating batch job"
                complete_rate = completed / total
                if verbose > 0:
                    print(f"🔥 Batch Inference Complete Rate: {complete_rate*100:.2f}%")
                    time.sleep(verbose)  # Sleep for `verbose`` seconds before the next iteration
            except Exception as e:
                time.sleep(verbose)  # Sleep for `verbose`` seconds before the next iteration

        output_file_id = batch_job.output_file_id
        batch_results = self.client.files.content(output_file_id).content

        results_fname = "batch_results.jsonl"

        with open(results_fname, "wb") as file:
            file.write(batch_results)

        # Loading data from saved file
        results = []
        with open(results_fname, "r") as file:
            for line in file:
                # Parsing the JSON string into a dict and appending to the list of results
                json_object = json.loads(line.strip())
                custom_id = json_object["custom_id"]
                input_object = df_input[df_input["custom_id"] == custom_id].iloc[0]
                messages = input_object["body"]["messages"]
                response = DotDict(json_object).response.body
                raw_output = response.choices[0].message.content
                try:
                    fine_output = json.loads(raw_output)
                except Exception as e:
                    print(f"👾 Warning: Failed to refine the output of LLM because: {e}")
                    fine_output = raw_output

                result = {
                    "request_id": response.id,
                    "llm_model": self.llm_model,
                    "input": messages,
                    "output": fine_output,
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "batch_api": True,
                }
                results.append(result)
        try:
            insert_results = copy.deepcopy(results)
            self.mongo_logger.insert_many(data=insert_results)
        except Exception as e:
            print(f"👾 Warning: Failed to insert DB because: {e}")
        return results

    def batch_cancel(self, batch_job):
        response = self.client.batches.cancel(batch_job.id)
        print(response)


class ChatOpenAIVision:
    """
    A class to interact with OpenAI's vision-enhanced language model for generating responses based on text and images.

    Attributes:
        llm_model (str): The language model to use, default is "gpt-4-0125-preview".
        headers (dict): The headers for the OpenAI API requests.
        client (OpenAI): The OpenAI client initialized with the API key.

    Methods:
        encode_image(image_path):
            Encodes an image file to a base64 string.

        question_image(url, user_prompt, max_tokens=1024, temperature=1e-4):
            Sends a request to the OpenAI API with the provided prompt and image URL or file path, and returns the response.
    """

    def __init__(
        self,
        llm_model: str = "gpt-4-0125-preview",
    ):
        """
        Initializes the ChatVision class with the specified language model.

        Args:
            llm_model (str): The language model to use. Default is "gpt-4-0125-preview".
        """
        open_api_key = os.environ.get("OPENAI_API_KEY")
        assert open_api_key, "OPENAI_API_KEY environment variable not set"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {open_api_key}",
        }
        self.client = OpenAI(api_key=open_api_key)
        self.llm_model = llm_model
        self.mongo_logger = MongoDBLogger()

    def encode_image(self, image_path):
        """
        Encodes an image file to a base64 string.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The base64-encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def request(
        self,
        image_url: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 1e-4,
        show_preview: bool = False,
    ):
        """
        Sends a request to the OpenAI API with the provided prompt and image URL or file path, and returns the response.

        Args:
            image_url (str): The URL of the image or the local file path to the image.
            user_prompt (str): The prompt or question to ask the model.
            max_tokens (int): The maximum number of tokens to generate. Default is 1024.
            temperature (float): Sampling temperature to control the randomness of the response. Default is 1e-4.

        Returns:
            dict: A dictionary containing the request ID, output, completion tokens, prompt tokens, and total tokens.
        """

        if "http" in image_url:
            if show_preview:
                display(Image(url=image_url))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{user_prompt}"},
                        {
                            "type": "image_url",
                            "image_url": image_url,
                        },
                    ],
                }
            ]
            allow_request = self.mongo_logger.is_within_daily_quota()
            assert (
                allow_request
            ), f"👾 Error: You have exceeded the daily quota of ${self.mongo_logger.default_quota_by_day}."

            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=max_tokens,
            )
        else:
            if show_preview:
                display(Image(filename=image_url))
            base64_image = self.encode_image(image_url)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{user_prompt}?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ]
            payload = {
                "model": self.llm_model,
                "temperature": temperature,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            allow_request = self.mongo_logger.is_within_daily_quota()
            assert (
                allow_request
            ), f"👾 Error: You have exceeded the daily quota of ${self.mongo_logger.default_quota_by_day}."

            response = requests.post(
                url="https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json=payload,
            )
            response = response.json()
            response = DotDict(response)

        raw_output = response.choices[0].message.content
        try:
            fine_output = json.loads(raw_output)
        except Exception as e:
            print(f"👾 Warning: Failed to refine the output of LLM because: {e}")
            fine_output = raw_output

        result = {
            "request_id": response.id,
            "llm_model": self.llm_model,
            "input": messages,
            "output": fine_output,
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
            "batch_api": True,
        }

        try:
            self.mongo_logger.insert_one(data=result)
        except Exception as e:
            print(f"👾 Warning: Failed to insert DB because: {e}")

        return result


class DotDict(dict):
    """A dictionary with dot notation access."""

    def __getattr__(self, item):
        try:
            value = self[item]
            if isinstance(value, dict):
                return DotDict(value)
            if isinstance(value, list):
                return [DotDict(x) if isinstance(x, dict) else x for x in value]
            return value
        except KeyError:
            raise AttributeError(f"No such attribute: {item}")
