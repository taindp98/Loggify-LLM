import os
import pymongo
from dotenv import load_dotenv
from urllib.parse import quote_plus
from datetime import datetime
import zlib
import copy
import json
import pandas as pd
from loggify_llm.chat.utils import unit_price, supported_batch_api_llm_models

load_dotenv()


class MongoDBLogger:
    def __init__(self, collection_name=None) -> None:
        """
        Connects to a MongoDB database using credentials and connection details stored in environment variables.

        Environment Variables:
            COLLECTION_NAME (str): The name of the collection to connect to.
            DB_NAME (str): The name of the database.
            CLUSTER_ADDRESS (str): The address of the MongoDB cluster.
            USRNAME (str): The username for authentication, which will be URL-encoded.
            PASSWD (str): The password for authentication, which will be URL-encoded.

        Returns:
            pymongo.collection.Collection: The MongoDB collection object.

        Prints:
            str: Success message upon successful connection to the database.
        """
        if collection_name:
            self.collection_name = collection_name
        else:
            self.collection_name = os.getenv("COLLECTION_NAME")

        db_name = os.getenv("DB_NAME")
        cluster_address = os.getenv("CLUSTER_ADDRESS")
        usrname = quote_plus(os.getenv("USRNAME"))
        passwd = quote_plus(os.getenv("PASSWD"))
        mongo_uri = (
            f"mongodb+srv://{usrname}:{passwd}@{cluster_address}/?retryWrites=true&w=majority"
        )
        client = pymongo.MongoClient(mongo_uri)
        self.db = client[db_name]
        self.collection = self.db[self.collection_name]
        self.default_quota_by_day = 10  ## $30/day
        print(
            f"🔥 Successfully Get Access Collection `{self.collection_name}` in Database `{db_name}`"
        )
        print(f"💰 Your default quota is ${self.default_quota_by_day} per day")
        self.keys_need_compress = ["input", "output"]

    def _compress_messages(self, messages):
        messages = json.dumps(messages)
        compressed_msg = zlib.compress(messages.encode("utf-8"))
        return compressed_msg

    def _decompress_messages(self, compressed_messages):
        decompressed_msg = zlib.decompress(compressed_messages).decode("utf-8")
        return decompressed_msg

    def insert_one(self, data: dict):
        """
        Inserts specified keys and a timestamp into a MongoDB collection.

        Args:
            collection (pymongo.collection.Collection): The MongoDB collection to insert data into.
            data (dict): A dictionary containing the data to be inserted. Must contain the keys 'request_id', 'completion_tokens', 'prompt_tokens', and 'total_tokens'.

        Returns:
            None

        Prints:
            str: Success message upon successful insertion.
            str: Failure message if insertion fails, along with the exception message.
        """
        insert_data = copy.copy(data)
        for k in list(insert_data.keys()):
            if k in self.keys_need_compress:
                insert_data[k] = self._compress_messages(insert_data[k])

        insert_data["time"] = datetime.utcnow()
        try:
            allow_insert = self._get_collection_status()
            if allow_insert:
                self.collection.insert_one(insert_data)
                print("🔥 Successfully Log Request to Database")
            else:
                print("👾 Fail To Log Request to Database because of Exceeding Storage Size")
        except Exception as e:
            print(f"👾 Fail To Log Request to Database because {e}")

    def insert_many(self, data: list):
        """
        Inserts a list of dictionaries into the MongoDB collection after adding a "time" key
        with the current UTC timestamp to each dictionary. The insertion only proceeds if
        the collection's storage size is within the allowed quota.

        Parameters:
        - data (list): A list of dictionaries representing the data to be inserted into the collection.

        Returns:
        - None: Prints a success message if the data is inserted successfully. If the insertion fails due
        to exceeding storage size or any other exception, an error message is printed.
        """

        insert_data = data
        for item in insert_data:
            for k in list(item.keys()):
                if k in self.keys_need_compress:
                    item[k] = self._compress_messages(item[k])
            item["time"] = datetime.utcnow()

        try:
            allow_insert = self._get_collection_status()
            if allow_insert:
                self.collection.insert_many(insert_data)
                print("🔥 Successfully logged request to database")
            else:
                print("👾 Failed to log request to database due to exceeding storage size")
        except Exception as e:
            print(f"👾 Failed to log request to database due to {e}")

    def query_collection(self, query: dict = {}):
        """
        Queries the MongoDB collection with the specified query.

        Args:
            query (dict): The MongoDB query.

        Returns:
            list: A list of documents that match the query.
        """
        raw_results = list(self.collection.find(query))
        fine_results = []
        for res in raw_results:
            for k in list(res.keys()):
                if k in self.keys_need_compress:
                    res[k] = self._decompress_messages(res[k])
            fine_results.append(res)
        return fine_results

    def estimate_cost(self, records: list, batch_api: bool = False):
        """
        Estimates the total cost of using various language models based on token usage.

        This function calculates the cost associated with using different language models
        (LLMs) by considering the number of prompt and completion tokens used, along with
        the associated cost per token for each model. The cost is calculated in dollars
        based on a pricing structure defined for each model.

        Args:
            records (list): A list of dictionaries where each dictionary contains information
                            about the usage of an LLM, including the model name (`llm_model`),
                            the number of prompt tokens used (`prompt_tokens`), and the number
                            of completion tokens generated (`completion_tokens`).
            batch_api (bool): Batch API costs a half price of normal API

        Returns:
            float: The total estimated cost in dollars for using the LLMs across all records.
        """

        # Get a list of unique LLM models used in the records
        supported_llm_models = list(unit_price.keys())

        # Initialize the total cost to zero
        total_cost = {item: 0 for item in supported_llm_models}

        if len(records) > 0:
            # Convert the records list into a DataFrame for easier processing
            df_total = pd.DataFrame(records)

            # Loop through each LLM model to calculate the cost
            for llm_model in list(total_cost.keys()):
                df_ = df_total[df_total["llm_model"] == llm_model]
                # Calculate the cost for prompt tokens
                prompt_tokens_cost = (
                    (df_["prompt_tokens"].sum() * unit_price[llm_model]["prompt_tokens"] * 1e-6)
                    if len(df_) > 0
                    else 0
                )

                # Calculate the cost for completion tokens
                completion_tokens_cost = (
                    (
                        df_["completion_tokens"].sum()
                        * unit_price[llm_model]["completion_tokens"]
                        * 1e-6
                    )
                    if len(df_) > 0
                    else 0
                )

                if llm_model in supported_batch_api_llm_models and batch_api:
                    prompt_tokens_cost = prompt_tokens_cost / 2
                    completion_tokens_cost = completion_tokens_cost / 2

                # Add the calculated costs to the total cost
                total_cost[llm_model] = prompt_tokens_cost + completion_tokens_cost
        # Return the total estimated cost in dollars
        return total_cost

    def _get_collection_status(self, quota: int = 256):
        """
        Checks the current storage usage of the database and compares it against the specified quota.
        If the storage usage exceeds the given threshold percentage of the quota, a warning message is displayed.

        Parameters:
        - quota (int): The maximum allowed storage size in MB. Default is 256 MB.

        Returns:
        - None: Prints a warning if the storage size exceeds the threshold.
        """
        dbstats = self.db.command("dbstats")
        storage_size = dbstats["storageSize"] / 1024**2
        if storage_size >= 0.85 * quota:
            print("👾 Warning: Storage size exceeds 60% of the free tier quota!")
            print("👾 Warning: Please contact your admin to back up your logs.")
            allow_insert = False
        else:
            allow_insert = True
        return allow_insert

    def is_within_daily_quota(self, date=None):
        """
        Checks if the usage for a specified date (or the current date) is within the allowed daily quota.

        Parameters:
        - date (datetime, optional): The date to check the usage for. If not provided, defaults to the current date with time set to 00:00:00.

        Returns:
        - bool: Returns True if the usage cost is within the daily quota, otherwise returns False.
        """

        if not date:
            date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        records = self.query_collection(query={"time": {"$gte": date}})
        costs = sum(list(self.estimate_cost(records=records, batch_api=False).values()))
        allow_request = costs < self.default_quota_by_day
        if not allow_request:
            print(f"📈 {date} costs ${costs}")
        return allow_request
