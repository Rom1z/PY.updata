import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize client with API key from environment variable
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def make_api_request_with_retry(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=150
            )
            return response

        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Rate limited. Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

        except openai.APIConnectionError as e:
            print(f"Connection error: {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)

        except openai.APIError as e:
            print(f"API error: {e}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)


try:
    response = make_api_request_with_retry(
        messages=[
            {"role": "user", "content": "Hello world"}
        ]
    )

    print("Response:", response.choices[0].message.content)

except Exception as e:
    print(f"Failed after retries: {e}")