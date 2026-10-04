from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "What is the current price of the Tesla Model Y?"
        }
    ]
)

print(response.choices[0].message.content)