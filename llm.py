import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SYSTEM_PROMPT = """You are Droid, a helpful and careful coding assistant.

Help the user understand, write, debug, and improve software. Give accurate,
practical answers and explain important trade-offs briefly. When writing code,
provide complete, runnable examples when appropriate, use clear names, and
follow the language and conventions of the user's project. Do not invent files,
APIs, test results, or actions you have not performed. Ask a concise clarifying
question when essential requirements are missing; otherwise state reasonable
assumptions and proceed. Point out security, data-loss, or compatibility risks
before recommending risky changes. Keep responses focused on the user's request.
"""


def main() -> None:
    """Run a simple streaming chat session with the configured Groq model."""
    model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    client = Groq()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Agent conversation ({model}). Type /exit or /quit to end.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit", "/bye"}:
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        print("Agent: ", end="", flush=True)
        answer_parts = []

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.6,
                max_completion_tokens=2048,
                top_p=0.95,
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    answer_parts.append(content)
            print("\n")
        except Exception as error:
            # Remove the unanswered turn so a later request has a valid history.
            messages.pop()
            print(f"\nRequest failed: {error}\n")
            continue

        messages.append({"role": "assistant", "content": "".join(answer_parts)})


if __name__ == "__main__":
    main()
