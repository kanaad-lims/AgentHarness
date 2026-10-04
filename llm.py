import json
import os

from dotenv import load_dotenv
from groq import Groq

from tools.bash_tool import BASH_TOOL_SCHEMA, run_bash
from tools.browser_tool import run_browser

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
TOOL_SCHEMAS = [BASH_TOOL_SCHEMA]
TOOLS = {"bash": run_bash}

SYSTEM_PROMPT = """You are a helpful and careful coding assistant.

Help the user understand, write, debug, and improve software. Give accurate,
practical answers and explain important trade-offs briefly. When writing code,
provide complete, runnable examples when appropriate, use clear names, and
follow the language and conventions of the user's project. Do not invent files,
APIs, test results, or actions you have not performed. Ask a concise clarifying
question when essential requirements are missing; otherwise state reasonable
assumptions and proceed. Point out security, data-loss, or compatibility risks
before recommending risky changes. Keep responses focused on the user's request.
Use the bash tool for inspecting the files.
RULE: NEVER execute the bash tool if you are asked to delete or modify any file inside any directory.
"""


def call_llm(messages, client, tool_choice="auto"):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice=tool_choice,
        max_completion_tokens=2048,
    )
    return response.choices[0].message


def main():
    client = None
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"Agent ({MODEL}). Type /bye to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"/bye", "/exit", "/quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        if user_input == "/browser" or user_input.startswith("/browser "):
            browser_command = user_input[len("/browser"):].strip()
            print(f"\n{run_browser(browser_command)}\n")
            continue

        if client is None:
            client = Groq()

        messages.append({"role": "user", "content": user_input})

        try:
            for _ in range(10):  # Prevent unbounded tool-call loops.
                message = call_llm(messages, client)
                tool_calls = message.tool_calls or []

                if not tool_calls:
                    answer = message.content or ""
                    print(f"\nAgent: {answer}\n")
                    messages.append({"role": "assistant", "content": answer})
                    break

                # Preserve the assistant's tool-call request in the history.
                messages.append(message.model_dump(exclude_none=True))

                for tool_call in tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    if name not in TOOLS:
                        result = f"Error: unknown tool '{name}'."
                    elif name == "bash":
                        command = args.get("command", "")
                        print(f"\nBash command requested:\n{command}")
                        approved = input("Run this command? [y/N]: ").strip().lower()

                        if approved in {"y", "yes"}:
                            result = TOOLS[name](**args)
                            # removed printing raw bash output. Directly passed to the llm as context.
                        else:
                            result = "Command was not run; the user did not approve it."

                    else:
                        result = TOOLS[name](**args)

                    # Return tool output to the model for its follow-up answer.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": name,
                            "content": result,
                        }
                    )
            else:
                print("Agent: Stopped after reaching the tool-call limit.\n")

        except Exception as error:
            print(f"\nRequest failed: {error}\n")


if __name__ == "__main__":
    main()
