from openai import OpenAI

client = OpenAI()

class LLM:
    def invoke(self, messages, tools=None):
        response = client.chat.completions.create(
            model="gpt-4o",   # or gpt-4.1 / gpt-3.5
            messages=messages,
            tools=tools if tools else None
        )
        return response.choices[0].message

llm = LLM()

# ========================
#  HELPER: Safe Tool Output
# ========================
def safe_tool_output(text, max_words=1500):
    """
    Prevents Pinecone/Postgres results from blowing up context size.
    - If text is too long, truncate it.
    - You can later swap this with an LLM-based summarizer if you want.
    """
    words = text.split()
    if len(words) > max_words:
        # Keep only first + last chunk so context stays relevant
        head = " ".join(words[:750])
        tail = " ".join(words[-200:])
        return f"[Truncated Output]\n{head}\n...\n{tail}"
    return text