import os
import dotenv
from _langchain.minimal_KB import LangchainKnowledgeBase
from _haystack.minimal_KB import HaystackKnowledgeBase

dotenv.load_dotenv()


def langchain(question: str):
    kb = LangchainKnowledgeBase(
        docs_dir="docs/",
        azure_api_key=os.getenv("AZURE_API_KEY"),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        endpoint=os.getenv("ENDPOINT"),
    )
    kb.run(query=question, output_file="_langchain/output.txt")


def haystack(question: str):
    kb = HaystackKnowledgeBase(
        docs_dir="docs/",
        azure_api_key=os.getenv("AZURE_API_KEY"),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        endpoint=os.getenv("ENDPOINT"),
    )
    kb.run(query=question, output_file="_haystack/output.txt")


if __name__ == "__main__":
    query = "Give me detailed information about sulfuric acid based on the documents provided."

    # langchain(query)
    haystack(query)
