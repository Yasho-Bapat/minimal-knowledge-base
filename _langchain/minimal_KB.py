import os
from time import perf_counter
import dotenv

from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()


class LangchainKnowledgeBase:
    def __init__(self, docs_dir, azure_api_key, cohere_api_key, endpoint):
        self.cohere_api_key = cohere_api_key
        self.docs_dir = docs_dir
        self.endpoint = endpoint
        self.azure_api_key = azure_api_key
        self.embedding_function = CohereEmbeddings()
        self.llm = ChatCohere(model="command-r")
        self.documents = []

    def load_documents(self):
        file_paths = [self.docs_dir + file for file in os.listdir(self.docs_dir)]
        for file_path in file_paths:
            loader = AzureAIDocumentIntelligenceLoader(
                api_endpoint=self.endpoint,
                api_key=self.azure_api_key,
                file_path=file_path,
                api_model="prebuilt-layout",
            )
            self.documents.extend(loader.load())

    def preprocess_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=50, separators=[" ", "\n", "\t"]
        )
        self.docs = text_splitter.split_documents(self.documents)

    def embed_documents(self):
        self.db = Chroma.from_documents(self.docs, self.embedding_function)

    def query_documents(self, query, k=10):
        docs = self.db.similarity_search(query, k)
        result = self.llm.invoke(
            f"You are an expert Material Safety Document Analyser."
            + f"Context: {[doc.page_content for doc in docs]} "
            + "using only this context, answer the following question: "
            + f"Question: {query}. Make sure there are full stops after every sentence."
        )
        return result.content

    def save_result(self, result, filename, elapsed_time):
        with open(filename, "w") as file:
            file.write(result + "\n\n" + f"Langchain took: {elapsed_time} seconds")

    def run(self, query, output_file):
        start = perf_counter()
        self.load_documents()
        self.preprocess_documents()
        self.embed_documents()
        preprocessed_time = perf_counter() - start
        result = self.query_documents(query)
        retrieval_time = perf_counter() - start - preprocessed_time
        self.save_result(result, output_file, retrieval_time)
        # print(result)
        # print(f"Langchain took: {retrieval_time} seconds")
        return result, [
            f"Ingestion and Preprocessing Time: {preprocessed_time}",
            f"Retrieval Time: {retrieval_time}",
            f"Elapsed TIme: {preprocessed_time + retrieval_time}",
        ]


if __name__ == "__main__":
    kb = LangchainKnowledgeBase(
        docs_dir="../docs/",
        azure_api_key=os.getenv("AZURE_API_KEY"),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        endpoint=os.getenv("ENDPOINT"),
    )
    kb.run(
        query="Give me detailed information about sulfuric acid based on the documents provided.",
        output_file="output.txt",
    )
