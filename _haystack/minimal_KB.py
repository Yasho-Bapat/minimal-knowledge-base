import os
from time import perf_counter
import dotenv

from haystack.components.writers import DocumentWriter
from haystack.components.converters import PyPDFToDocument, AzureOCRDocumentConverter
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.cohere import CohereGenerator
from haystack_integrations.components.embedders.cohere import (
    CohereDocumentEmbedder,
    CohereTextEmbedder,
)
from haystack.utils import Secret
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

dotenv.load_dotenv()


class HaystackKnowledgeBase:
    def __init__(self, docs_dir, azure_api_key, cohere_api_key, endpoint):
        self.docs_dir = docs_dir
        self.endpoint = endpoint
        self.azure_api_key = azure_api_key
        self.cohere_api_key = cohere_api_key

        self.document_store = ChromaDocumentStore()
        self.document_embedder = CohereDocumentEmbedder(
            api_key=Secret.from_token(self.cohere_api_key)
        )
        self.pdf_converter = PyPDFToDocument()
        self.document_cleaner = DocumentCleaner()
        self.document_splitter = DocumentSplitter(
            split_by="word", split_length=200, split_overlap=50
        )
        self.document_writer = DocumentWriter(document_store=self.document_store)
        self.embedding_function = CohereTextEmbedder(
            api_key=Secret.from_token(self.cohere_api_key)
        )
        self.llm = CohereGenerator(api_key=Secret.from_token(self.cohere_api_key))

    def preprocess_documents(self):
        preprocessing_pipeline = Pipeline()
        preprocessing_pipeline.add_component(
            "converter",
            AzureOCRDocumentConverter(
                endpoint=self.endpoint, api_key=Secret.from_token(self.azure_api_key)
            ),
        )
        preprocessing_pipeline.add_component("cleaner", DocumentCleaner())
        preprocessing_pipeline.add_component(
            "splitter", DocumentSplitter(split_by="sentence", split_length=5)
        )
        preprocessing_pipeline.add_component("embedder", self.document_embedder)
        preprocessing_pipeline.add_component(
            "writer", DocumentWriter(document_store=self.document_store)
        )

        preprocessing_pipeline.connect("converter", "cleaner")
        preprocessing_pipeline.connect("cleaner", "splitter")
        preprocessing_pipeline.connect("splitter", "embedder")
        preprocessing_pipeline.connect("embedder", "writer")

        file_paths = [self.docs_dir + file for file in os.listdir(self.docs_dir)]
        preprocessing_pipeline.run({"converter": {"sources": file_paths}})

    def retrieve_documents(self, query):
        # improve
        template = """
        You are an expert Material Safety Document Analyser.
        Given the following information, answer the question.
        Make sure there are full stops after every sentence.

        Context: 
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{ query }}?
        """

        retrieval_pipeline = Pipeline()
        retrieval_pipeline.add_component("embedder", self.embedding_function)
        retrieval_pipeline.add_component(
            "retriever",
            ChromaEmbeddingRetriever(document_store=self.document_store, top_k=10),
        )
        retrieval_pipeline.add_component(
            "prompt_builder", PromptBuilder(template=template)
        )
        retrieval_pipeline.add_component("llm", self.llm)

        retrieval_pipeline.connect("embedder.embedding", "retriever.query_embedding")
        retrieval_pipeline.connect("retriever", "prompt_builder.documents")
        retrieval_pipeline.connect("prompt_builder", "llm")

        result = retrieval_pipeline.run(
            {
                "embedder": {"text": query},
                "prompt_builder": {"query": query},
            }
        )

        return result["llm"]["replies"][0]

    def save_result(self, result, filename, elapsed_time):
        with open(filename, "w") as file:
            file.write(result + "\n\n" + f"Haystack took: {elapsed_time} seconds")

    def run(self, query, output_file):
        start = perf_counter()
        self.preprocess_documents()
        preprocessed_time = perf_counter() - start
        result = self.retrieve_documents(query)
        retrieval_time = perf_counter() - start - preprocessed_time
        self.save_result(result, output_file, retrieval_time)
        # print(result)
        # print(f"Haystack took: {retrieval_time} seconds")
        return result, [
            f"Ingestion and Preprocessing Time: {preprocessed_time}",
            f"Retrieval Time: {retrieval_time}",
            f"Elapsed TIme: {preprocessed_time + retrieval_time}",
        ]


if __name__ == "__main__":
    kb = HaystackKnowledgeBase(
        docs_dir="../docs/",
        azure_api_key=os.getenv("AZURE_API_KEY"),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        endpoint=os.getenv("ENDPOINT"),
    )
    kb.run(
        query="Give me detailed information about sulfuric acid based on the documents provided.",
        output_file="output.txt",
    )
