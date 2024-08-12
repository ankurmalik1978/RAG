import json
from typing import List, Union
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools import Tool, tool
from langchain import HuggingFacePipeline
from langchain.tools.render import render_text_description
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from transformers import AutoTokenizer

# Process
	# Step 1: Load the blog into Langchain
	# Step 2: Split it using Text Splitter into smaller chunks
	# Step 3: Create an embedding model and convert chunks into Vectors 
    # Step 4: Embed them into Vector DB 

if __name__ == "__main__":

    # Step 1 : Load the blog into Langchain
    loader = TextLoader("mediumblog.txt", encoding="utf-8")
    document =loader.load()

    # Step 2 : Split it using Text Splitter into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    texts = text_splitter.split_documents(document)

    print(f"Number of chunks: {len(texts)} chunks")


    # Step 3 : Create an embedding model and convert chunks into Vectors 
    model_name = "C:/Users/I038077/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33"

    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # There was an error:
    # "Asking to pad but the tokenizer does not have a padding token. Please select a token to use as `pad_token"
    # This is because the tokenizer does not have a padding token. We will set the pad token to be the eos token.
    tokenizer = embeddings.client.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    # We will ingest chunks into the vector store
    vector_store = PineconeVectorStore(index="medium-blog-embeddings-index", embedding=embeddings)
    
    if not vector_store.is_index_created():
        PineconeVectorStore.from_documents(texts, embeddings, index_name="medium-blog-embeddings-index")

    hfgppl = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100},
        model_kwargs={"temperature": 0},  # {"stop":["\nObservation", "Observation"]}
    )

