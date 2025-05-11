from src.helper import load_pdf_file, download_hugging_face_embeddings, text_split
from pinecone.grpc import PineconeGRPC as pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
load_dotenv()

PINECONE_API_KEY= os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
extracted_data= load_pdf_file(data='Data')
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embeddings()

pc= pinecone(api_key=PINECONE_API_KEY)
index_name='mdicalchatbot'


pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

docsearch= PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)