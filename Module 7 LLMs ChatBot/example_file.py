# !pip install openai
# !pip install PyPDF2
# !pip install faiss-cpu
# !pip install tiktoken
# !pip install openai

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import openai
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

import os
os.environ['OPENAI_API_KEY'] = "OPENAIKEY"
# os.environ['SERPAPI_API_KEY'] = ""

pdfreader = PdfReader('CRM scope of work (1).pdf')

from typing_extensions import Concatenate
raw_text = ''
for i, page in enumerate(pdfreader.pages):
  content = page.extract_text()
  if content:
    raw_text += content

raw_text

text_spliter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

texts = text_spliter.split_text(raw_text)

embeddings = OpenAIEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")

query = "What is this document about?"
docs = document_search.similarity_search(query)
print(chain.run(input_documents=docs, question=query))