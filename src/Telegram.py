import os

import telebot

BOT_TOKEN = os.environ.get('BOT_TOKEN')

print(BOT_TOKEN)
bot = telebot.TeleBot(BOT_TOKEN)

import os

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

feynman_lectures_path = '/home/pi/Documents/academIA/docs/The_Feynman_Lectures_on_Physics_-_VOL1.pdf'
feynman_lectures_path = '/home/pi/Documents/academIA/docs/Serway.pdf'

list_pdfs = ['Serway.pdf', 'Purcell.pdf', 'Feynman_vol2.pdf']
# feynman_lectures_path = '/home/pi/Documents/academIA/docs/Reitz-Milford.pdf'

from dotenv import load_dotenv, find_dotenv
import os

from langchain.document_loaders import PyMuPDFLoader, AmazonTextractPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# python -m pip install chromadb=0.3.29

from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import datetime as dt
import pytz




load_dotenv(find_dotenv())
api_key = os.environ['OPENAI_API_KEY']

# Data Loader
pdf_path = "./pdfs/NIPS-2017-attention-is-all-you-need-Paper.pdf"

# documents = []
# for iter_pdf in list_pdfs:
#     loader = PyMuPDFLoader(f'/home/pi/Documents/academIA/docs/{iter_pdf}')
#     documents.extend(loader.load())
#
# # Splitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
# texts = text_splitter.split_documents(documents)

# Embeddings
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory="/home/pi/Documents/academIA/data",
            embedding_function=OpenAIEmbeddings())
print(db._collection.count())
retriever = db.as_retriever()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompts
prompt_template = """Usa los siguientes elementos de contexto para responder la pregunta al final. 
Si la respuesta no se encuentra en el contexto, responde amablemente y con un mensaje alegre.

{context}

Pregunta: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

# Generate
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                 retriever=retriever,
                                 memory=memory,
                                 chain_type_kwargs=chain_type_kwargs
                                 )

print('Preguntas!')


# MongoDB
mongo_user = os.environ.get('MONGO_USER')
mongo_pass = os.environ.get('MONGO_USER')
mongo_host = f'mongodb+srv://{mongo_user}:{mongo_pass}@atheneia.dqt9y1t.mongodb.net/?retryWrites=true&w=majority'
mongo_port = 27017

from pymongo import MongoClient
mongo_client = MongoClient(mongo_host, mongo_port)
db = mongo_client.Atheneia.Conversations

# Obten un objeto de zona horaria para Madrid/Amsterdam
zona_horaria = pytz.timezone('Europe/Madrid')



@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    user_id = message.from_user.id
    ts_mensaje = dt.datetime.now(tz=zona_horaria)


# Prompt You are a friendly chatbot assistant that responds in a conversational
    # manner to users questions. Keep the answers short, unless specifically
    # asked by the user to elaborate on something.
    llm_response = qa('Dado el contexto dado, '+ message.text+'. Explicalo para un alumno que esta aprendiendo los conceptos.')
    ts_respuesta = dt.datetime.now(tz=zona_horaria)
    ts_diferencia = ts_respuesta - ts_mensaje
    ts_diferencia = ts_diferencia.seconds
    answer = llm_response['result']


    # bot.reply_to(message, answer)
    bot.send_message(chat_id=user_id, text=answer)

    info_conversation = {'user_id': user_id,
                         'question': message.text,
                         'answer': answer,
                         'ts_respuesta': ts_respuesta,
                         'ts_mensaje': ts_mensaje,
                         'tiempo_respuesta': ts_diferencia,
                         'timestamp': dt.datetime.now(tz=zona_horaria)
                         }
    db.insert_one(info_conversation)
    # bot.reply_to(message, user_id)

bot.infinity_polling()