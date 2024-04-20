from flask import Flask, render_template, request, flash, redirect
import os
from werkzeug.utils import secure_filename
import os,sys
from flask_sqlalchemy import SQLAlchemy
import datetime
import pandas as pd
from logging.config import dictConfig
from  dotenv import dotenv_values, load_dotenv
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_babel import Babel
config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
direc=os.path.abspath(os.path.dirname(__file__))
app=Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"]="sqlite:///"+os.path.join(direc,"data.sqlite")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"]=False
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = os.path.join(direc,"static","Upload")
db=SQLAlchemy(app)

datestamp = datetime.datetime.now().strftime('%Y-%m-%d')
    
basedir = os.path.abspath(os.path.dirname(__file__))
log_folder = basedir+"/static/logs"
os.makedirs(log_folder, exist_ok=True)

admin = Admin(app, name='IDP Customs', template_mode='bootstrap4')
babel = Babel(app)

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


dictConfig(
    {
        "version": 1,
         "formatters":{
            "default": {
                "format": "[%(asctime)s] %(levelname)s | %(module)s >>> %(message)s",
                "datefmt": "%B %d, %Y %H:%M:%S %Z",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": basedir+"/static/logs/application_log-"+datestamp+".log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)