from flask import Flask, render_template, request, flash, redirect
import os
from werkzeug.utils import secure_filename
import os,sys
from flask_sqlalchemy import SQLAlchemy
import datetime
import pandas as pd
from logging.config import dictConfig
from  dotenv import dotenv_values, load_dotenv
config = dotenv_values(".env")

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