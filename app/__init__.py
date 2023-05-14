# app/__init__.py
from flask import Flask
from .routes import app

def create_app():
    app = Flask(__name__,template_folder='templates', static_folder='..\static')
    return app

app = create_app()