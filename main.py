from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return "image classification model"