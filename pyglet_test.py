from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_socketio import SocketIO
import os
import Interface
from slmsuite.hardware.slms.screenmirrored import ScreenMirrored
import PhaseManager
import CorrectedSLM
#import CameraClient
#import slmsuite.hardware.cameras.thorlabs
import utils
import re
import numpy as np
import yaml
import ast
import screeninfo
import mss.tools
import datetime
import pyglet
import threading

from pyglet.window import key
from pyglet.window import mouse

app = Flask(__name__)
app.secret_key = os.urandom(24)
socketio = SocketIO(app)

def start_flask_app():
    socketio.run(app, host="0.0.0.0", port=8080, debug=False)

window = pyglet.window.Window(visible=True)

@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.A:
        print('The "A" key was pressed.')
    elif symbol == key.LEFT:
        print('The left arrow key was pressed.')
    elif symbol == key.ENTER:
        print('The enter key was pressed.')

event_logger = pyglet.window.event.WindowEventLogger()
window.push_handlers(event_logger)

@window.event
def on_draw():
    window.clear()

event_logger = pyglet.window.event.WindowEventLogger()
window.push_handlers(event_logger)

pyglet.app.run()

    