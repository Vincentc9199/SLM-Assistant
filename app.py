from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_socketio import SocketIO

import Interface
import PhaseManager
import CorrectedSLM
#import CameraClient
import utils

from slmsuite.hardware.slms.screenmirrored import ScreenMirrored
import slmsuite.hardware.slms.slm
from slmsuite.hardware.cameras.thorlabs import ThorCam
from slmsuite.hardware.cameras.camera import Camera

import os
import numpy as np
import yaml
import ast
import screeninfo
import mss.tools
import datetime
import pyglet
import threading
from matplotlib import pyplot as plt

###################################################################################################

#FLASK APP

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app)

def start_flask_app():
    print("Starting Flask App")
    socketio.run(app, host="0.0.0.0", port=8080, debug=False)

###################################################################################################

# INITIALIZE HISTORIES

base_load_history = []
additional_load_history = []
additional_save_history = []
config_load_history = []
config_save_history = []


###################################################################################################

# GET CURRENT WORKING DIRECTORY

directory = os.getcwd()

###################################################################################################

# PYGLET APP

def start_pyglet_app():
    print("Starting Pyglet App")

    while True:
        pyglet.clock.tick()

        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()

class SLMEventDispatcher(pyglet.event.EventDispatcher):
    def create_slm(self):
        self.dispatch_event('on_create_slm')

    def project_pattern(self):
        self.dispatch_event('on_project_pattern')

SLMEventDispatcher.register_event_type('on_create_slm')
SLMEventDispatcher.register_event_type('on_project_pattern')
dispatcher = SLMEventDispatcher()

###################################################################################################

# SETUP COMPUTATIONAL SPACE AND DEFAULT # OF ITERATIONS

computational_space = (2048, 2048)
n_iterations = 20

@app.route('/setup_calculation', methods=['GET', 'POST'])
def setup_calculation():
    global computational_space, n_iterations

    if request.method == 'POST':
        
        size = request.form['computational_space']
        if size:
            size = int(size)  
            computational_space = (size, size)
            print("Updated Computational Space to: " + str(computational_space))
            #flash("Updated Computational Space to: " + str(computational_space))
            

        new_n_iterations = request.form['n_iterations']
        if new_n_iterations:
            new_n_iterations = int(new_n_iterations)  
            n_iterations = new_n_iterations
            print("Updated Number of Iterations to: " + str(n_iterations))
            #flash("Updated Number of Iterations to: " + str(n_iterations))

        return redirect(url_for('setup_calculation'))
    
    return render_template('setup_calculation.html', 
                           computational_space=computational_space, 
                           n_iterations=n_iterations)

###################################################################################################

# SETUP MANUALLY COMPUTED CALIBRATION MATRIX

A = np.array([[-1.3347, 0.89309], [0.89306, 1.3414]])
b = np.array([[1254.758], [277.627]])

@app.route('/setup_calibration', methods=['GET', 'POST'])
def setup_calibration():
    global A, b

    if request.method == "POST":

        top_left = request.form['top_left']
        top_right = request.form['top_right']
        bottom_left = request.form['bottom_left']
        bottom_right = request.form['bottom_right']

        vector_x = request.form['vector_x']
        vector_y = request.form['vector_y']

        if all([top_left, top_right, bottom_left, bottom_right, vector_x, vector_y]):
            
            top_left = float(top_left)
            top_right = float(top_right)
            bottom_left = float(bottom_left)
            bottom_right = float(bottom_right)
            vector_x = float(vector_x)
            vector_y = float(vector_y)

            A = np.array([[top_left, top_right], [bottom_left, bottom_right]])
            b = np.array([[vector_x], [vector_y]])

            print("Updated calibration to: " + str(A) + "+ " + str(b))
            #flash("Updated calibration to: " + str(A) + "+ " + str(b))
            
        else:
            print("One of the entries is missing")

        return redirect(url_for('setup_calibration'))
    
    return render_template('setup_calibration.html', A=A, b=b)

###################################################################################################

# CONNECT TO AN SLM

# List of dictionaries containing SLM settings
slm_list = []

setup_slm_settings = {}

@app.route('/setup_slm', methods=['GET', 'POST'])
def setup_slm():
    global setup_slm_settings, slm_list

    if request.method == 'POST':
        
        # Get SLM settings from user
        display_num = request.form['display_num']
        bitdepth = request.form['bitdepth']
        wav_design_um = request.form['wav_design_um']
        wav_um = request.form['wav_um']

        if all([display_num, bitdepth, wav_design_um, wav_um]):
            
            setup_slm_settings['display_num'] = int(display_num)
            setup_slm_settings['bitdepth'] = int(bitdepth)
            setup_slm_settings['wav_design_um'] = float(wav_design_um)
            setup_slm_settings['wav_um'] = float(wav_um)
    
            # Call to create pyglet window
            pyglet.clock.schedule_once(lambda dt: dispatcher.create_slm(), 0)
        else:
            print("Missing one of the SLM settings")

        return redirect(url_for('setup_slm'))

    return render_template('setup_slm.html', slm_list=slm_list)

@dispatcher.event
def on_create_slm():
    global setup_slm_settings, slm_list

    # Create pyglet window
    slm = ScreenMirrored(setup_slm_settings['display_num'], 
                            setup_slm_settings['bitdepth'], 
                            wav_design_um=setup_slm_settings['wav_design_um'], 
                            wav_um=setup_slm_settings['wav_um'])

    # Create phase manager for the slm
    phase_mgr = PhaseManager.PhaseManager(slm)

    # Wrap together slm and phase manager
    wrapped_slm = CorrectedSLM.CorrectedSLM(slm, phase_mgr)

    # Add the wrapped slm to the settings
    setup_slm_settings['slm'] = wrapped_slm

    # Add settings dictionary to the list
    slm_list.append(setup_slm_settings.copy())

    print("Succesfully connected SLM on display: " + str(setup_slm_settings['display_num']))

@app.route('/setup_virtual_slm', methods=['GET'])
def setup_virtual_slm():
    global setup_slm_settings, slm_list

    if request.method == 'GET':
        setup_slm_settings['display_num'] = "virtual"
        setup_slm_settings['bitdepth'] = "virtual"
        setup_slm_settings['wav_design_um'] = "virtual"
        setup_slm_settings['wav_um'] = "virtual"

        # Create a virtual slm (default size 1272x1024)
        slm = slmsuite.hardware.slms.slm.SLM(1272, 1024)

        # Create phase manager for the slm
        phase_mgr = PhaseManager.PhaseManager(slm)

        # Wrap together slm and phase manager
        wrapped_slm = CorrectedSLM.CorrectedSLM(slm, phase_mgr)

        # Add the wrapped slm to the settings
        setup_slm_settings['slm'] = wrapped_slm

        # Add settings dictionary to the list
        slm_list.append(setup_slm_settings.copy())

        print("Succesfully setup virtual SLM")

        return redirect(url_for('setup_slm'))

    return redirect(url_for('setup_slm'))

# Structure of CorrectedSLM
"""
CorrectedSLM

Initalized with:
--> slm
--> phase_mgr

Attributes:
- slm --> passed ScreenMirrored SLM
- phase_mgr --> passed phase_mgr

from the abstract SLM class
- width 
- height
- bitdepth
- wav_um
- dx um
- dy um
the values are set to match the ScreenMirrored SLM
"""

# Structure of PhaseManager

"""
PhaseManager

Initialized with:
slm --> the ScreenMirrored SLM
shape --> the ScreenMirrored shape
base --> array containing base phase
base_source --> file path for loaded base pattern
additional --> array containing additional phase
add_log --> log of everything that has been added to the additional
aperture
mask

"""

###################################################################################################

# CONENCT TO A CAMERA

# List of camera settings dictionaries
camera_list = []
setup_camera_settings = {}

@app.route('/setup_camera', methods=['GET', 'POST'])
def setup_camera():
    global setup_camera_settings, camera_list

    if request.method == 'POST':
        # Get camera settings from user
        camera_type = request.form['camera_type']
        serial_num = request.form['serial_num']
        fliplr = request.form['fliplr']
        
        # Create the camera
        if all([camera_type, serial_num, fliplr]):
            
            if camera_type == "thorlabs":

                if fliplr == 'False':
                    fliplr = False
                elif fliplr == 'True':
                    fliplr = True
                else:
                    print("Select a value for fliplr")
                    return redirect(url_for('setup_camera'))
                
                camera = ThorCam(serial=serial_num, fliplr=fliplr)

                # Add them to the settings dictionary
                setup_camera_settings['camera_type'] = camera_type
                setup_camera_settings['serial_num'] = serial_num
                setup_camera_settings['fliplr'] = fliplr
                setup_camera_settings['camera'] = camera

                # Add settings to the list of cameras
                camera_list.append(setup_camera_settings.copy())

                print("Succesfully connected to camera of type: " + setup_camera_settings['camera_type']
                        + "with serual number " + setup_camera_settings['serial_num'])
            else:
                print("Camera Type Not Setup Yet")
        else:
            print("Enter all of the camera settings")

        return redirect(url_for('setup_camera'))

    return render_template('setup_camera.html', camera_list=camera_list)

@app.route('/setup_virtual_camera', methods=['GET'])
def setup_virtual_camera():
    global setup_camera_settings, camera_list

    if request.method == 'GET':
        setup_camera_settings['camera_type'] = 'virtual'
        setup_camera_settings['serial_num'] = 'virtual'
        setup_camera_settings['fliplr'] = 'virtual'

        # Create virtual camera, default size 1024x1024
        camera = Camera(1024, 1024)

        setup_camera_settings['camera'] = camera

        # Add settings to the list of cameras
        camera_list.append(setup_camera_settings.copy())

        print("Succesfully setup virtual camera")

        return redirect(url_for('setup_camera'))

    return redirect(url_for('setup_camera'))
###################################################################################################

# SETUP THE INTERFACE

# Initialize the Interface
iface = Interface.SLMSuiteInterface()

@app.route('/setup_iface', methods=['GET', 'POST'])
def setup_iface():
    global iface, slm_list, camera_list
    if request.method == 'POST':
        # Get user selected slm index
        slm_num = request.form['slm_num']
        
        
        slm_num = int(slm_num)
        # Get corresponding slm settings
        slm_settings = slm_list[slm_num]

        # Extract the CorrectedSLM
        slm = slm_settings['slm']

        # Set as the SLM in iface
        iface.set_SLM(slm=slm, slm_settings=slm_settings)

        print("Selected SLM with display number: " + str(slm_settings['display_num']))

        # Get user selected camera index
        camera_num = request.form['camera_num']
        
        camera_num = int(camera_num)

        # Get corresponsing camera settings
        camera_settings = camera_list[camera_num]

        # Extract camera
        camera = camera_settings['camera']

        # Set as the camera in iface
        iface.set_camera(camera, camera_settings=camera_settings)
    
        print("Selected camera of type: " + str(camera_settings['camera_type']) + " with serial number" + str(camera_settings['serial_num']))

        return redirect(url_for('setup_iface'))
    else:
        camera_settings={}
        slm_settings={}

    return render_template('setup_iface.html', 
                           slm_list=slm_list,
                           camera_list=camera_list,
                           slm_settings=iface.slm_settings,
                           camera_settings=iface.camera_settings)

#Structure of Interface
"""
Interface

slm
camera
camerslm
hologram
fourier calibration source
input amplitudes
input targets
slm settings
camera settings
"""
###################################################################################################

# Fourier Calibration (Perform, Save and Load)

###################################################################################################

"""
@app.route('/setup_slm_amp', methods=['GET', 'POST'])
def setup_slm_amp():
    global iface

    if request.method == 'POST':
        if iface.cameraslm is not None:
            func = request.form['func']
            waist_x = request.form['waist_x']
            waist_y = request.form['waist_y']
            try:
                func = str(func)

                if func == "gaussian":
                    waist_x = float(waist_x)
                    waist_x = np.array([waist_x])
                    waist_y = float(waist_y)
                    waist_y = np.array([waist_y])

                    shape = iface.slm.shape
                    xpix = (shape[1] - 1) *  np.linspace(-.5, .5, shape[1])
                    ypix = (shape[0] - 1) * np.linspace(-.5, .5, shape[0])

                    x_grid, y_grid = np.meshgrid(xpix, ypix)

                    gaussian_amp = np.exp(-np.square(x_grid) * (1 / waist_x**2)) * np.exp(-np.square(y_grid) * (1 / waist_y**2))

                    iface.set_slm_amplitude(gaussian_amp)
                    print(f"Set SLM amplitude to Gaussian with waist: ({waist_x}, {waist_y})")
                    #flash(f"Set SLM amplitude to Gaussian with waist: ({waist_x}, {waist_y})")
                else:
                    print("Amp type not yet setup")
            except:
                print("Input correct datatypes")
        else:
            print('Select a camera and an SLM')
        
        return redirect(url_for('setup_slm_amp'))
    
    return render_template('setup_slm_amp.html')
"""

###################################################################################################

#Set Measured Amplitude Analytics

###################################################################################################

#Wavefront Calibration (Perform, Save, Load)

###################################################################################################
###################################################################################################
###################################################################################################


@app.route('/', methods=['GET', 'POST'])
def dashboard():
    global iface
    if iface.cameraslm is not None:
        phase_info = get_phase_info()
        slm_settings = iface.slm_settings
        camera_settings = iface.camera_settings
        if slm_settings['display_num'] != 'virtual':
            get_screenshot()
    else:
        phase_info=None
        slm_settings=None
        camera_settings=None
    return render_template('dashboard.html', 
                           phase_info=phase_info,
                           slm_settings=slm_settings,
                           camera_settings=camera_settings)


###################################################################################################

# GET ALL OF THE PHASE INFORMATION IN THE PHASE MANAGER

def get_phase_info():
    global iface, directory

    if iface.cameraslm is not None:
        # Get the phase manager
        phase_mgr = iface.slm.phase_mgr

        # Get the file path of the base pattern
        base_str = phase_mgr.base_source

        base_path = os.path.join(directory, 'data', 'base')
        base_str = base_str.replace(base_path, "")

        # Additional Pattern List
        add_list = []
        # Log of additional
        log = phase_mgr.add_log
        # Iterate over additional phase patterns and add to string
        for item in log:
            add_list.append(str(item[0]) + ":" + str(item[1]))

        # Get the aperture
        aperture = phase_mgr.aperture
        aperture_str = str(aperture)

        print("Succesfully got phase info")
        #flash("Succesfully got phase info")

        return base_str, add_list, aperture_str
    
    else:
        print("Select a Camera and an SLM")


###################################################################################################

# TAKE A SCREENSHOT OF THE SLM


def get_screenshot():
    global iface
    if iface.cameraslm is not None:
        if iface.slm_settings['display_num'] != 'virtual':
            displays = screeninfo.get_monitors()
            
            display_num = iface.slm_settings['display_num']

            display = displays[display_num]

            # Create area for screenshot
            display_rect = {
                "top": display.y,
                "left": display.x,
                "width": display.width,
                "height": display.height
            }
            output = "static/images/slm_screenshot.png"

            # Take screenshot
            sct = mss.mss()
            screenshot = sct.grab(display_rect)

            # Save to the picture file
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=output)
            print("Succesfully saved screenshot to: " + output)
            #flash("Succesfully saved screenshot to: " + output)
        else:
            print("Cannot take screenshot of virtual SLM")
    else:
        print("Select a camera and an SLM")

        

###################################################################################################

# PROJECT THE PHASE IN THE PHASE MANAGER ONTO THE SLM


@app.route('/project', methods=['POST'])
def project():
    global iface
    
    if request.method == 'POST':
        if iface.cameraslm is not None:
            if iface.slm_settings['display_num'] != 'virtual':
                pyglet.clock.schedule_once(lambda dt: dispatcher.project_pattern(), 0)
                
            else:
                print("Cannot project to a virtual SLM")
        else:
            print("Select a camera and an SLM")

        return redirect(url_for('dashboard'))
    
    return redirect(url_for('dashboard'))

@dispatcher.event
def on_project_pattern():
    global iface

    phase_mgr = iface.slm.phase_mgr
    iface.write_to_SLM(phase_mgr.base, phase_mgr.base_source)
    
    print("Succesfully projected to display: " + str(iface.slm_settings['display_num']))


###################################################################################################

# DISPLAY THE TARGET OF THE LOADED BASE PHASE IN THE PHASE MANAGER


@app.route('/display_target_from_base')
def display_target_from_base():
    global iface

    if iface.cameraslm is not None:
        phase_mgr = iface.slm.phase_mgr

        if phase_mgr.base_source:
    
            _,data = utils.load_slm_calculation(phase_mgr.base_source, 0, 1)
            input_targets = data['input_targets']
            x_coords = input_targets[0].tolist()
            y_coords = input_targets[1].tolist()
            labels = list(range(1, len(x_coords) + 1))

            print("Displaying targets from: " + phase_mgr.base_source)
            #flash("Displaying targets from: " + phase_mgr.base_source)

            return jsonify({'x': x_coords, 'y': y_coords, 'labels': labels})
        else:
            print("No base phase is loaded")
    else:
        print("Select a camera and an SLM")



###################################################################################################
###################################################################################################
###################################################################################################

@app.route('/phase_summary', methods=['GET', 'POST'])
def phase_summary():
    global base_load_history, computational_space, n_iterations, A, b
    phase_info = get_phase_info()
    get_plots()
    return render_template('phase_summary.html', phase_info=phase_info)

def get_plots():
    global iface
    if iface.cameraslm is not None:
        phase_mgr = iface.slm.phase_mgr
        phase_mgr.plot_base()
        phase_mgr.plot_additional()
        phase_mgr.plot_total_phase()
    else:
        print("Select a camera and an SLM")
###################################################################################################

@app.route('/load_from_file', methods=['GET', 'POST'])
def load_from_file():
    global base_load_history, additional_save_history, additional_load_history
    return render_template('load_from_file.html',
                           base_load_history=base_load_history,
                           additional_save_history=additional_save_history,
                           additional_load_history=additional_load_history)


###################################################################################################

# LOAD BASE PHASE FROM SAVED FILE INTO PHASE MANAGER


@app.route('/use_pattern', methods=['POST'])
def use_pattern():
    global directory, iface

    if request.method == 'POST':
        if iface.cameraslm is not None:
            # Get the file name input by the user
            fname = request.form['fname']
            path = os.path.join(directory, 'data', 'base', fname)

            load_base(path)
        else:
            print("Select a camera and an SLM")
        return redirect(url_for('load_from_file'))
    return redirect(url_for('load_from_file'))


def load_base(path):
    global base_load_history, iface, computational_space, socketio

    # Get the phase pattern from the file
    _,data = utils.load_slm_calculation(path, 0, 1)
    phase = data["slm_phase"]

    phase_mgr = iface.slm.phase_mgr
    # Set the phase pattern as the base of the phase manager
    phase_mgr.set_base(phase, path)
    print("Base Pattern added from: " + path)

    # Get the time the file was uploaded
    upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Add the file name and upload time to the history
    base_load_history.append({'fname': path, 'upload_time': upload_time})
 

###################################################################################################

# CONVERT BETWEEN ANDOR CAMERA AND SLM K-SPACE COORDINATES

def andor_to_k(x):
    global A, b
    targets = np.matmul(A,x)+b
    return targets

def k_to_andor(k):
    global A, b
    invA = np.linalg.inv(A)
    targets = np.matmul(invA, k - b)
    return targets

###################################################################################################

# CREATE A HOLOGRAM BY MANUALLY INPUTTING COORDINATES

@app.route('/manual', methods=['GET', 'POST'])
def manual():
    global iface, computational_space, socketio

    if request.method == 'POST':
        if iface.cameraslm is not None:

            # Get the user input
            x_coords = request.form.getlist('x_coords')
            y_coords = request.form.getlist('y_coords')
            amplitudes = request.form.getlist('amplitudes')
            coord_type = request.form['coord_type']

            
            x_coords = list(map(float, x_coords))
            y_coords = list(map(float, y_coords))
            targets = np.array([x_coords, y_coords])

            if coord_type == 'andor':
                targets = andor_to_k(targets)
                print("Converted from Andor coords to k-space")

            # Copy the targets
            target_data_for_input = np.copy(targets)

            print("Received Targets:" + str(targets))
            
            # Convert amplitudes to floats
            amplitudes = list(map(float, amplitudes))
            # Create 1D numpy array containing amplitudes
            amp_data = np.array(amplitudes)
            # Create a copy of what the user input
            amp_data_for_input = np.copy(amp_data)

            print("Received amplitudes: " + str(amp_data))
            
            # Store the input target and amplitudes in the iface
            iface.input_targets = target_data_for_input
            iface.input_amplitudes = amp_data_for_input
            
            basis = 'knm'
            # Change to ij if fourier calibration

            # Create the hologram in the iface
            iface.set_hologram(computational_shape=computational_space, target_spot_array=targets, target_amps=amp_data, basis=basis, socketio=socketio)

            # Save an image of the target hologram
            iface.plot_target()

            print("Succesfully created hologram")
        else:
            print("Select a camera and SLM")
        
        return redirect(url_for('manual'))

    return render_template('manual.html')

###################################################################################################

# CREATE A HOLOGRAM AS A LATTICE FILLING UP A BOX

@app.route('/lattice_box', methods=['GET', 'POST'])
def lattice_box():
    global iface, computational_space, socketio

    if request.method == 'POST':
        if iface.cameraslm is not None:
            lv11 = request.form['lv11']
            lv12 = request.form['lv12']
            lv21 = request.form['lv21']
            lv22 = request.form['lv22']
            offset_x = request.form['offset_x']
            offset_y = request.form['offset_y']
            width = request.form['width']
            height = request.form['height']
            center_x = request.form['center_x']
            center_y = request.form['center_y']
            edge_buffer = request.form['edge_buffer']
            coord_type = request.form['coord_type']

            lv11 = float(lv11)
            lv12 = float(lv12)
            lv21 = float(lv21)
            lv22 = float(lv22)
            offset_x = float(offset_x)
            offset_y = float(offset_y)
            width = float(width)
            height = float(height)
            center_x = float(center_x)
            center_y = float(center_y)
            edge_buffer = float(edge_buffer)

            lattice_vectors = [
                np.array([lv11, lv21]),
                np.array([lv12, lv22])]
            image_shape = (width, height)
            offset = (offset_x, offset_y)
            center_pix = (center_x, center_y)
            spots, x_coords, y_coords = generate_lattice(image_shape, lattice_vectors, offset, center_pix, edge_buffer)

            x_coords = list(map(float, x_coords))
            y_coords = list(map(float, y_coords))
            targets = np.array([x_coords, y_coords])

            if coord_type == 'andor':
                targets = andor_to_k(targets)
                print("Converted from Andor coords to k-space")

            # Copy the targets
            target_data_for_input = np.copy(targets)

            print("Received Targets:" + str(targets))

            num_spots = len(x_coords)

            amp_data = np.ones(num_spots, dtype = float)

            amp_data_for_input = np.copy(amp_data)

            print("Received amplitudes: " + str(amp_data))
            
            iface.input_amplitudes = amp_data_for_input
            iface.input_targets = target_data_for_input
            
            basis = 'knm'

            iface.set_hologram(computational_shape=computational_space, target_spot_array=targets, target_amps=amp_data, basis=basis, socketio=socketio)
            iface.plot_target()
            
            print("Succesfully created hologram")
        else:
            print("Select a camera and an SLM")

        return redirect(url_for('lattice_box'))
    
    return render_template('lattice_box.html')

def generate_lattice( #Help me speed up this function, please!
    image_shape, lattice_vectors, offset=(0, 0), center_pix='image', edge_buffer=2):

    ##Preprocessing. Not much of a bottleneck:
    if center_pix == 'image':
        center_pix = np.array(image_shape) // 2
    else: ##Express the center pixel in terms of the lattice vectors
        center_pix = np.array(center_pix) - (np.array(image_shape) // 2)
        lattice_components = np.linalg.solve(
            np.vstack(lattice_vectors[:2]).T,
            center_pix)
        lattice_components -= lattice_components // 1
        center_pix = (lattice_vectors[0] * lattice_components[0] +
                      lattice_vectors[1] * lattice_components[1] +
                      np.array(image_shape)//2)
    num_vectors = int( ##Estimate how many lattice points we need
        max(image_shape) / np.sqrt(lattice_vectors[0]**2).sum())
    lattice_points = []
    x_coords = []
    y_coords = []
    lower_bounds = np.array((edge_buffer, edge_buffer))
    upper_bounds = np.array(image_shape) - edge_buffer

    ##SLOW LOOP HERE. 'num_vectors' is often quite large.
    for i in range(-num_vectors, num_vectors):
        for j in range(-num_vectors, num_vectors):
            lp = i * lattice_vectors[0] + j * lattice_vectors[1] + center_pix
            if all(lower_bounds < lp) and all(lp < upper_bounds):
                lp = lp + np.array(offset)
                lattice_points.append(lp)
                x_coords.append(lp[0])
                y_coords.append(lp[1])
    return lattice_points, x_coords, y_coords

###################################################################################################

# CREATE A HOLOGRAM BY DRAWING ON A CANVAS OR GRID

@app.route('/canvas', methods=['GET', 'POST'])
def canvas():
    return render_template('canvas.html')

@app.route('/grid', methods=['GET', 'POST'])
def grid():
    return render_template('grid.html')

@app.route('/submit_points', methods=['POST'])
def submit_points():
    global iface, computational_space, socketio
    if iface.cameraslm is not None:
        data = request.json
        x_coords = data['Coords1']
        y_coords = data['Coords2']
        amplitudes = data['amplitudes']
        coord_type = data['coord_type']
        
        
        x_coords = list(map(float, x_coords))
        y_coords = list(map(float, y_coords))
        targets = np.array([x_coords, y_coords])

        if coord_type == 'andor':
                targets = andor_to_k(targets)
                print("Converted from Andor coords to k-space")

        # Copy the targets
        target_data_for_input = np.copy(targets)

        print("Received Targets:" + str(targets))

        
        # Convert amplitudes to floats
        amplitudes = list(map(float, amplitudes))
        # Create 1D numpy array containing amplitudes
        amp_data = np.array(amplitudes)
        amp_data_for_input = np.copy(amp_data)

        print("Received amplitudes: " + str(amp_data))
        
        iface.input_amplitudes = amp_data_for_input
        iface.input_targets = target_data_for_input

        basis = 'knm'
        
        iface.set_hologram(computational_shape=computational_space, target_spot_array=targets, target_amps=amp_data, basis=basis, socketio=socketio)
        iface.plot_target()
        
        return jsonify({'status': 'success'})
    else:
        print("Select a camera and an SLM")
        return jsonify({'status': 'error'})
    
###################################################################################################

# CREATE A HOLOGRAM FROM THE COORDINATES AND AMPLITUDES IN A FILE


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    global directory, iface

    if request.method == 'POST':
        if iface.cameraslm is not None:

            fname = request.form['fname']
            input_amps = request.form['input_amps']
            
            path = os.path.join(directory, 'data', 'base', fname)
            _,data = utils.load_slm_calculation(path, 0, 1)
            input_targets = data['input_targets']
            x_coords = input_targets[0].tolist()
            y_coords = input_targets[1].tolist()

            if input_amps:
                input_amps_list = input_amps.split(',')
                input_amplitudes = list(map(float, input_amps_list))

            else:
                input_amplitudes = data['input_amplitudes']

            num_points = len(x_coords)
            return render_template('feedback.html', x_coords=x_coords, y_coords=y_coords, num_points=num_points, input_amplitudes=input_amplitudes)
        else:
            print("Select a camera and an SLM")
            return redirect(url_for('feedback'))
    else:
        x_coords = []
        y_coords = []
        input_amplitudes = []
        num_points = 0
    return render_template('feedback.html', x_coords=x_coords, y_coords=y_coords, num_points=num_points, input_amplitudes=input_amplitudes)

###################################################################################################

@app.route('/calculate', methods=['GET', 'POST'])
def calculate():
    global iface
    hologram = iface.hologram
    if iface.cameraslm is not None:
        base_source = iface.slm.phase_mgr.base_source
    else:
        base_source = ""
    return render_template('calculate.html', hologram=hologram, base_source=base_source)

@app.route('/calculate_phase', methods=['POST'])
def calculate_phase():
    global n_iterations, computational_space, directory, iface

    if iface.hologram is not None:

        data = request.json
        save_name = data["save_name"]
        iteration_number = data['iteration_number']
        guess_name = data['guess_name']

        if not iteration_number:
            iteration_number = n_iterations
        else:
            iteration_number= int(iteration_number)

        # Initialize guess phase
        guess_phase = None

        # If there is a guess file path
        if guess_name:
            # Add the pattern folder path
            guess_path = os.path.join(directory, 'data', 'base', guess_name)
            # Extract guess phase pattern
            _,data = utils.load_slm_calculation(guess_path, 1, 1)
            # Check if data was in the file
            if "raw_slm_phase" in data:
                # Store guess phase pattern
                guess_phase = data["raw_slm_phase"]
                iface.hologram.phase = guess_phase
                print("Stored initial guess phase pattern")
            else:
                print ("Cannot initiate the guess phase, since it was not saved")

        # Calculate the base pattern to create the target using GS or WGS algo 

        iface.optimize(iteration_number)

        iface.plot_slmplane()
        iface.plot_farfield()
        iface.plot_stats()

        saved_pattern_path = save_calculation(save_name)[:-9]
        
        load_base(saved_pattern_path)
        print("Finished Calculation, Save and Load")
        return jsonify({'status': 'success'})
    else:
        print("Must have a hologram to calculate base phase")
        return jsonify({'status': 'error'})

def save_calculation(save_name):
    global directory, iface
    
    # Add pattern path if its not an absolute path
    save_path = os.path.join(directory, 'data', 'base')

    # Dictionary to store save options
    save_options = dict()
    save_options["config"] = True # This option saves the configuration of this run of the algorithm
    save_options["slm_pattern"] = True # This option saves the slm phase pattern and amplitude pattern (the amplitude pattern is not calculated. So far, the above have assumed a constant amplitude and we have not described functionality to change this)
    save_options["ff_pattern"] = True # This option saves the far field amplitude pattern
    save_options["target"] = True # This option saves the desired target
    save_options["path"] = save_path # Enable this to save to a desired path. By default it is the current working directory
    save_options["name"] = save_name # This name will be used in the path.
    save_options["crop"] = True # This option crops the slm pattern to the slm, instead of an array the shape of the computational space size.

    # Save the calculated pattern to a new file
    config_path, saved_pattern_path, err = iface.save_calculation(save_options)
    print("Saved calculation to: " + saved_pattern_path)
    return saved_pattern_path

###################################################################################################

@app.route('/reset_pattern', methods=['GET', 'POST'])
def reset_pattern():
    global iface

    if request.method == "POST":
        if iface.cameraslm is not None:
            phase_mgr = iface.slm.phase_mgr
            phase_mgr.reset_base()
            print("Reset Base Phase")
        else:
            print("Select a camera and an SLM")

        return redirect(url_for('phase_summary'))

    return redirect(url_for('phase_summary'))

###################################################################################################

"""
target_path = ""

@app.route('/targets', methods=['GET', 'POST'])
def targets():
    global main_path, directory, target_path
    if request.method == "POST":
        #fname = request.files['fname'].filename
        fname = request.form['fname']
        target_path = os.path.join(directory, 'data', 'base', fname)

        return redirect(url_for('targets'))
    
    return render_template('targets.html')

@app.route('/display_targets')
def display_targets():
    global target_path
    
    #targets = utils.get_target_from_file(target_path)
    #x_coords = targets[0].tolist()
    #y_coords = targets[1].tolist()
    _,data = utils.load_slm_calculation(target_path, 0, 1)
    input_targets = data['input_targets']
    x_coords = input_targets[0].tolist()
    y_coords = input_targets[1].tolist()
    labels = list(range(1, len(x_coords) + 1))

    print("Displaying targets from: " + target_path)
    #flash("Displaying targets from: " + phase_mgr.base_source)

    return jsonify({'x': x_coords, 'y': y_coords, 'labels': labels})
"""

###################################################################################################
###################################################################################################
###################################################################################################


@app.route('/reset_additional_phase', methods=['POST'])
def reset_additional_phase():
    global iface
    if request.method == 'POST':
        if iface.cameraslm is not None:
            phase_mgr = iface.slm.phase_mgr
            phase_mgr.reset_additional()
            print("Reset Additional Phase") 
        else:
            print("Select a camera and an SLM")

        return redirect(url_for('phase_summary'))
    return redirect(url_for('phase_summary'))


###################################################################################################

@app.route('/reset_aperture', methods=['POST'])
def reset_aperture():
    global iface
    if request.method == 'POST':
        if iface.cameraslm is not None:
            phase_mgr = iface.slm.phase_mgr
            phase_mgr.reset_aperture()
            print("Aperture Reset")
  
        else:
            print("Select a camera and an SLM")

        return redirect(url_for('phase_summary'))
    
    return redirect(url_for('phase_summary'))

###################################################################################################


@app.route('/use_add_phase', methods=['GET', 'POST'])
def use_add_phase():
    global directory, additional_load_history, iface

    if request.method == 'POST':
        if iface.cameraslm is not None:

       
            fname = request.form['fname']

            path = os.path.join(directory, 'data', 'additional', fname)

            # Add additional phase pattern to phase manager
            phase_mgr = iface.slm.phase_mgr
            phase_mgr.add_from_file(path)

            # Get the time the file was uploaded
            upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Add the file name and upload time to the history
            additional_load_history.append({'fname': path, 'upload_time': upload_time})

            print("Additional Phase Added from:" + path)
        else:
            print("Select a camera and an SLM")

        return redirect(url_for('load_from_file'))
    
    return redirect(url_for('load_from_file'))


###################################################################################################


@app.route('/add_pattern_to_add_phase', methods=['POST'])
def add_pattern_to_add_phase():
    global directory, additional_load_history, iface
    if request.method == 'POST':
        if iface.cameraslm is not None:

            fname = request.form['fname']

            # Add pattern path if its not global
            path = os.path.join(directory, 'data', 'additional', fname)

            # Add the additional phase pattern
            phase_mgr = iface.slm.phase_mgr
            phase_mgr.add_pattern_to_additional(path)

            # Get the time the file was uploaded
            upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Add the file name and upload time to the history
            additional_load_history.append({'fname': path, 'upload_time': upload_time})

            print("Base Pattern Added as Additional Phase:" + path)

        else:
            print("Select a camera and an SLM")

        return redirect(url_for('load_from_file'))
    
    return redirect(url_for('load_from_file'))


###################################################################################################


@app.route('/save_add_phase', methods=['GET', 'POST'])
def save_add_phase():
    global directory, additional_save_history, iface

    if request.method == 'POST':
        if iface.cameraslm is not None:
            # Get the file name from user
            save_name = request.form['save_name']
            # Add pattern path if its not an absolute path
            save_path = os.path.join(directory, 'data', 'additional')
        
            # Dictionary containing save options
            save_options = dict()
            save_options["config"] = True # This option saves the information about how this additional phase was created
            save_options["phase"] = True # saves the actual phase
            save_options["path"] = save_path # Enable this to save to a desired path. By default it is the current working directory
            save_options["name"] = save_name # This name will be used in the path.

            # Save additional phase pattern to new file
            phase_mgr = iface.slm.phase_mgr
            config_path, saved_additional_path = phase_mgr.save_to_file(save_options)

            # Get the time the file was uploaded
            upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Add the file name and upload time to the history
            additional_save_history.append({'fname': save_path, 'upload_time': upload_time})

            print("Saved additional phase to: " + saved_additional_path)

        else:
            print("Select a camera and an SLM")

        return redirect(url_for('load_from_file'))
    return redirect(url_for('load_from_file'))


###################################################################################################


@app.route('/correction', methods=['POST'])
def correction():
    global directory, additional_load_history, iface

    if request.method == 'POST':
        if iface.cameraslm is not None:
        
            fname = request.form['fname']

            path = os.path.join(directory, 'data', 'manufacturer', fname)

            phase_mgr = iface.slm.phase_mgr
            phase_mgr.add_correction(path, iface.slm_settings['bitdepth'], 1)

            # Get the time the file was uploaded
            upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Add the file name and upload time to the history
            additional_load_history.append({'fname': path, 'upload_time': upload_time})

            print("Added Manufacturer Correction from: " + path)
        else:
            print("Select a camera and an SLM")
        return redirect(url_for('load_from_file'))
    return redirect(url_for('load_from_file'))


###################################################################################################

@app.route('/input_additional', methods=['GET', 'POST'])
def input_additional():

    return render_template('input_additional.html')

###################################################################################################

@app.route('/add_fresnel_lens', methods=['POST'])
def add_fresnel_lens():
    global iface
    if request.method == 'POST':
        if iface.cameraslm is not None:

            # Got focal length from user
            focal_length = float(request.form['focal_length'])
            # Store focal length in a 1D numpy array
            focal_length = np.array([focal_length])

            # Add the fresnel lens
            phase_mgr = iface.slm.phase_mgr
            phase_mgr.add_fresnel_lens(focal_length[0])
            print("Added fresnel lens with focal length: " + str(focal_length[0]))

        else:
            print("Select a camera and an SLM")

        return redirect(url_for('input_additional'))
    
    return redirect(url_for('input_additional'))

###################################################################################################

@app.route('/add_offset', methods=['POST'])
def add_offset():
    global iface
    if request.method == 'POST':
        if iface.cameraslm is not None:

            # Get x,y coordinates for the offset
            offset_x = float(request.form['offset_x'])
            offset_y = float(request.form['offset_y'])
            # Store offset coords in a 1D numpy array
            offset = np.array([offset_x, offset_y])

            # Add the offset
            phase_mgr = iface.slm.phase_mgr
            phase_mgr.add_offset(offset)

            print("Added Offset: " + str(offset))
            
        else:
            print("Select a camera and an SLM")

        return redirect(url_for('input_additional'))
    
    return redirect(url_for('input_additional'))

###################################################################################################

@app.route('/add_zernike_poly', methods=['POST'])
def add_zernike_poly():
    global iface
    if request.method == 'POST':
        if iface.cameraslm is not None:
 
            # Get the number of zernikes in the sum
            npolys = (len(request.form)) // 3

            # Initialize list of zernikes
            poly_list = []

            # Loop over zernikes and append to the list
            for i in range(npolys):
                n = int(request.form.get(f'n{i}'))
                m = int(request.form.get(f'm{i}'))
                weight = float(request.form.get(f'weight{i}'))
                poly_list.append(((n, m), weight))

            # Add the sum of zernikes
            phase_mgr = iface.slm.phase_mgr
            phase_mgr.add_zernike_poly(poly_list)
            print("Added Zernike Polynomials: " + str(poly_list))
 
        else:
            print("Select a camera and an SLM")

        return redirect(url_for('input_additional'))
    return redirect(url_for('input_additional'))

###################################################################################################

@app.route('/use_aperture', methods=['POST'])
def use_aperture():
    global iface
    if request.method == 'POST':
        if iface.cameraslm is not None:

            # Get the aperture size from the user
            aperture_size = float(request.form['aperture_size'])
            # Store twice in a 1D numpy array
            aperture = np.array([aperture_size, aperture_size])

            # Set the aperture
            phase_mgr = iface.slm.phase_mgr
            phase_mgr.set_aperture(aperture)

            print("Added Aperture of Size: " + str(aperture_size))
            
        else:
            print("Select a camera and an SLM")

        return redirect(url_for('input_additional'))
    
    return redirect(url_for('input_additional'))     

###################################################################################################
###################################################################################################
###################################################################################################

@app.route('/config', methods=['GET', 'POST'])
def config():

    return render_template('config.html', 
                           config_load_history=config_load_history, 
                           config_save_history=config_save_history)

###################################################################################################


@app.route('/load_config', methods=['POST'])
def load_config():
    global directory, config_load_history, iface
    if request.method == 'POST':
        if iface.cameraslm is not None:
            phase_mgr = iface.slm.phase_mgr
            
            #filename = request.files['filename'].filename
            filename = request.form['filename']
            filepath = os.path.join(directory, 'data', 'config', filename)

            with open(filepath, 'r') as fhdl:
                config = yaml.load(fhdl, Loader=yaml.FullLoader)
            
            for key in config:
                if key == "pattern":
                    path = config["pattern"]
                    load_base(path)
                #elif key == "fourier_calibration":
                    #send_load_fourier_calibration(config["fourier_calibration"])
                elif key.startswith("file_correction"):
                    fname = config[key] 
                    phase_mgr.add_correction(fname, iface.slm_settings['bitdepth'], 1)
                    
                elif key.startswith("fresnel_lens"):
                    focal_length = np.array([ast.literal_eval(config[key])])
                    print(str(focal_length))
                    if len(focal_length) == 1:
                        phase_mgr.add_fresnel_lens(focal_length[0])
                    else:
                        phase_mgr.add_fresnel_lens(focal_length)
                elif key == "zernike":
                    res = ast.literal_eval(config["zernike"])
                    new_list = []
                    for item in res:
                        new_list.append(((item[0][0], item[0][1]), item[1]))
                    phase_mgr.add_zernike_poly(new_list)
                elif key == "offset":
                    offset = np.array(ast.literal_eval(config[key]))
                    phase_mgr.add_offset(offset)

            # Get the time the file was uploaded
            upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Add the file name and upload time to the history
            config_load_history.append({'fname': filepath, 'upload_time': upload_time})

            print("Config Loaded from: " + filepath)

        else:
            print("Select a camera and an SLM")

        return redirect(url_for('config'))
    return redirect(url_for('config'))


###################################################################################################


@app.route('/save_config', methods=['POST'])
def save_config():
    global directory, config_save_history, iface
    if request.method == 'POST':
        if iface.cameraslm is not None:
            #current_slm_settings = slm_list[slm_num]
            phase_mgr = iface.slm.phase_mgr

            config_dict = dict()
            base_str = phase_mgr.base_source
            
            if base_str != "":
                config_dict["pattern"] = base_str

            rep = ""
            log = phase_mgr.add_log
            for item in log:
                rep = rep + str(item[0]) + ";" + str(item[1]) + ";"
            
            add_str = rep

            corrections = add_str.split(';')
            correction_pattern_idx = 0
            file_idx = 0
            fresnel_lens_idx = 0
            zernike_idx = 0
            offset_idx = 0
            for i in range(int(np.floor(len(corrections)/2))):
                this_key = corrections[2 * i]
                this_val = corrections[2 * i + 1]
                if this_key == 'file_correction':
                    if correction_pattern_idx > 0:
                        config_dict[this_key + str(correction_pattern_idx)] = this_val
                    else:
                        config_dict[this_key] = this_val
                    correction_pattern_idx += 1
                elif this_key == "file":
                    if file_idx > 0:
                        config_dict[this_key + str(file_idx)] = this_val
                    else:
                        config_dict[this_key] = this_val
                    file_idx += 1
                elif this_key == 'fresnel_lens':
                    if fresnel_lens_idx > 0:
                        config_dict[this_key + str(fresnel_lens_idx)] = this_val
                    else:
                        config_dict[this_key] = this_val
                    fresnel_lens_idx += 1
                elif this_key == "zernike":
                    if zernike_idx > 0:
                        config_dict[this_key + str(zernike_idx)] = this_val
                    else:
                        config_dict[this_key] = this_val
                    zernike_idx += 1
                elif this_key == "offset":
                    if offset_idx > 0:
                        config_dict[this_key + str(offset_idx)] = this_val
                    else:
                        config_dict[this_key] = this_val
                    offset_idx += 1

            save_name = request.form['save_name']
            path = os.path.join(directory, 'data', 'config', save_name)
            with open(path, 'x') as fhdl:
                yaml.dump(config_dict, fhdl)

            # Get the time the file was uploaded
            upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Add the file name and upload time to the history
            config_save_history.append({'fname': path, 'upload_time': upload_time})

            print("Config Saved to: " + path)

        else:
            print("Select a camera and an SLM")

        return redirect(url_for('config'))
    return redirect(url_for('config'))


###################################################################################################
###################################################################################################
###################################################################################################

"""
@app.route('/camera', methods=['GET', 'POST'])
def camera():
    return render_template('camera.html')
"""

###################################################################################################


@app.route('/get_image', methods=['POST'])
def get_image():
    global iface, directory
    if request.method == 'POST':
        if iface.cameraslm is not None:
            if iface.camera_settings['camera_type'] != 'virtual':
                img = iface.camera.get_image(attempts=20)
                plt.figure(figsize=(24, 12))
                plt.imshow(img)
                path = os.path.join(directory, 'static', 'images', 'cam_img.png')
                plt.savefig(path)
                plt.close()
            else:
                print("Cannot get image from virtual camera")
        else:
            print("Select a camera and an SLM")

        return redirect(url_for('dashboard'))
    
    return redirect(url_for('dashboard'))


###################################################################################################

@app.route('/set_exposure', methods=['POST'])
def set_exposure():
    global iface, directory
    if request.method == 'POST':
        if iface.cameraslm is not None:
            if iface.camera_settings['camera_type'] != 'virtual':
                exposure = float(request.form['exposure'])
                iface.camera.set_exposure(exposure)
            else:
                print("Cannot get image from virtual camera")
        else:
            print("Select a camera and an SLM")

        return redirect(url_for('dashboard'))
    
    return redirect(url_for('dashboard'))

###################################################################################################

"""
@app.route('/fourier_calibrate', methods=['POST'])
def fourier_calibrate():
    global iface
    if request.method == 'POST' and iface.cameraslm is not None:
        iface.perform_fourier_calibration()

        return redirect(url_for('camera'))
    return redirect(url_for('camera'))
"""

###################################################################################################

if __name__ == '__main__':
    flask_thread = threading.Thread(target=start_flask_app, daemon=True)
    flask_thread.start()
    start_pyglet_app()
