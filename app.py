from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_socketio import SocketIO
import os
import Interface
from slmsuite.hardware.slms.screenmirrored import ScreenMirrored
import slmsuite.hardware.slms.slm
import PhaseManager
import CorrectedSLM
#import CameraClient
from slmsuite.hardware.cameras.thorlabs import ThorCam
from slmsuite.hardware.cameras.camera import Camera
import utils
#import re
import numpy as np
import yaml
import ast
import screeninfo
import mss.tools
import datetime
import pyglet
import threading

###################################################################################################
app = Flask(__name__)
#app.secret_key = os.urandom(24)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app)

def start_flask_app():
    print("Starting Flask App")
    socketio.run(app, host="0.0.0.0", port=8080, debug=False)

# History Lists TODO: make this a database
base_load_history = []
additional_load_history = []
additional_save_history = []
config_load_history = []
config_save_history = []

#main_path = None
directory = os.getcwd()

###################################################################################################

# Pyglet App
def start_pyglet_app():
    #initial_window = pyglet.window.Window(visible=True)
    #event_logger = pyglet.window.event.WindowEventLogger()
    #initial_window.push_handlers(event_logger)
    print("Starting Pyglet App")
    #pyglet.app.run()
    while True:
        pyglet.clock.tick()

        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            #window.dispatch_event('on_draw')
            #window.flip()

class SLMEventDispatcher(pyglet.event.EventDispatcher):
    def create_slm(self):
        self.dispatch_event('on_create_slm')

    def project_pattern(self):
        self.dispatch_event('on_project_pattern')

SLMEventDispatcher.register_event_type('on_create_slm')
SLMEventDispatcher.register_event_type('on_project_pattern')
dispatcher = SLMEventDispatcher()

###################################################################################################

computational_space = (2048, 2048)
n_iterations = 20

@app.route('/setup_calculation', methods=['GET', 'POST'])
def setup_calculation():
    global main_path, computational_space, n_iterations

    if request.method == 'POST':

        """
        new_main_path = request.form['main_path']
        if new_main_path:
            main_path = str(new_main_path)
            print("Updated Pattern Path to: " + main_path)
            flash("Updated Pattern Path to: " + main_path)
        """

        size = request.form['computational_space']
        if size:
            computational_space = (int(size), int(size))
            print("Updated Computational Space to: " + str(computational_space))
            #flash("Updated Computational Space to: " + str(computational_space))

        new_n_iterations = request.form['n_iterations']
        if new_n_iterations:
            n_iterations = int(new_n_iterations)
            print("Updated Number of Iterations to: " + str(n_iterations))
            #flash("Updated Number of Iterations to: " + str(n_iterations))
        
        return redirect(url_for('setup_calculation'))
    
    return render_template('setup_calculation.html')

###################################################################################################

A = np.array([[-1.3347, 0.89309], [0.89306, 1.3414]])
b = np.array([[1254.758], [277.627]])

@app.route('/setup_calibration', methods=['POST'])
def setup_calibration():
    global A, b
    if request.method == "POST":
        top_left = float(request.form['top_left'])
        top_right = float(request.form['top_right'])
        bottom_left = float(request.form['bottom_left'])
        bottom_right = float(request.form['bottom_right'])

        vector_x = float(request.form['vector_x'])
        vector_y = float(request.form['vector_y'])

        A = np.array([[top_left, top_right], [bottom_left, bottom_right]])
        b = np.array([[vector_x], [vector_y]])

        print("Updated calibration to: " + str(A) + "+ " + str(b))

        return redirect(url_for('setup_calculation'))
    
    return redirect(url_for('setup_calculation'))

###################################################################################################

# List of dictionaries with SLM settings
slm_list = []

# Dictionary of SLM settings
setup_slm_settings = {}

@app.route('/setup_slm', methods=['GET', 'POST'])
def setup_slm():
    global setup_slm_settings

    if request.method == 'POST':
        
        # Get SLM settings from user
        display_num = int(request.form['display_num'])
        bitdepth = int(request.form['bitdepth'])
        wav_design_um = float(request.form['wav_design_um'])
        wav_um = float(request.form['wav_um'])

        # Add them to settings dictionary
        setup_slm_settings['display_num'] = display_num
        setup_slm_settings['bitdepth'] = bitdepth
        setup_slm_settings['wav_design_um'] = wav_design_um
        setup_slm_settings['wav_um'] = wav_um
        
        # Call to create pyglet window
        pyglet.clock.schedule_once(lambda dt: dispatcher.create_slm(), 0)

        return redirect(url_for('setup_slm'))

    return render_template('setup_slm.html')

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

    print("Succesfully setup SLM on display: " + str(setup_slm_settings['display_num']))
    #create_slm_function(slm)

@app.route('/setup_virtual_slm', methods=['GET'])
def setup_virtual_slm():
    global setup_slm_settings, slm_list

    if request.method == 'GET':
        setup_slm_settings['display_num'] = "virtual"
        setup_slm_settings['bitdepth'] = None
        setup_slm_settings['wav_design_um'] = None
        setup_slm_settings['wav_um'] = None

        # Create a virtual slm
        slm = slmsuite.hardware.slms.slm.SLM(1272, 1024)

        # Create phase manager for the slm
        phase_mgr = PhaseManager.PhaseManager(slm)

        # Wrap together slm and phase manager
        wrapped_slm = CorrectedSLM.CorrectedSLM(slm, phase_mgr)

        # Add the wrapped slm to the settings
        setup_slm_settings['slm'] = wrapped_slm

        # Add settings dictionary to the list
        slm_list.append(setup_slm_settings.copy())
        #create_slm_function()

        print("Succesfully setup SLM on display: " + str(setup_slm_settings['display_num']))

        return redirect(url_for('setup_slm'))

    return redirect(url_for('setup_slm'))

"""
def create_slm_function(slm=None):
    global setup_slm_settings, slm_list

    #iface = Interface.SLMSuiteInterface()

    if slm is None:
        slm = iface.set_SLM()

    phase_mgr = PhaseManager.PhaseManager(slm)
    wrapped_slm = CorrectedSLM.CorrectedSLM(slm, phase_mgr)
    iface.set_SLM(wrapped_slm)
    #iface.set_camera()

    setup_slm_settings['iface'] = iface
    setup_slm_settings['phase_mgr'] = phase_mgr

    slm_list.append(setup_slm_settings.copy())

    print("Succesfully setup SLM on display: " + str(setup_slm_settings['display_num']))
"""

###################################################################################################
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
        if camera_type == "thorlabs":

            # Connect to the camera
            camera = ThorCam(serial=serial_num, fliplr=fliplr)

            # Add them to the settings dictionary
            setup_camera_settings['camera_type'] = camera_type
            setup_camera_settings['serial_num'] = serial_num
            setup_camera_settings['fliplr'] = fliplr
            setup_camera_settings['camera'] = camera

            # Add settings to the list of cameras
            camera_list.append(setup_camera_settings.copy())

            print("Succesfully setup Camera of type: " + setup_camera_settings['camera_type'])
        else:
            print("Camera Type Not Setup Yet")

        return redirect(url_for('setup_camera'))

    return render_template('setup_camera.html')

@app.route('/setup_virtual_camera', methods=['GET'])
def setup_virtual_camera():
    global setup_camera_settings, camera_list

    if request.method == 'GET':
        setup_camera_settings['camera_type'] = 'virtual'
        setup_camera_settings['serial_num'] = None
        setup_camera_settings['fliplr'] = None

        # Create virtual camera
        camera = Camera(1024, 1024)

        setup_camera_settings['camera'] = camera

        # Add settings to the list of cameras
        camera_list.append(setup_camera_settings.copy())

        print("Succesfully setup Camera of type: " + setup_camera_settings['camera_type'])

        return redirect(url_for('setup_camera'))

    return redirect(url_for('setup_camera'))
###################################################################################################
# Initialize the Interface
iface = Interface.SLMSuiteInterface()

@app.route('/setup_iface', methods=['POST'])
def setup_iface():
    global iface, slm_list, camera_list
    if request.method == 'POST':
        # Get user selected slm index
        slm_num = int(request.form['slm_num'])
        
        # Get corresponding slm settings
        slm_settings = slm_list[slm_num]

        # Extract the wrapped slm
        slm = slm_settings['slm']

        # Set SLM in iface
        iface.set_SLM(slm=slm, slm_settings=slm_settings)

        print("Selected SLM with display number: " + str(slm_settings['display_num']))

        # Get user selected camera index
        camera_num = int(request.form['camera_num'])

        # Get corresponsing camera settings
        camera_settings = camera_list[camera_num]

        # Extract camera
        camera = camera_settings['camera']

        # Set camera in iface
        iface.set_camera(camera, camera_settings=camera_settings)
        
        print("Selected camera of type: " + str(camera_settings['camera_type']) + " with serial number" + str(camera_settings['serial_num']))

        return redirect(url_for('dashboard'))
                
    return redirect(url_for('dashboard'))

###################################################################################################

@app.route('/setup_slm_amp', methods=['GET', 'POST'])
def setup_slm_amp():
    global iface

    if request.method == 'POST':
        if iface.slm is not None:
            func = request.form['func']

            if func == "gaussian":
                waist_x = float(request.form['waist_x'])
                waist_x = np.array([waist_x])
                waist_y = float(request.form['waist_y'])
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
                print("Unknown amp type")
        else:
            print('No SLM Selected')
        
        return redirect(url_for('setup_slm_amp'))
    
    return render_template('setup_slm_amp.html')

###################################################################################################
###################################################################################################
###################################################################################################

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    global iface, slm_list, camera_list
    if iface.cameraslm is not None:
        slm_settings = iface.slm_settings
        camera_settings = iface.camera_settings
        phase_info = get_phase_info()
        if slm_settings['display_num'] != 'virtual':
            get_screenshot()
    else:
        phase_info = None
        camera_settings = None
        slm_settings = None
        print("No SLM Connected")
     
    return render_template('dashboard.html', 
                           slm_list=slm_list,
                           camera_list=camera_list,
                           phase_info=phase_info,
                           slm_settings=slm_settings,
                           camera_settings=camera_settings,
                           iface=iface)

###################################################################################################

def get_phase_info():
    global iface, directory

    # Get the phase manager
    phase_mgr = iface.slm.phase_mgr

    # Get the file path of the base pattern
    base_str = phase_mgr.base_source

    remove_path = os.path.join(directory, 'data', 'base')
    base_str = base_str.replace(remove_path, "")

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

###################################################################################################

def get_screenshot():

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

###################################################################################################

@app.route('/project', methods=['POST'])
def project():
    global iface
    
    if request.method == 'POST' and iface.cameraslm is not None and not iface.slm_settings['display_num'] == 'virtual':
        pyglet.clock.schedule_once(lambda dt: dispatcher.project_pattern(), 0)
        print("Ran the HTTP route to project")
        return redirect(url_for('dashboard'))
    else:
        print("No Hardware SLM Connected")

    return redirect(url_for('dashboard'))

@dispatcher.event
def on_project_pattern():
    global iface

    phase_mgr = iface.slm.phase_mgr
    iface.write_to_SLM(phase_mgr.base, phase_mgr.base_source)
    
    print("Succesfully projected to display: " + str(iface.slm_settings['display_num']))

###################################################################################################
    
@app.route('/display_targets_dashboard')
def display_targets_dashboard():
    global iface

    if iface.cameraslm is not None:
        phase_mgr = iface.slm.phase_mgr

        if phase_mgr.base_source:
    
            #targets = utils.get_target_from_file(phase_mgr.base_source)
            #x_coords = targets[0].tolist()
            #y_coords = targets[1].tolist()
            _,data = utils.load_slm_calculation(phase_mgr.base_source, 0, 1)
            input_targets = data['input_targets']
            x_coords = input_targets[0].tolist()
            y_coords = input_targets[1].tolist()
            labels = list(range(1, len(x_coords) + 1))

            print("Displaying targets from: " + phase_mgr.base_source)
            #flash("Displaying targets from: " + phase_mgr.base_source)

            return jsonify({'x': x_coords, 'y': y_coords, 'labels': labels})
    else:
        print("No SLM Selected")



###################################################################################################
###################################################################################################
###################################################################################################

@app.route('/base_pattern', methods=['GET', 'POST'])
def base_pattern():
    global base_load_history, computational_space, n_iterations, A, b

    return render_template('base_pattern.html', 
                           base_load_history=base_load_history,
                           computational_space=computational_space, 
                           n_iterations=n_iterations,
                           A=A,
                           b=b)

###################################################################################################

@app.route('/use_pattern', methods=['POST'])
def use_pattern():
    global directory

    if request.method == 'POST':

        # Get the file name input by the user
        #file = request.files['fname']
        #fname = file.filename[:-9]
        fname = request.form['fname']

        # Add the pattern path (if it is just a file name)
        path = os.path.join(directory, 'data', 'base', fname)
        #print("Loading Base Pattern from: " + path)

        load_base(path)

        return redirect(url_for('base_pattern'))
    
    return render_template('base_pattern')

def load_base(path):
    global base_load_history, iface, computational_space, socketio

    if iface.cameraslm is not None:

        # Get the phase pattern from the file
        _,data = utils.load_slm_calculation(path, 0, 1)
        phase = data["slm_phase"]

        phase_mgr = iface.slm.phase_mgr
        # Set the phase pattern as the base of the phase manager
        phase_mgr.set_base(phase, path)
        print("Base Pattern added from: " + path)

        input_targets = data['input_targets']
        x_coords = input_targets[0].tolist()
        y_coords = input_targets[1].tolist()
        x_coords = list(map(float, x_coords))
        y_coords = list(map(float, y_coords))
        targets = np.array([x_coords, y_coords])

        amplitudes = data['input_amplitudes']
        # Convert amplitudes to floats
        amplitudes = list(map(float, amplitudes))
        # Create 1D numpy array containing amplitudes
        amp_data = np.array(amplitudes)
        amp_data_for_input = np.copy(amp_data)

        print("Received amplitudes: " + str(amp_data))
        
        iface.input_amplitudes = amp_data_for_input
        iface.input_targets = targets

        iface.set_hologram(computational_shape=computational_space, target_spot_array=targets, target_amps=amp_data, socketio=socketio)
        iface.plot_farfield(plot_target=True)

        # Get the time the file was uploaded
        upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add the file name and upload time to the history
        base_load_history.append({'fname': path, 'upload_time': upload_time})
    else:
        print("No SLM Connected")

###################################################################################################

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

@app.route('/manual', methods=['GET', 'POST'])
def manual():
    global iface, computational_space, socketio

    if request.method == 'POST':
        
        x_coords = request.form.getlist('x_coords')
        y_coords = request.form.getlist('y_coords')
        x_coords = list(map(float, x_coords))
        y_coords = list(map(float, y_coords))
        targets = np.array([x_coords, y_coords])

        camera = request.form['camera']
        if camera == 'yes':
            targets = andor_to_k(targets)
            print("Converted to camera coords")

        print("Received Targets:" + str(targets))

        amplitudes = request.form.getlist('amplitudes')

        # Convert amplitudes to floats
        amplitudes = list(map(float, amplitudes))
        # Create 1D numpy array containing amplitudes
        amp_data = np.array(amplitudes)
        amp_data_for_input = np.copy(amp_data)

        print("Received amplitudes: " + str(amp_data))
        
        iface.input_amplitudes = amp_data_for_input
        iface.input_targets = targets

        iface.set_hologram(computational_shape=computational_space, target_spot_array=targets, target_amps=amp_data, socketio=socketio)
        iface.plot_farfield(plot_target=True)

        #calculate_function(x_coords, y_coords, amplitudes, iteration_number, camera, guess_name, save_name)
        return redirect(url_for('manual'))
    
    return render_template('manual.html')

###################################################################################################

@app.route('/lattice_box', methods=['GET', 'POST'])
def lattice_box():
    global iface, computational_space, socketio

    if request.method == 'POST':

        lv11 = float(request.form['lv11'])
        lv12 = float(request.form['lv12'])
        lv21 = float(request.form['lv21'])
        lv22 = float(request.form['lv22'])
        offset_x = float(request.form['offset_x'])
        offset_y = float(request.form['offset_y'])
        width = float(request.form['width'])
        height = float(request.form['height'])
        center_x = float(request.form['center_x'])
        center_y = float(request.form['center_y'])

        #iteration_number = request.form['iteration_number']
        camera = request.form['camera']
        #guess_name = request.form['guess_name']
        #save_name = request.form['save_name']
        edge_buffer = int(request.form['edge_buffer'])

        lattice_vectors = [
            np.array([lv11, lv21]),
            np.array([lv12, lv22])]
        
        image_shape = (width, height)

        offset = (offset_x, offset_y)

        center_pix = (center_x, center_y)

        spots, x_coords, y_coords = generate_lattice(image_shape, lattice_vectors, offset, center_pix, edge_buffer)

        num_spots = len(x_coords)

        amplitudes = np.ones(num_spots, dtype = float)

        x_coords = list(map(float, x_coords))
        y_coords = list(map(float, y_coords))
        targets = np.array([x_coords, y_coords])

        camera = request.form['camera']
        if camera == 'yes':
            targets = andor_to_k(targets)
            print("Converted to camera coords")

        print("Received Targets:" + str(targets))

        # Convert amplitudes to floats
        amplitudes = list(map(float, amplitudes))
        # Create 1D numpy array containing amplitudes
        amp_data = np.array(amplitudes)
        amp_data_for_input = np.copy(amp_data)

        print("Received amplitudes: " + str(amp_data))
        
        iface.input_amplitudes = amp_data_for_input
        iface.input_targets = targets
        
        iface.set_hologram(computational_shape=computational_space, target_spot_array=targets, target_amps=amp_data, socketio=socketio)
        iface.plot_farfield(plot_target=True)
        #calculate_function(x_coords, y_coords, amplitudes, iteration_number, camera, guess_name, save_name)

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

@app.route('/canvas')
def canvas():
    return render_template('canvas.html')

@app.route('/grid', methods=['GET', 'POST'])
def grid():
    return render_template('grid.html')

@app.route('/submit_points', methods=['POST'])
def submit_points():
    global iface, computational_space, socketio

    data = request.json
    print(data)
    
    x_coords = data['Coords1']
    y_coords = data['Coords2']

    amplitudes = data['amplitudes']
    #iteration_number = data['iteration_number']
    #camera = data['camera']
    camera = 'yes'
    #guess_name = data['guess_name']
    #save_name = data['save_name']
    
    x_coords = list(map(float, x_coords))
    y_coords = list(map(float, y_coords))
    targets = np.array([x_coords, y_coords])

    #camera = request.form['camera']
    if camera == 'yes':
        targets = andor_to_k(targets)
        print("Converted to camera coords")

    print("Received Targets:" + str(targets))
    
    # Convert amplitudes to floats
    amplitudes = list(map(float, amplitudes))
    # Create 1D numpy array containing amplitudes
    amp_data = np.array(amplitudes)
    amp_data_for_input = np.copy(amp_data)

    print("Received amplitudes: " + str(amp_data))
    
    iface.input_amplitudes = amp_data_for_input
    iface.input_targets = targets
    
    iface.set_hologram(computational_shape=computational_space, target_spot_array=targets, target_amps=amp_data, socketio=socketio)
    iface.plot_farfield(plot_target=True)
    
    #calculate_function(x_coords, y_coords, amplitudes, iteration_number, camera, guess_name, save_name)
    return jsonify({'status': 'success'})

###################################################################################################

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    global directory
    #TODO: simply load in directly from iface attributes

    if request.method == 'POST':

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
        x_coords = []
        y_coords = []
        num_points = 0
    return render_template('feedback.html', x_coords=x_coords, y_coords=y_coords, num_points=num_points)

###################################################################################################

@app.route('/calculate_page', methods=['GET', 'POST'])
def calculate_page():
    return render_template('calculate.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    global n_iterations, computational_space, directory, iface

    if iface.hologram is not None:
        #current_slm_settings = slm_list[slm_num]
        #iface = current_slm_settings['iface']

        #x_coords = list(map(float, x_coords))
        #y_coords = list(map(float, y_coords))
        #targets = np.array([x_coords, y_coords])

        #if camera == "yes":
            #targets = andor_to_k(targets)
            #print("Converted to camera coords")

        #print("Received Targets:" + str(targets))
        
        # Convert amplitudes to floats
        #amplitudes = list(map(float, amplitudes))
        # Create 1D numpy array containing amplitudes
        #amp_data = np.array(amplitudes)
        #amp_data_for_input = np.copy(amp_data)

        #print("Received amplitudes: " + str(amp_data))

        #iface.input_amplitudes = amp_data_for_input
        #iface.input_targets = targets

        # If user specified nothing, set to default

        data = request.json
        print(data)

        iteration_number = int(data['iteration_number'])
        if not iteration_number:
            iteration_number = n_iterations
        
        # Initialize guess phase
        guess_phase = None

        guess_name = data['guess_name']
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
        #iface.calculate(computational_space, targets, amp_data, n_iters=int(iteration_number), phase=guess_phase, socketio=socketio)

        iface.optimize(iteration_number)

        iface.plot_slmplane()
        iface.plot_farfield()
        iface.plot_stats()

        save_name = data['save_name']
        saved_pattern_path = save_calculation(save_name)[:-9]
        #TODO: probably an easier way to extract the phase pattern
        #load_base(saved_pattern_path)
        print("Finished Calculation, Save and Load")
        
    else:
        print("No SLM Selected")
        
    return jsonify({'status': 'success'})
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

    if iface.slm is not None and request.method == "POST":
        
        phase_mgr = iface.slm.phase_mgr
        phase_mgr.reset_base()
        print("Reset Base Pattern")

        return redirect(url_for('base_pattern'))

    return redirect(url_for('base_pattern'))

###################################################################################################

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

###################################################################################################
###################################################################################################
###################################################################################################

@app.route('/additional_pattern', methods=['GET', 'POST'])
def additional_pattern():
    global additional_load_history, additional_save_history
    return render_template('additional_pattern.html', 
                           additional_load_history = additional_load_history,
                           additional_save_history=additional_save_history)

@app.route('/reset_additional_phase', methods=['POST'])
def reset_additional_phase():
    global slm_list, slm_num
    if slm_num is not None and request.method == 'POST':
        current_slm_settings = slm_list[slm_num]
        # Reset the additional phase pattern
        phase_mgr = current_slm_settings['phase_mgr']
        phase_mgr.reset_additional()
        print("Reset Additional Phase")

        return redirect(url_for('additional_pattern'))
    else:
        print("No SLM Selected")

    return redirect(url_for('additional_pattern'))

@app.route('/reset_aperture', methods=['POST'])
def reset_aperture():
    global slm_list, slm_num
    if slm_num is not None and request.method == 'POST':
        current_slm_settings = slm_list[slm_num]
        # Reset the aperture
        phase_mgr = current_slm_settings['phase_mgr']
        phase_mgr.reset_aperture()
        print("Aperture Reset")

        return redirect(url_for('additional_pattern'))
    else:
        print("No SLM Selected")

    return redirect(url_for('additional_pattern'))

@app.route('/use_add_phase', methods=['GET', 'POST'])
def use_add_phase():
    global slm_list, slm_num, main_path, directory, additional_load_history
    if slm_num is not None and request.method == 'POST':
        current_slm_settings = slm_list[slm_num]

        # Get file name input by user
        #file = request.files['fname']
        #fname = file.filename[:-19]
        fname = request.form['fname']

        path = os.path.join(directory, 'data', 'additional', fname)

        # Add additional phase pattern to phase manager
        phase_mgr = current_slm_settings['phase_mgr']
        phase_mgr.add_from_file(path)

        # Get the time the file was uploaded
        upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add the file name and upload time to the history
        additional_load_history.append({'fname': path, 'upload_time': upload_time})

        print("Additional Phase Added from:" + path)

        return redirect(url_for('additional_pattern'))
    else:
        print("No SLM Selected")
    return redirect(url_for('additional_pattern'))

@app.route('/add_pattern_to_add_phase', methods=['POST'])
def add_pattern_to_add_phase():
    global slm_list, slm_num, main_path, directory, additional_load_history
    if slm_num is not None and request.method == 'POST':
        current_slm_settings = slm_list[slm_num]

        # Get file path for additional phase from user
        #file = request.files['path']
        #fname = file.filename[:-19]
        fname = request.form['fname']

        # Add pattern path if its not global
        path = os.path.join(directory, 'data', 'additional', fname)

        # Add the additional phase pattern
        phase_mgr = current_slm_settings['phase_mgr']
        phase_mgr.add_pattern_to_additional(path)

        # Get the time the file was uploaded
        upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add the file name and upload time to the history
        additional_load_history.append({'fname': path, 'upload_time': upload_time})

        print("Additional Phase Added from:" + path)

        return redirect(url_for('additional_pattern'))
    else:
        print("No SLM Selected")
    return redirect(url_for('additional_pattern'))

@app.route('/save_add_phase', methods=['GET', 'POST'])
def save_add_phase():
    global slm_list, slm_num, main_path, directory, additional_save_history
    if slm_num is not None and request.method == 'POST':
        current_slm_settings = slm_list[slm_num]

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
        phase_mgr = current_slm_settings['phase_mgr']
        config_path, saved_additional_path = phase_mgr.save_to_file(save_options)

        # Get the time the file was uploaded
        upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add the file name and upload time to the history
        additional_save_history.append({'fname': save_path, 'upload_time': upload_time})

        print("Saved additional phase to: " + saved_additional_path)

        return redirect(url_for('additional_pattern'))
    else:
        print("No SLM Selected")
    return redirect(url_for('additional_pattern'))

@app.route('/correction', methods=['POST'])
def correction():
    global slm_list, slm_num, main_path, directory, additional_load_history
    if slm_num is not None and request.method == 'POST':
        current_slm_settings = slm_list[slm_num]
        # Get correction file name from user
        #file = request.files['fname']
        #fname = file.filename
        fname = request.form['fname']

        path = os.path.join(directory, 'data', 'manufacturer', fname)

        phase_mgr = current_slm_settings['phase_mgr']
        phase_mgr.add_correction(path, current_slm_settings['bitdepth'], 1)

        # Get the time the file was uploaded
        upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add the file name and upload time to the history
        additional_load_history.append({'fname': path, 'upload_time': upload_time})

        print("Added Manufacturer Correction from: " + path)
        """
        # Check if connected to hardware
        if  slm_settings['slm_type'] == "hamamatsu":
            # Add correction phase pattern
            phase_mgr.add_correction(fname, slm_settings['bitdepth'], 1)
        else:
            phase_mgr.add_correction(fname, slm_settings['bitdepth'], 1) #TODO, in case you need to scale.
            #TODO: ask what this is for?
        """

        return redirect(url_for('additional_pattern'))
    else:
        print("No SLM Connected")

    return redirect(url_for('additional_pattern'))

@app.route('/input_additional', methods=['GET', 'POST'])
def input_additional():

    return render_template('input_additional.html')

@app.route('/add_fresnel_lens', methods=['POST'])
def add_fresnel_lens():
    global slm_list, slm_num
    if slm_num is not None and request.method == 'POST':
        current_slm_settings = slm_list[slm_num]
        #TODO: two focal lengths

        # Got focal length from user
        focal_length = float(request.form['focal_length'])
        # Store focal length in a 1D numpy array
        focal_length = np.array([focal_length])

        # Add the fresnel lens
        phase_mgr = current_slm_settings['phase_mgr']
        phase_mgr.add_fresnel_lens(focal_length[0])
        print("Added fresnel lens with focal length: " + str(focal_length[0]))

        return redirect(url_for('input_additional'))
    else:
        print("No SLM Selected")
    return redirect(url_for('input_additional'))

@app.route('/add_offset', methods=['POST'])
def add_offset():
    global slm_list, slm_num
    if slm_num is not None and request.method == 'POST':
        current_slm_settings = slm_list[slm_num]

        # Get x,y coordinates for the offset
        offset_x = float(request.form['offset_x'])
        offset_y = float(request.form['offset_y'])
        # Store offset coords in a 1D numpy array
        offset = np.array([offset_x, offset_y])

        # Add the offset
        phase_mgr = current_slm_settings['phase_mgr']
        phase_mgr.add_offset(offset)

        print("Added Offset: " + str(offset))
        return redirect(url_for('input_additional'))
    else:
        print("No SLM Selected")

    return redirect(url_for('input_additional'))

@app.route('/add_zernike_poly', methods=['POST'])
def add_zernike_poly():
    global slm_list, slm_num
    if slm_num is not None and request.method == 'POST':
        current_slm_settings = slm_list[slm_num]
 
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
        phase_mgr = current_slm_settings['phase_mgr']
        phase_mgr.add_zernike_poly(poly_list)
        print("Added Zernike Polynomials: " + str(poly_list))

        return redirect(url_for('input_additional'))
    else:
        print("No SLM Selected")
    return redirect(url_for('input_additional'))

@app.route('/use_aperture', methods=['POST'])
def use_aperture():
    global slm_list, slm_num
    if slm_num is not None and request.method == 'POST':
        current_slm_settings = slm_list[slm_num]

        # Get the aperture size from the user
        aperture_size = float(request.form['aperture_size'])
        # Store twice in a 1D numpy array
        #TODO: why twice, should the user be able to pass a second aperture size?
        aperture = np.array([aperture_size, aperture_size])

        # Set the aperture
        phase_mgr = current_slm_settings['phase_mgr']
        phase_mgr.set_aperture(aperture)

        print("Added Aperture of Size: " + str(aperture_size))
        return redirect(url_for('input_additional'))
    else:
        print("No SLM Selected")
    return redirect(url_for('input_additional'))     

@app.route('/config', methods=['GET', 'POST'])
def config():

    return render_template('config.html', 
                           config_load_history=config_load_history, 
                           config_save_history=config_save_history)

@app.route('/load_config', methods=['POST'])
def load_config():
    global slm_list, slm_num, main_path, directory, config_load_history
    if slm_num is not None and request.method == 'POST':
        current_slm_settings = slm_list[slm_num]
        phase_mgr = current_slm_settings['phase_mgr']
        
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
                phase_mgr.add_correction(fname, current_slm_settings['bitdepth'], 1)
                
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

        return redirect(url_for('config'))
    else:
        print("No SLM Selected")

    return redirect(url_for('config'))

@app.route('/save_config', methods=['POST'])
def save_config():
    global slm_list, slm_num, main_path, directory, config_save_history
    if slm_num is not None and request.method == 'POST':
        current_slm_settings = slm_list[slm_num]
        phase_mgr = current_slm_settings['phase_mgr']

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

        return redirect(url_for('config'))
    else:
        print("No SLM Selected")

    return redirect(url_for('config'))


if __name__ == '__main__':
    flask_thread = threading.Thread(target=start_flask_app, daemon=True)
    flask_thread.start()
    start_pyglet_app()
    

"""
@app.route('/setup_camera', methods=['GET', 'POST'])
def setup_camera():
    global iface
    global camera_settings

    if request.method == 'POST':
    
        camera_type = request.form['camera_type']

        if camera_type == "virtual":
            camera = iface.set_camera()
       
        elif camera_type == "network":
            url = request.form['camera_url']
            camera = CameraClient.CameraClient(url)
            iface.set_camera(camera)     

            camera_settings['camera_url'] = url

        elif camera_type == "thorcam_scientific_camera":
            serial_num = request.form['serial_num']
            if serial_num:
                serial = serial_num
            else:
                serial = ""
            camera = slmsuite.hardware.cameras.thorlabs.ThorCam(serial)
            iface.set_camera(camera)

            camera_settings['serial_num'] = serial_num
        else:
            print("Camera type not recognized")

        camera_settings['camera_type'] = camera_type

        print("Camera setup succesful")
        
        return redirect(url_for('setup_camera'))

    return render_template('setup_camera.html', camera_settings=camera_settings)
"""

#TODO: ask what this does?
"""
@app.route('/init_hologram', methods=['GET', 'POST'])
def init_hologram():
    global pattern_path
    global iface
    global computational_space

    if request.method == 'POST':
        path = request.form['path']
        if re.match(r'[A-Z]:', path) is None:
            # check to see if it's an absolute path
            path = pattern_path + path
        msg = iface.init_hologram(path, computational_space)
        print(msg)

        return redirect(url_for('init_hologram'))
    
    return render_template('init_hologram.html')
"""

#TODO: do we need to be able to get just the base or just the additional phase, or is just
# the current phase info enough?
"""
@app.route('/get_base', methods=['GET', 'POST'])
def get_base():
    global phase_mgr

    if request.method == 'POST':
        print(phase_mgr.base_source)

        return redirect(url_for('get_base'))
    
    return render_template('get_base.html')
"""

"""
@app.route('/get_additional_phase', methods=['GET', 'POST'])
def get_additional_phase():
    global phase_mgr

    if request.method == 'POST':
        rep = ""
        log = phase_mgr.add_log
        for item in log:
            rep = rep + str(item[0]) + ";" + str(item[1]) + ";"
        print(rep)

        return redirect(url_for('get_additional_phase'))
    
    return render_template('get_additional_phase.html')
"""

"""
@app.route('/get_displays', methods=['GET', 'POST'])
def get_displays():
    global displays

    if request.method == 'POST':
        displays = screeninfo.get_monitors()
        displays_info_list = []
        for index, display in enumerate(displays):
            # Get name if it exists
            if hasattr(display, 'name'):
                name = display.name
            else:
                name = 'Unknown'
            # Get width
            width = display.width
            # Get height
            height = display.height
            # Create list element with index, name and dimensions
            displays_info_list.append(f"Index: {index}, Name: {name}, Dimensions: {width}x{height}")
        
        # Join the list element into a single string, with a line break in between for each display
        displays_info_string = "\n".join(displays_info_list)
        print(displays_info_string)

        return redirect(url_for('get_displays'))
    
    return render_template('get_displays.html')
"""
    
"""
@app.route('/calculate_square_array', methods=['GET', 'POST'])
def calculate_square_array():
    global iface
    global pattern_path

    if request.method == 'POST':
        side_length = request.form['side_length']
        pixel_spacing = request.form['pixel_spacing']
        rot_angle = request.form['rot_angle']
        offset = request.form['offset']

        square_targets = utils.gen_square_targets(side_length, pixel_spacing, rot_angle, offset)

        targets = square_targets[0]
        amp_data = square_targets[1]

        iteration_number = request.form['iteration_number']

        phase_path = ""
        phase_path = request.form['phase_path']

        if not iteration_number:
            iteration_number = n_iterations
     
        if phase_path == "":
            iface.calculate(computational_space, targets, amp_data, n_iters=int(iteration_number))
        else:
            if re.match(r'[A-Z]:', phase_path) is None:
                # check to see if it's an absolute path
                phase_path = pattern_path + phase_path
            _,data = utils.load_slm_calculation(phase_path, 1, 1)
            slm_phase = None
            if "raw_slm_phase" in data:
                slm_phase = data["raw_slm_phase"]
            else:
                return "Cannot initiate the phase, since it was not saved"
            
            iface.calculate(computational_space, targets, amp_data, n_iters=int(iteration_number), phase=slm_phase)

        #self.iface.calculate(self.computational_space, targets, amp_data, n_iters=self.n_iterations)
        # for debug
        iface.plot_slmplane()
        iface.plot_farfield()
        iface.plot_stats()

        return redirect(url_for('calculate_square_array'))
    
    return render_template('calculate_square_array.html')
"""

"""
@app.route('/calculate_square_array2', methods=['GET', 'POST'])
def calculate_square_array2():
    global iface
    global pattern_path

    if request.method == 'POST':
        side_length = request.form['side_length']
        pixel_spacing = request.form['pixel_spacing']
        rot_angle = request.form['rot_angle']
        offset = request.form['offset']

        square_targets = utils.gen_square_targets2(side_length, pixel_spacing, rot_angle, offset)

        targets = square_targets[0]
        amp_data = square_targets[1]

        iteration_number = request.form['iteration_number']

        phase_path = ""
        phase_path = request.form['phase_path']

        if not iteration_number:
            iteration_number = n_iterations
     
        if phase_path == "":
            iface.calculate(computational_space, targets, amp_data, n_iters=int(iteration_number))
        else:
            if re.match(r'[A-Z]:', phase_path) is None:
                # check to see if it's an absolute path
                phase_path = pattern_path + phase_path
            _,data = utils.load_slm_calculation(phase_path, 1, 1)
            slm_phase = None
            if "raw_slm_phase" in data:
                slm_phase = data["raw_slm_phase"]
            else:
                return "Cannot initiate the phase, since it was not saved"
            
            iface.calculate(computational_space, targets, amp_data, n_iters=int(iteration_number), phase=slm_phase)

        #self.iface.calculate(self.computational_space, targets, amp_data, n_iters=self.n_iterations)
        # for debug
        iface.plot_slmplane()
        iface.plot_farfield()
        iface.plot_stats()

        return redirect(url_for('calculate_square_array2'))
    
    return render_template('calculate_square_array2.html')
"""
