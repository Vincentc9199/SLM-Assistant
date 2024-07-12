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

app = Flask(__name__)
app.secret_key = os.urandom(24)
socketio = SocketIO(app)

#window = pyglet.window.Window(visible=True)

def start_flask_app():
    print("Starting Flask app...")
    socketio.run(app, port=8080, debug=False)

def start_pyglet_app():
    pyglet.app.run()

class SLMEventDispatcher(pyglet.event.EventDispatcher):
    def create_slm(self):
        self.dispatch_event('on_create_slm')

    def project_pattern(self):
        self.dispatch_event('on_project_pattern')

SLMEventDispatcher.register_event_type('on_create_slm')
SLMEventDispatcher.register_event_type('on_project_pattern')
dispatcher = SLMEventDispatcher()

base_load_history = []
    
main_path = '/Users/vincentcosta/Documents/Summer_Research/SLMdata/'

computational_space = (2048, 2048)
n_iterations = 20

slm_list = []
setup_slm_settings = {}
slm_num = None

@app.route('/setup_calculation', methods=['GET', 'POST'])
def setup_calculation():
    global main_path, computational_space, n_iterations

    if request.method == 'POST':

        new_main_path = request.form['main_path']
        if new_main_path:
            main_path = str(new_main_path)
            print("Updated Pattern Path to: " + main_path)
            flash("Updated Pattern Path to: " + main_path)

        size = request.form['computational_space']
        if size:
            computational_space = (int(size), int(size))
            print("Updated Computational Space to: " + str(computational_space))
            flash("Updated Computational Space to: " + str(computational_space))

        new_n_iterations = request.form['n_iterations']
        if new_n_iterations:
            n_iterations = int(new_n_iterations)
            print("Updated Number of Iterations to: " + str(n_iterations))
            flash("Updated Number of Iterations to: " + str(n_iterations))
        
        return redirect(url_for('setup_calculation'))
    
    return render_template('setup_calculation.html')

@app.route('/setup_slm', methods=['GET', 'POST'])
def setup_slm():
    global setup_slm_settings

    if request.method == 'POST':

        display_num = int(request.form['display_num'])
        bitdepth = int(request.form['bitdepth'])
        wav_design_um = float(request.form['wav_design_um'])
        wav_um = float(request.form['wav_um'])

        setup_slm_settings['display_num'] = display_num
        setup_slm_settings['bitdepth'] = bitdepth
        setup_slm_settings['wav_design_um'] = wav_design_um
        setup_slm_settings['wav_um'] = wav_um
        
        pyglet.clock.schedule_once(lambda dt: dispatcher.create_slm(), 0)

        return redirect(url_for('setup_slm'))

    return render_template('setup_slm.html')

@dispatcher.event
def on_create_slm():
    global setup_slm_settings, slm_list

    print("Creating SLM with settings:", setup_slm_settings)

    try:
        iface = Interface.SLMSuiteInterface()

        slm = ScreenMirrored(setup_slm_settings['display_num'], 
                                setup_slm_settings['bitdepth'], 
                                wav_design_um=setup_slm_settings['wav_design_um'], 
                                wav_um=setup_slm_settings['wav_um'])

        phase_mgr = PhaseManager.PhaseManager(slm)
        wrapped_slm = CorrectedSLM.CorrectedSLM(slm, phase_mgr)
        iface.set_SLM(wrapped_slm)
        iface.set_camera()

        setup_slm_settings['iface'] = iface
        setup_slm_settings['phase_mgr'] = phase_mgr

        slm_list.append(setup_slm_settings.copy())

        print("Succesfully setup SLM on display: " + str(setup_slm_settings['display_num']))
    except Exception as e:
        print("Error creating SLM:", e)

@app.route('/setup_virtual', methods=['GET'])
def setup_virtual():
    global setup_slm_settings, slm_list

    if request.method == 'GET':
        setup_slm_settings['display_num'] = "virtual"

        iface = Interface.SLMSuiteInterface()

        slm = iface.set_SLM()

        phase_mgr = PhaseManager.PhaseManager(slm)
        wrapped_slm = CorrectedSLM.CorrectedSLM(slm, phase_mgr)
        iface.set_SLM(wrapped_slm)
        iface.set_camera()

        setup_slm_settings['iface'] = iface
        setup_slm_settings['phase_mgr'] = phase_mgr

        slm_list.append(setup_slm_settings.copy())
        
        print("Succesfully setup SLM on display: " +str(setup_slm_settings['display_num']))
        flash("Succesfully setup SLM on display: " +str(setup_slm_settings['display_num']))

        return redirect(url_for('setup_slm'))

    return redirect(url_for('setup_slm'))

@app.route('/use_slm_amp', methods=['GET', 'POST'])
def use_slm_amp():
    global slm_list, slm_num

    if request.method == 'POST':
        if slm_num is not None:
            current_slm_settings = slm_list[slm_num]
            iface = current_slm_settings['iface']

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
                flash(f"Set SLM amplitude to Gaussian with waist: ({waist_x}, {waist_y})")
            else:
                print("Unknown amp type")

        return redirect(url_for('use_slm_amp'))
    
    return render_template('use_slm_amp.html')

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

        print("Matrix: " + str(A) + ", Vector: " + str(b))
        return redirect(url_for('setup_calculation'))
    
    return redirect(url_for('setup_calculation'))

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    global slm_list, slm_num
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]
        current_phase_info = get_current_phase_info()
        phase_mgr = current_slm_settings['phase_mgr']
        if not current_slm_settings['display_num'] == "virtual":
            get_screenshot()
    else:
        current_phase_info = None
        phase_mgr = None
        current_slm_settings = None

    return render_template('dashboard.html', 
                           current_phase_info=current_phase_info, 
                           current_slm_settings=current_slm_settings,
                           phase_mgr=phase_mgr,
                           slm_list=slm_list)

@app.route('/select_slm', methods=['POST'])
def select_slm():
    global slm_num

    if request.method == 'POST':
        user_input = request.form['slm_num']
        
        slm_num = int(user_input)
        print("SLM Number Set to: " + str(slm_num))
        flash("SLM Number Set to: " + str(slm_num))

        return redirect(url_for('dashboard'))
                
    return redirect(url_for('dashboard'))

@app.route('/project', methods=['POST'])
def project():
    global slm_num, slm_list
    current_slm_settings = slm_list[slm_num]

    if request.method == 'POST':
        if slm_num is not None and not current_slm_settings['display_num'] == "virtual":
            pyglet.clock.schedule_once(lambda dt: dispatcher.project_pattern(), 0)
            print("Ran the HTTP route to project")
            return redirect(url_for('dashboard'))
        
    return redirect(url_for('dashboard'))

@dispatcher.event
def on_project_pattern():
    global slm_list, slm_num

    current_slm_settings = slm_list[slm_num]
    iface = current_slm_settings['iface']
    phase_mgr = current_slm_settings['phase_mgr']
    iface.write_to_SLM(phase_mgr.base, phase_mgr.base_source)
    
    print("Succesfully projected to display: " + str(current_slm_settings['display_num']))

def get_current_phase_info():
    global slm_list, slm_num, main_path
    
    current_slm_settings = slm_list[slm_num]

    phase_mgr = current_slm_settings['phase_mgr']

    # Get the file path of the base pattern
    base_str = phase_mgr.base_source
    base_str = base_str.replace(main_path, "")
    # String for additional phase patterns
    add_list = []
    # Log of all additional phase patterns
    log = phase_mgr.add_log
    # Iterate over additional phase patterns and add to string
    for item in log:
        add_list.append(str(item[0]) + ":" + str(item[1]))

    # Get the aperture
    aperture = phase_mgr.aperture
    aperture_str = str(aperture)

    print("Succesfully got phase info")
    flash("Succesfully got phase info")

    return base_str, add_list, aperture_str
    
def get_screenshot():
    global slm_list, slm_num
    
    current_slm_settings = slm_list[slm_num]

    displays = screeninfo.get_monitors()
    display = displays[current_slm_settings['display_num']]
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
    flash("Succesfully saved screenshot to: " + output)

target_plot_flag = 0
target_path = ""

@app.route('/display_targets')
def display_targets():
    global slm_list, slm_num, target_plot_flag, target_path
    
    if target_plot_flag:
        targets = utils.get_target_from_file(target_path[:-9])
        x_coords = targets[0].tolist()
        y_coords = targets[1].tolist()
        labels = list(range(len(x_coords)))

        target_plot_flag = 0
        print("Displaying targets from: " + target_path)
        flash("Displaying targets from: " + target_path)

        return jsonify({'x': x_coords, 'y': y_coords, 'labels': labels})
    
    elif slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        phase_mgr = current_slm_settings['phase_mgr']
        targets = utils.get_target_from_file(phase_mgr.base_source)
        x_coords = targets[0].tolist()
        y_coords = targets[1].tolist()
        labels = list(range(len(x_coords)))

        print("Displaying targets from: " + phase_mgr.base_source)
        flash("Displaying targets from: " + phase_mgr.base_source)

        return jsonify({'x': x_coords, 'y': y_coords, 'labels': labels})

@app.route('/base_pattern', methods=['GET', 'POST'])
def base_pattern():
    global base_load_history

    return render_template('base_pattern.html', 
                           base_load_history = base_load_history,
                           main_path=main_path, 
                           computational_space=computational_space, 
                           n_iterations=n_iterations)

@app.route('/calculate', methods=['GET', 'POST'])
def calculate():
    global n_iterations, computational_space, main_path
    global slm_list, slm_num

    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        if request.method == 'POST':
            
            # Get list of target x and y coordinates input by user
            x_coords = request.form.getlist('x_coords')
            y_coords = request.form.getlist('y_coords')
            # Convert coordinates to integers
            x_coords = list(map(float, x_coords))
            y_coords = list(map(float, y_coords))
            # Create 2D numpy array containing x,y coords
            targets = np.array([x_coords, y_coords])

            camera = str(request.form['camera'])
            if camera == "yes":
                targets = andor_to_k(targets)

            print("Targets:" + str(targets))
            # Get list of target amplitudes
            amplitudes = request.form.getlist('amplitudes')
            # Convert amplitudes to floats
            amplitudes = list(map(float, amplitudes))
            # Create 1D numpy array containing amplitudes
            amp_data = np.array(amplitudes)
            
            # Get number of iterations from user
            iteration_number = request.form['iteration_number']
            # If user specified nothing, set to default
            if not iteration_number:
                iteration_number = n_iterations
            
            # Initialize guess phase
            guess_phase = None
            # Get initial guess file path from user
            guess_name = request.form['guess_name']
            guess_name = guess_name[:-11]
            # If there is a guess file path
            if guess_name:
                # Add the pattern folder path
                guess_path = main_path + "base/" + guess_name
                # Extract guess phase pattern
                _,data = utils.load_slm_calculation(guess_path, 1, 1)
                # Check if data was in the file
                if "raw_slm_phase" in data:
                    # Store guess phase pattern
                    guess_phase = data["raw_slm_phase"]
                    print("Stored initial guess phase pattern")
                else:
                    print ("Cannot initiate the guess phase, since it was not saved")

            iface = current_slm_settings['iface']
            # Calculate the base pattern to create the target using GS or WGS algo 
            iface.calculate(computational_space, targets, amp_data, n_iters=int(iteration_number), phase=guess_phase)

            iface.plot_slmplane()
            iface.plot_farfield()
            iface.plot_stats()

            save_name = request.form['save_name']

            new_pattern_path = save_calculation(save_name)
            load_base(new_pattern_path)

            return redirect(url_for('calculate'))
    
    return render_template('calculate.html')

def andor_to_k(x):
    global A, b
    targets = np.matmul(A,x)+b
    return targets

def k_to_andor(k):
    global A, b
    invA = np.linalg.inv(A)
    targets = np.matmul(invA, k - b)
    return targets

@app.route('/grid')
def grid():
    return render_template('grid.html')

@app.route('/submit_points', methods=['POST'])
def submit_points():
    data = request.json
    x_coords = data['xCoords']
    y_coords = data['yCoords']
    amplitudes = data['amplitudes']
    # Process the data as needed
    print(x_coords)
    print(y_coords)
    print(amplitudes)
    return jsonify({'status': 'success'})

@app.route('/calculate_grid', methods=['GET', 'POST'])
def calculate_grid():
    global n_iterations, computational_space, main_path
    global slm_list, slm_num
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        if request.method == 'POST':
            # Get JSON data from the user
            data = request.get_json()
            print(data)
            # Extract xCoords and yCoords from the JSON data
            x_coords = data['xCoords']
            y_coords = data['yCoords']
            # Convert to integers
            x_coords = list(map(int, x_coords))
            y_coords = list(map(int, y_coords))
            # Create 2D numpy array containing target x,y coords
            targets = np.array([x_coords, y_coords])
            # Scale up to computational space
            scaling_factor = computational_space[0] / 64
            targets = targets * scaling_factor
            print(targets)

            # Get the number of target points
            num_points = len(x_coords)
            # Create a 1D numpy array of 1s for target amplitudes
            #TODO: find a way for the user to specify non-uniform amplitudes
            amp_data = np.ones(num_points, float)

            # Get number of iterations from user
            iteration_number = data['iteration_number']
            # If user specified nothing, set to default
            if not iteration_number:
                iteration_number = n_iterations

            # Initialize guess phase
            guess_phase = None
            # Get initial guess file path from user
            guess_path = data['guess_path']
            # If there is a guess file path
            if guess_path:
                # Add the pattern folder path
                #guess_path = add_pattern_path(guess_path)
                # Extract guess phase pattern
                _,data = utils.load_slm_calculation(guess_path, 1, 1)
                # Check if data was in the file
                if "raw_slm_phase" in data:
                    # Store guess phase pattern
                    guess_phase = data["raw_slm_phase"]
                    print("Stored initial guess phase pattern")
                else:
                    print ("Cannot initiate the guess phase, since it was not saved")
            
            iface = current_slm_settings['iface']
            # Calculate the base pattern to create the target using GS or WGS algo 
            iface.calculate(computational_space, targets, amp_data, n_iters=int(iteration_number), phase=guess_phase)

            # Plot stuff about the calculation, does not work at the moment
            iface.plot_slmplane()
            iface.plot_farfield()
            iface.plot_stats()
            
            save_path = str(data['save_path'])
            save_name = str(data['save_name'])
            new_pattern_path = save_calculation(save_path, save_name)
            load_base(new_pattern_path)

            # Response to validate
            result = {'numPoints': num_points, 'xCoords': x_coords, 'yCoords': y_coords}
            print(result)
            return jsonify(result)
    
    return render_template('calculate_grid.html')

def save_calculation(save_name):
    global main_path, slm_list, slm_num
    
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        # Add pattern path if its not an absolute path
        save_path = main_path + "base/"

        # Dictionary to store save options
        save_options = dict()
        save_options["config"] = True # This option saves the configuration of this run of the algorithm
        save_options["slm_pattern"] = True # This option saves the slm phase pattern and amplitude pattern (the amplitude pattern is not calculated. So far, the above have assumed a constant amplitude and we have not described functionality to change this)
        save_options["ff_pattern"] = True # This option saves the far field amplitude pattern
        save_options["target"] = True # This option saves the desired target
        save_options["path"] = save_path # Enable this to save to a desired path. By default it is the current working directory
        save_options["name"] = save_name # This name will be used in the path.
        save_options["crop"] = True # This option crops the slm pattern to the slm, instead of an array the shape of the computational space size.

        iface = current_slm_settings['iface']
        # Save the calculated pattern to a new file
        config_path, new_pattern_path, err = iface.save_calculation(save_options)
        print(config_path)
        print(new_pattern_path)
        print(err)
        return new_pattern_path[:-9]

@app.route('/use_pattern', methods=['POST'])
def use_pattern():
    global main_path

    if request.method == 'POST':

        # Get the file name input by the user
        file = request.files['fname']
        fname = file.filename[:-9]
        print("Received " + fname)

        # Add the pattern path (if it is just a file name)
        path = main_path + "base/" + fname

        load_base(path)

        return redirect(url_for('base_pattern'))
    
    return render_template('base_pattern')

def load_base(path):
    global slm_list, slm_num, base_load_history
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        # Get the phase pattern from the file
        _,data = utils.load_slm_calculation(path, 0, 1)
        phase = data["slm_phase"]

        phase_mgr = current_slm_settings['phase_mgr']
        # Set the phase pattern as the base of the phase manager
        phase_mgr.set_base(phase, path)
        print("Pattern added succesfully")

        # Get the time the file was uploaded
        upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add the file name and upload time to the history
        base_load_history.append({'fname': path, 'upload_time': upload_time})

@app.route('/reset_pattern', methods=['GET', 'POST'])
def reset_pattern():
    global slm_list, slm_num

    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        if request.method == 'POST':
            # Reset the base phase pattern
            phase_mgr = current_slm_settings['phase_mgr']
            phase_mgr.reset_base()
            print("Sucesfully Reset Base Pattern")

            return redirect(url_for('base_pattern'))

    return redirect(url_for('base_pattern'))

@app.route('/targets', methods=['GET', 'POST'])
def targets():
    global main_path, target_path, target_plot_flag
    if request.method == "POST":
        fname = request.files['fname'].filename
        target_path = main_path + "base/" + fname

        target_plot_flag = 1
        return redirect(url_for('targets'))
    
    return render_template('targets.html')

@app.route('/additional_pattern', methods=['GET', 'POST'])
def additional_pattern():

    return render_template('additional_pattern.html')

@app.route('/input_additional', methods=['GET', 'POST'])
def input_additional():

    return render_template('input_additional.html')

@app.route('/add_fresnel_lens', methods=['POST'])
def add_fresnel_lens():
    global slm_list, slm_num
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        if request.method == 'POST':
            #TODO: two focal lengths

            # Got focal length from user
            focal_length = float(request.form['focal_length'])
            # Store focal length in a 1D numpy array
            focal_length = np.array([focal_length])

            # Add the fresnel lens
            phase_mgr = current_slm_settings['phase_mgr']
            phase_mgr.add_fresnel_lens(focal_length[0])
            print("Added fresnel lens")

            return redirect(url_for('input_additional'))
    
    return redirect(url_for('input_additional'))

@app.route('/add_offset', methods=['POST'])
def add_offset():
    global slm_list, slm_num
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        if request.method =='POST':
            # Get x,y coordinates for the offset
            offset_x = float(request.form['offset_x'])
            offset_y = float(request.form['offset_y'])
            # Store offset coords in a 1D numpy array
            offset = np.array([offset_x, offset_y])

            # Add the offset
            phase_mgr = current_slm_settings['phase_mgr']
            phase_mgr.add_offset(offset)

            return redirect(url_for('input_additional'))
    
    return redirect(url_for('input_additional'))

@app.route('/add_zernike_poly', methods=['POST'])
def add_zernike_poly():
    global slm_list, slm_num
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        if request.method == 'POST':
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
            print("Added Zernike")

            return redirect(url_for('input_additional'))
    
    return redirect(url_for('input_additional'))

@app.route('/use_aperture', methods=['POST'])
def use_aperture():
    global slm_list, slm_num
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        if request.method == 'POST':
            # Get the aperture size from the user
            aperture_size = float(request.form['aperture_size'])
            # Store twice in a 1D numpy array
            #TODO: why twice, should the user be able to pass a second aperture size?
            aperture = np.array([aperture_size, aperture_size])

            # Set the aperture
            phase_mgr = current_slm_settings['phase_mgr']
            phase_mgr.set_aperture(aperture)

            return redirect(url_for('input_additional'))
    
    return redirect(url_for('input_additional'))     

@app.route('/save_add_phase', methods=['GET', 'POST'])
def save_add_phase():
    global slm_list, slm_num, main_path
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        if request.method == 'POST':
            # Get the file name from user
            save_name = request.form['save_name']
            # Add pattern path if its not an absolute path
            save_path = main_path + "additional/"
            
            # Dictionary containing save options
            save_options = dict()
            save_options["config"] = True # This option saves the information about how this additional phase was created
            save_options["phase"] = True # saves the actual phase
            save_options["path"] = save_path # Enable this to save to a desired path. By default it is the current working directory
            save_options["name"] = save_name # This name will be used in the path.

            # Save additional phase pattern to new file
            phase_mgr = current_slm_settings['phase_mgr']
            config_path, new_pattern_path = phase_mgr.save_to_file(save_options)
            print(config_path)
            print(new_pattern_path)

            return redirect(url_for('save_add_phase'))
    
    return render_template('save_add_phase.html')

@app.route('/use_add_phase', methods=['GET', 'POST'])
def use_add_phase():
    global slm_list, slm_num, main_path
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        if request.method == 'POST':
            # Get file name input by user
            file = request.files['fname']
            fname = file.filename[:-19]
            print("Received for add phase: " + fname)

            # Add pattern path if its just a file name
            path = main_path + "additional/" + fname

            # Add additional phase pattern to phase manager
            phase_mgr = current_slm_settings['phase_mgr']
            phase_mgr.add_from_file(path)
            print("Additional phase added succesfully")

            return redirect(url_for('use_add_phase'))
    
    return render_template('use_add_phase.html')

@app.route('/add_pattern_to_add_phase', methods=['POST'])
def add_pattern_to_add_phase():
    global slm_list, slm_num, main_path
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        if request.method == 'POST':

            # Get file path for additional phase from user
            file = request.files['path']
            fname = file.filename[:-19]
            print("Received " + path)

            # Add pattern path if its not global
            path = main_path + "additional/" + fname

            # Add the additional phase pattern
            phase_mgr = current_slm_settings['phase_mgr']
            phase_mgr.add_pattern_to_additional(path)

            return redirect(url_for('use_add_phase'))
    
    return redirect(url_for('use_add_phase'))

@app.route('/reset_additional_phase', methods=['POST'])
def reset_additional_phase():
    global slm_list, slm_num
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        if request.method == 'POST':
            # Reset the additional phase pattern
            phase_mgr = current_slm_settings['phase_mgr']
            phase_mgr.reset_additional()
            print("Sucesfully Reset Additional Phase")

            return redirect(url_for('additional_pattern'))

    return redirect(url_for('additional_pattern'))

@app.route('/reset_aperture', methods=['POST'])
def reset_aperture():
    global slm_list, slm_num
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        if request.method == 'POST':
            # Reset the aperture
            phase_mgr = current_slm_settings['phase_mgr']
            phase_mgr.reset_aperture()
            print("Aperture Reset")

            return redirect(url_for('additional_pattern'))
    
    return redirect(url_for('additional_pattern'))

@app.route('/correction', methods=['GET', 'POST'])
def correction():
    global slm_list, slm_num, main_path
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]

        if request.method == 'POST':
            # Get correction file name from user
            file = request.files['fname']
            fname = file.filename
            print("Received correction pattern: " + fname)

            path = main_path + "manufacturer/" + fname

            phase_mgr = current_slm_settings['phase_mgr']
            phase_mgr.add_correction(path, current_slm_settings['bitdepth'], 1)

            """
            # Check if connected to hardware
            if  slm_settings['slm_type'] == "hamamatsu":
                # Add correction phase pattern
                phase_mgr.add_correction(fname, slm_settings['bitdepth'], 1)
            else:
                phase_mgr.add_correction(fname, slm_settings['bitdepth'], 1) #TODO, in case you need to scale.
                #TODO: ask what this is for?
            """

            return redirect(url_for('correction'))
    
    return render_template('correction.html')

@app.route('/config', methods=['GET', 'POST'])
def config():

    return render_template('config.html')

@app.route('/load_config', methods=['GET', 'POST'])
def load_config():
    global slm_list, slm_num, main_path
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]
        phase_mgr = current_slm_settings['phase_mgr']
        if request.method == 'POST':
            filename = request.files['filename'].filename

            filepath = main_path + "config/" + filename

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
                    focal_length = np.array(ast.literal_eval(config[key]))
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

            return redirect(url_for('config'))
    
    return redirect(url_for('config'))

@app.route('/save_config', methods=['GET', 'POST'])
def save_config():
    global slm_list, slm_num, main_path
    if slm_num is not None:
        current_slm_settings = slm_list[slm_num]
        phase_mgr = current_slm_settings['phase_mgr']

        if request.method == 'POST':
            config_dict = dict()
            base_str = phase_mgr.base_source
            print(base_str)
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
            path = main_path + "config/" + save_name
            with open(path, 'x') as fhdl:
                yaml.dump(config_dict, fhdl)

            return redirect(url_for('config'))
        
    return redirect(url_for('config'))


if __name__ == '__main__':
    flask_thread = threading.Thread(target=start_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    print("Starting Pyglet app...")   
    pyglet.app.run()
    

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
