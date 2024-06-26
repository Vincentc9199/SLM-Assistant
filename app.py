from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import Interface
from slmsuite.hardware.slms.screenmirrored import ScreenMirrored
import PhaseManager
import CorrectedSLM
import CameraClient
import slmsuite.hardware.cameras.thorlabs
import utils
import re
import numpy as np
import yaml
import ast
import screeninfo
import mss
import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Global variables
iface = Interface.SLMSuiteInterface()
phase_mgr = None
pattern_path = '/Users/vincentcosta/Documents/Summer_Research/NaCsSLM-master-2/lib/'
computational_space = (2048, 2048)
n_iterations = 20
config = None
displays = screeninfo.get_monitors()
current_phase_info = ""

# SLM settings
slm_settings = {}

# Virtual setup
#slm_settings['slm_type'] = 'virtual'
#slm = iface.set_SLM()

# Hardware setup
"""
slm_settings['slm_type'] = 'hamamatsu'
slm_settings['display_num'] = 1
slm_settings['bitdepth'] = 8
slm_settings['wav_design_um'] = 0.7
slm_settings['wav_um'] = 0.616

slm = ScreenMirrored(slm_settings['display_num'], slm_settings['bitdepth'], wav_design_um=slm_settings['wav_design_um'], wav_um=slm_settings['wav_um'])


phase_mgr = PhaseManager.PhaseManager(slm)
wrapped_slm = CorrectedSLM.CorrectedSLM(slm, phase_mgr)
iface.set_SLM(wrapped_slm)
"""

# Camera settings
camera_settings = {}

# Virtual setup
camera_settings['camera_type'] = 'virtual'
camera = iface.set_camera()

# History lists
pattern_load_history = []
add_load_history = []
config_load_history = []
calculation_save_history = []
add_save_history = []
config_save_history = []

# Home page
@app.route('/')
def home():
    return render_template('home.html')


# SLM Setup Page
@app.route('/setup_slm', methods=['GET', 'POST'])
def setup_slm():
    global iface
    global phase_mgr
    global slm_settings

    if request.method == 'POST':

        slm_type = request.form['slm_type']
        print(slm_type)

        if slm_type == "virtual":
            slm = iface.set_SLM()
        
        elif slm_type == "hamamatsu":
            
            display_num = int(request.form['display_num'])
            print(display_num)
            bitdepth = int(request.form['bitdepth'])
            print(bitdepth)
            wav_design_um = float(request.form['wav_design_um'])
            print(wav_design_um)
            wav_um = float(request.form['wav_um'])
            print(wav_um) 
            
            slm = ScreenMirrored(display_num, bitdepth, wav_design_um=wav_design_um, wav_um=wav_um)

            slm_settings['display_num'] = display_num
            slm_settings['bitdepth'] = bitdepth
            slm_settings['wav_design_um'] = wav_design_um
            slm_settings['wav_um'] = wav_um
        else:
            print("SLM type not recognized")

        phase_mgr = PhaseManager.PhaseManager(slm)
        wrapped_slm = CorrectedSLM.CorrectedSLM(slm, phase_mgr)
        iface.set_SLM(wrapped_slm)

        slm_settings['slm_type'] = slm_type
        
        print("SLM setup succesful") 
        return redirect(url_for('setup_slm'))

    return render_template('setup_slm.html', slm_settings=slm_settings)

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

@app.route('/use_pattern', methods=['GET', 'POST'])
def use_pattern():
    global phase_mgr
    global pattern_load_history

    if request.method == 'POST':
        fname = request.form['fname']
        print("Received " + fname)

        if re.match(r'[A-Z]:', fname) is None:
        # check to see if it's an absolute path
            fname = pattern_path + fname

        _,data = utils.load_slm_calculation(fname, 0, 1)

        phase = data["slm_phase"]

        phase_mgr.set_base(phase, fname)

        upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pattern_load_history.append({'fname': fname, 'upload_time': upload_time})

        print("Pattern added succesfully")

        return redirect(url_for('use_pattern'))
    
    return render_template('use_pattern.html', pattern_load_history = pattern_load_history)

@app.route('/use_add_phase', methods=['GET', 'POST'])
def use_add_phase():

    global pattern_path
    global phase_mgr
    global add_load_history

    if request.method == 'POST':
        fname = request.form['fname']
        print("Received for add phase: " + fname)
        if re.match(r'[A-Z]:', fname) is None:
            # check to see if it's an absolute path
            fname = pattern_path + fname
        phase_mgr.add_from_file(fname)

        upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        add_load_history.append({'fname': fname, 'upload_time': upload_time})

        print("Additional phase added succesfully")

        return redirect(url_for('use_add_phase'))
    
    return render_template('use_add_phase.html', add_load_history = add_load_history)

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

@app.route('/calculate', methods=['GET', 'POST'])
def calculate():
    global n_iterations
    global iface
    global computational_space
    global pattern_path

    if request.method == 'POST':
        x_coords = request.form.getlist('x_coords')
        y_coords = request.form.getlist('y_coords')

        x_coords = list(map(int, x_coords))
        y_coords = list(map(int, y_coords))

        targets = np.array([x_coords, y_coords])

        amplitudes = request.form.getlist('amplitudes')
        amplitudes = list(map(float, amplitudes))
        amp_data = np.array(amplitudes)
        amp_data = np.copy(amp_data)

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

        return redirect(url_for('calculate'))
    
    return render_template('calculate.html')


@app.route('/calculate_grid', methods=['GET', 'POST'])
def calculate_grid():
    global n_iterations
    global iface
    global computational_space
    global pattern_path

    if request.method == 'POST':
        data = request.get_json()

        # Extract xCoords and yCoords from the JSON data
        x_coords = data['xCoords']
        y_coords = data['yCoords']

        x_coords = list(map(int, x_coords))
        y_coords = list(map(int, y_coords))

        targets = np.array([x_coords, y_coords])

        num_points = len(x_coords)

        amp_data = np.ones(num_points, float)

        phase_path = ""

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

        # Example response
        result = {'numPoints': num_points, 'xCoords': x_coords, 'yCoords': y_coords}
        print(result)
        return jsonify(result)
    
    return render_template('calculate_grid.html')


@app.route('/save_calculation', methods=['GET', 'POST'])
def save_calculation():
    global pattern_path
    global iface
    global calculation_save_history

    if request.method == 'POST':
        save_path = request.form['save_path']
        save_name = request.form['save_name']
        if re.match(r'[A-Z]:', save_path) is None:
            # check to see if it's an absolute path
            save_path = pattern_path + save_path
        
        print(save_path)

        save_options = dict()
        save_options["config"] = True # This option saves the configuration of this run of the algorithm
        save_options["slm_pattern"] = True # This option saves the slm phase pattern and amplitude pattern (the amplitude pattern is not calculated. So far, the above have assumed a constant amplitude and we have not described functionality to change this)
        # This was changed to false to fix a bug, not sure why this stopped working
        save_options["ff_pattern"] = True # This option saves the far field amplitude pattern
        save_options["target"] = True # This option saves the desired target
        save_options["path"] = save_path # Enable this to save to a desired path. By default it is the current working directory
        save_options["name"] = save_name # This name will be used in the path.
        save_options["crop"] = True # This option crops the slm pattern to the slm, instead of an array the shape of the computational space size.
        config_path, new_pattern_path, err = iface.save_calculation(save_options)
        print(config_path)
        print(new_pattern_path)
        print(err)
        upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        calculation_save_history.append({'save_path': save_path, 'save_name': save_name, 'upload_time': upload_time})

        return redirect(url_for('save_calculation'))

    return render_template('save_calculation.html', calculation_save_history=calculation_save_history)

@app.route('/save_add_phase', methods=['GET', 'POST'])
def save_add_phase():
    global pattern_path
    global phase_mgr
    global add_save_history

    if request.method == 'POST':
        save_path = request.form['save_path']
        save_name = request.form['save_name']
        
        if re.match(r'[A-Z]:', save_path) is None:
            # check to see if it's an absolute path
            save_path = pattern_path + save_path
        save_options = dict()
        save_options["config"] = True # This option saves the information about how this additional phase was created
        save_options["phase"] = True # saves the actual phase
        save_options["path"] = save_path # Enable this to save to a desired path. By default it is the current working directory
        save_options["name"] = save_name # This name will be used in the path.
        config_path, new_pattern_path = phase_mgr.save_to_file(save_options)
        
        print(config_path)
        print(new_pattern_path)

        upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        add_save_history.append({'save_path': save_path, 'save_name': save_name, 'upload_time': upload_time})

        return redirect(url_for('save_add_phase'))

    return render_template('save_add_phase.html', add_save_history=add_save_history)

@app.route('/use_correction', methods=['GET', 'POST'])
def use_correction():

    global pattern_path
    global phase_mgr
    global slm_settings

    if request.method == 'POST':

        fname = request.form['fname']
        print("Received correction pattern: " + fname)

        if  slm_settings['slm_type'] == "hamamatsu":
            phase_mgr.add_correction(fname, slm_settings['bitdepth'], 1)
        else:
            phase_mgr.add_correction(fname, slm_settings['bitdepth'], 1) #TODO, in case you need to scale.

        return redirect(url_for('use_correction'))
    
    return render_template('use_correction.html')


@app.route('/add_pattern_to_add_phase', methods=['GET', 'POST'])
def add_pattern_to_add_phase():

    global pattern_path
    global phase_mgr

    if request.method == 'POST':

        path = request.form['path']
        
        print("Received " + path)

        if re.match(r'[A-Z]:', path) is None:
            # check to see if it's an absolute path
            path = pattern_path + path
        phase_mgr.add_pattern_to_additional(path)

        return redirect(url_for('add_pattern_to_add_phase'))
    
    return render_template('add_pattern_to_add_phase.html')

@app.route('/add_fresnel_lens', methods=['GET', 'POST'])
def add_fresnel_lens():
    global phase_mgr

    if request.method == 'POST':

        focal_length = float(request.form['focal_length'])
        focal_length = np.array([focal_length])
        phase_mgr.add_fresnel_lens(focal_length[0])
        print("Added fresnel lens")

        return redirect(url_for('add_fresnel_lens'))
    
    return render_template('add_fresnel_lens.html')

@app.route('/add_offset', methods=['GET', 'POST'])
def add_offset():

    if request.method =='POST':
        offset_x = float(request.form['offset_x'])
        offset_y = float(request.form['offset_y'])
    
        offset = np.array([offset_x, offset_y])
        phase_mgr.add_offset(offset)

        return redirect(url_for('add_offset'))
    
    return render_template('add_offset.html')

@app.route('/add_zernike_poly', methods=['GET', 'POST'])
def add_zernike_poly():
    global phase_mgr

    if request.method == 'POST':
        
        npolys = (len(request.form)) // 3

        poly_list = []

        for i in range(npolys):
            n = int(request.form.get(f'n{i}'))
            m = int(request.form.get(f'm{i}'))
            weight = float(request.form.get(f'weight{i}'))
            poly_list.append(((n, m), weight))

        phase_mgr.add_zernike_poly(poly_list)
        print("Added Zernike")

        return redirect(url_for('add_zernike_poly'))
    
    return render_template('add_zernike_poly.html')

@app.route('/reset_additional_phase', methods=['GET', 'POST'])
def reset_additional_phase():
    global phase_mgr

    if request.method == 'POST':
        phase_mgr.reset_additional()
        print("Sucesfully Reset Additional Phase")
        return redirect(url_for('reset_additional_phase'))

    return render_template('reset_additional_phase.html')

@app.route('/reset_pattern', methods=['GET', 'POST'])
def reset_pattern():
    global phase_mgr

    if request.method == 'POST':
        phase_mgr.reset_base()
        print("Sucesfully Reset Base Pattern")
        return redirect(url_for('reset_pattern'))

    return render_template('reset_pattern.html')

"""
@app.route('/use_slm_amp', methods=['GET', 'POST'])
def use_slm_amp():
    global iface

    if request.method == 'POST':

        func = request.form['func']
        if func == "gaussian":
            waist_x = request.form['waist_x']
            waist_y = request.form['waist_y']

            shape = iface.slm.shape
            xpix = (shape[1] - 1) *  np.linspace(-.5, .5, shape[1])
            ypix = (shape[0] - 1) * np.linspace(-.5, .5, shape[0])

            x_grid, y_grid = np.meshgrid(xpix, ypix)

            gaussian_amp = np.exp(-np.square(x_grid) * (1 / waist_x**2)) * np.exp(-np.square(y_grid) * (1 / waist_y**2))

            iface.set_slm_amplitude(gaussian_amp)
        else:
            print("Unknown amp type")

        return redirect(url_for('use_slm_amp'))
    
    return render_template('use_slm_amp.html')
"""

@app.route('/use_aperture', methods=['GET', 'POST'])
def use_aperture():
    global phase_mgr

    if request.method == 'POST':
        aperture_size = float(request.form['aperture_size'])
        aperture = np.array([aperture_size, aperture_size])
        phase_mgr.set_aperture(aperture)

        return redirect(url_for('use_aperture'))
    
    return render_template('use_aperture.html')

@app.route('/reset_aperture', methods=['GET', 'POST'])
def reset_aperture():
    global phase_mgr

    if request.method == 'POST':
        phase_mgr.reset_aperture()
        print("Aperture Reset")
        return redirect(url_for('reset_aperture'))
    
    return render_template('reset_aperture.html')

@app.route('/project', methods=['GET', 'POST'])
def project():
    global iface
    global phase_mgr

    if request.method == 'POST':
        iface.write_to_SLM(phase_mgr.base, phase_mgr.base_source)
        print("Projected to SLM") 
        return redirect(url_for('project'))
    
    return render_template('project.html')

@app.route('/get_current_phase_info', methods=['GET', 'POST'])
def get_current_phase_info():
    global phase_mgr
    global current_phase_info

    if request.method == 'POST':
        base_str = "base: " + phase_mgr.base_source
        add_str = " additional: "
        log = phase_mgr.add_log
        for item in log:
            add_str = add_str + str(item[0]) + ":" + str(item[1]) + ","
        aperture = phase_mgr.aperture
        aperture_str = " aperture: " + str(aperture)
        current_phase_info = base_str + add_str + aperture_str
        print(current_phase_info)

        return redirect(url_for('get_current_phase_info'))
    
    return render_template('get_current_phase_info.html', current_phase_info=current_phase_info)
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

@app.route('/load_config', methods=['GET', 'POST'])
def load_config():
    global config
    global phase_mgr
    global slm_settings
    global config_load_history

    if request.method == 'POST':
        fname = request.form['fname']

        with open(fname, 'r') as fhdl:
            config = yaml.load(fhdl, Loader=yaml.FullLoader)
        
        for key in config:
            if key == "pattern":
                path = config["pattern"]
                if re.match(r'[A-Z]:', path) is None:
                # check to see if it's an absolute path
                    path = pattern_path + path
                _,data = utils.load_slm_calculation(path, 0, 1)
                phase = data["slm_phase"]
                phase_mgr.set_base(phase, fname)
            #elif key == "fourier_calibration":
                #self.send_load_fourier_calibration(config["fourier_calibration"])
            elif key.startswith("file_correction"):
                fname = config[key]
                if  slm_settings['slm_type'] == "hamamatsu":
                    phase_mgr.add_correction(fname, slm_settings['bitdepth'], 1)
                else:
                    phase_mgr.add_correction(fname, slm_settings['bitdepth'], 1) #TODO, in case you need to scale.
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
            
            upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            config_load_history.append({'fname': fname, 'upload_time': upload_time})

        return redirect(url_for('load_config'))
    
    return render_template('load_config.html', config_load_history=config_load_history)

@app.route('/save_config', methods=['GET', 'POST'])
def save_config():
    global config_save_history

    if request.method == 'POST':
        config_dict = dict()
        base_str = phase_mgr.base_source

        if base_str != "":
            config_dict["pattern"] = base_str[0]

        rep = ""
        log = phase_mgr.add_log
        for item in log:
            rep = rep + str(item[0]) + ";" + str(item[1]) + ";"
        
        add_str = rep

        corrections = add_str[0].split(';')
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
        fname = request.form['fname']
        with open(fname, 'x') as fhdl:
            yaml.dump(config_dict, fhdl)

        upload_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        config_save_history.append({'fname': fname, 'upload_time': upload_time})

        return redirect(url_for('save_config'))
    return render_template('save_config.html', config_save_history=config_save_history)

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
@app.route('/get_screenshot', methods=['GET', 'POST'])
def get_screenshot():
    global slm_settings
    global displays

    if request.method == 'POST':
        display = displays[slm_settings['display_num']]
        # Create area for screenshot
        display_rect = {
            "top": display.y,
            "left": display.x,
            "width": display.width,
            "height": display.height
        }
        # Take screenshot
        sct = mss.mss()
        screenshot = sct.grab(display_rect)
        # Do something with the screenshot
        return redirect(url_for('get_screenshot'))
    
    return render_template('get_screenshot.html')
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