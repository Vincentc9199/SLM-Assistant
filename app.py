from flask import Flask, render_template, request, redirect, url_for
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

app = Flask(__name__)
app.secret_key = os.urandom(24)


# Global variables
iface = Interface.SLMSuiteInterface()
pattern_path = ""
computational_space = (2048, 2048)
n_iterations = 20
phase_mgr = None
slm_type = "virtual"
bitdepth = 0
config = None

# GLobal Functions
def load_pattern(path):
    if re.match(r'[A-Z]:', path) is None:
        # check to see if it's an absolute path
        path = pattern_path + path
    _,data = utils.load_slm_calculation(path, 0, 1)
    return data["slm_phase"]



# Home page
@app.route('/')
def home():
    return render_template('home.html')

# SLM Setup Page
@app.route('/setup_slm', methods=['GET', 'POST'])
def setup_slm():
    global iface
    global phase_mgr
    global slm_type
    global bitdepth

    if request.method == 'POST':

        slm_type = request.form['slm_type']

        if slm_type == "virtual":
            slm = iface.set_SLM()
        
        elif slm_type == "hamamatsu":
            
            display_num = int(request.form['display_num'])
            bitdepth = int(request.form['bitdepth'])
            wav_design_um = float(request.form['wav_design_um'])
            wav_um = float(request.form['wav_um'])    
            slm = ScreenMirrored(display_num, bitdepth, wav_design_um=wav_design_um, wav_um=wav_um)
        else:
            print("SLM type not recognized")

        phase_mgr = PhaseManager.PhaseManager(slm)
        wrapped_slm = CorrectedSLM.CorrectedSLM(slm, phase_mgr)
        iface.set_SLM(wrapped_slm)
            
        return redirect(url_for('setup_slm'))

    return render_template('setup_slm.html')

@app.route('/setup_camera', methods=['GET', 'POST'])
def setup_camera():
    global iface

    if request.method == 'POST':
    
        camera_type = request.form['camera_type']
            
        if camera_type == "virtual":
            camera = iface.set_camera()
       
        elif camera_type == "network":
            url = request.form['camera_url']
            camera = CameraClient.CameraClient(url)
            iface.set_camera(camera)     
        
        elif camera_type == "thorcam_scientific_camera":
            serial_num = request.form['serial_num']
            if serial_num:
                serial = serial_num
            else:
                serial = ""
            camera = slmsuite.hardware.cameras.thorlabs.ThorCam(serial)
            iface.set_camera(camera)
        else:
            print("Camera type not recognized")

        return redirect(url_for('setup_camera'))

    return render_template('setup_camera.html')

@app.route('/use_pattern', methods=['GET', 'POST'])
def use_pattern(fname):
    global phase_mgr

    if request.method == 'POST':
        fname = request.form['fname']
        print("Received " + fname)
        phase = load_pattern(fname)
        phase_mgr.set_base(phase, fname)

        return redirect(url_for('use_pattern'))
    
    return render_template('use_pattern.html')

@app.route('/use_add_phase', methods=['GET', 'POST'])
def use_add_phase():

    global pattern_path
    global phase_mgr

    if request.method == 'POST':
        fname = request.form['fname']
        print("Received for add phase: " + fname)
        if re.match(r'[A-Z]:', fname) is None:
            # check to see if it's an absolute path
            fname = pattern_path + fname
        phase_mgr.add_from_file(fname)

        return redirect(url_for('use_add_phase'))
    
    return render_template('use_add_phase.html')

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


@app.route('/save_calculation', methods=['GET', 'POST'])
def save_calculation():
    global pattern_path
    global iface

    if request.method == 'POST':
        save_path = request.form['save_path']
        save_name = request.form['save_name']
        if re.match(r'[A-Z]:', save_path) is None:
            # check to see if it's an absolute path
            save_path = pattern_path + save_path
        save_options = dict()
        save_options["config"] = True # This option saves the configuration of this run of the algorithm
        save_options["slm_pattern"] = True # This option saves the slm phase pattern and amplitude pattern (the amplitude pattern is not calculated. So far, the above have assumed a constant amplitude and we have not described functionality to change this)
        save_options["ff_pattern"] = True # This option saves the far field amplitude pattern
        save_options["target"] = True # This option saves the desired target
        save_options["path"] = save_path # Enable this to save to a desired path. By default it is the current working directory
        save_options["name"] = save_name # This name will be used in the path.
        save_options["crop"] = True # This option crops the slm pattern to the slm, instead of an array the shape of the computational space size.
        config_path, pattern_path = iface.save_calculation(save_options)
        print(config_path)
        print(pattern_path)

        return redirect(url_for('save_calculation'))

    return render_template('save_calculation.html')

@app.route('/save_add_phase', methods=['GET', 'POST'])
def save_add_phase():
    global pattern_path
    global phase_mgr

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
        config_path, pattern_path = phase_mgr.save_to_file(save_options)
        
        print(config_path)
        print(pattern_path)

        return redirect(url_for('save_add_phase'))

    return render_template('save_add_phase.html')

@app.route('/use_correction', methods=['GET', 'POST'])
def use_correction():

    global pattern_path
    global phase_mgr
    global slm_type
    global bitdepth

    if request.method == 'POST':

        fname = request.form['fname']
        print("Received correction pattern: " + fname)

        if  slm_type == "hamamatsu":
            phase_mgr.add_correction(fname, bitdepth, 1)
        else:
            phase_mgr.add_correction(fname, bitdepth, 1) #TODO, in case you need to scale.

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

        focal_length_1 = request.form['focal_length_1']
        focal_length_2 = request.form['focal_length_2']

        focal_length = np.array(focal_length_1, focal_length_2)

        if len(focal_length) == 1:
            phase_mgr.add_fresnel_lens(focal_length[0])
        else:
            phase_mgr.add_fresnel_lens(focal_length)

        return redirect(url_for('add_fresnel_lens'))
    
    return render_template('add_fresnel_lens.html')

@app.route('/add_offset', methods=['GET', 'POST'])
def add_offset():

    if request.method =='POST':
        offset_x = request.form['offset_x']
        offset_y = request.form['offset_y']
    
        offset = np.array([offset_x, offset_y])
        phase_mgr.add_offset(offset)

        return redirect(url_for('add_offset'))
    
    return render_template('add_offset')

@app.route('/add_zernike_poly', methods=['GET', 'POST'])
def add_zernike_poly():
    global phase_mgr

    if request.method == 'POST':
        
        npolys = (len(request.form) - 1) // 3

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

        return redirect(url_for('reset_additional_phase'))

    return render_template('reset_additional_phase.html')

@app.route('/reset_pattern', methods=['GET', 'POST'])
def reset_pattern():
    global phase_mgr

    if request.method == 'POST':
        phase_mgr.reset_base()

        return redirect(url_for('reset_pattern'))

    return render_template('reset_pattern.html')

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

@app.route('/use_aperture', methods=['GET', 'POST'])
def use_aperture():

    if request.method == 'POST':

        return redirect(url_for('use_aperture'))
    
    return render_template('use_aperture.html')

@app.route('/reset_aperture', methods=['GET', 'POST'])
def reset_aperture():
    global phase_mgr

    if request.method == 'POST':
        phase_mgr.reset_aperture()
        return redirect(url_for('reset_aperture'))
    
    return render_template('reset_aperture.html')

@app.route('/project', methods=['GET', 'POST'])
def project():
    global iface
    global phase_mgr
    if request.method == 'POST':
        iface.write_to_SLM(phase_mgr.base, phase_mgr.base_source) 
        return redirect(url_for('project'))
    
    return render_template('project.html')

@app.route('/get_current_phase_info', methods=['GET', 'POST'])
def get_current_phase_info():
    global phase_mgr

    if request.method == 'POST':
        base_str = "base: " + phase_mgr.base_source
        add_str = " additional: "
        log = phase_mgr.add_log
        for item in log:
            add_str = add_str + str(item[0]) + ":" + str(item[1]) + ","
        aperture = phase_mgr.aperture
        aperture_str = " aperture: " + str(aperture)

        print(base_str + add_str + aperture_str)

        return redirect(url_for('get_current_phase_info'))
    
    return render_template('get_current_phase_info.html')

@app.route('/get_base', methods=['GET', 'POST'])
def get_base():
    global phase_mgr

    if request.method == 'POST':
        print(phase_mgr.base_source)

        return redirect(url_for('get_base'))
    
    return render_template('get_base.html')

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

@app.route('/load_config', methods=['GET', 'POST'])
def load_config():
    global config
    global phase_mgr

    if request.method == 'POST':
        fname = request.form['fname']

        with open(fname, 'r') as fhdl:
            config = yaml.load(fhdl, Loader=yaml.FullLoader)
        
        for key in config:
            if key == "pattern":
                phase = load_pattern(config["pattern"])
                phase_mgr.set_base(phase, fname)
            #elif key == "fourier_calibration":
                #self.send_load_fourier_calibration(config["fourier_calibration"])
            elif key.startswith("file_correction"):
                fname = config[key]
                if  slm_type == "hamamatsu":
                    phase_mgr.add_correction(fname, bitdepth, 1)
                else:
                    phase_mgr.add_correction(fname, bitdepth, 1) #TODO, in case you need to scale.
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
                    new_list.append([item[0][0], item[0][1], item[1]])
                #self.send_zernike_poly(np.array(new_list))
            #elif key == "offset":
                #self.send_offset(np.array(ast.literal_eval(config[key])))
            
        return redirect(url_for('load_config'))
    
    return render_template('load_config.html')


if __name__ == '__main__':
    app.run(debug=False)