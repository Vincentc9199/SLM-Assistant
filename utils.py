import yaml
import os
from datetime import datetime
from slmsuite.holography import toolbox
import numpy as np

# Data saving related

def save_slm_calculation(hologram, save_options, extra_data = None):
    """

    """
    if not "config" in save_options:
        save_options["config"] = False
    if not "slm_pattern" in save_options:
        save_options["slm_pattern"] = False
    if not "ff_pattern" in save_options:
        save_options["ff_pattern"] = False
    if not "target" in save_options:
        save_options["target"] = False
    if not "path" in save_options:
        save_options["path"] = os.getcwd()
    if not "name" in save_options:
        save_options["name"] = ""
    if not "prefix" in save_options:
        now = datetime.now()
        save_options["prefix"] = now.strftime("%Y%m%d_%H%M%S")
    if not "crop" in save_options:
        save_options["crop"] = True

    full_path = None
    full_path2 = None
    # Now, we gather all the information to save in the config
    if save_options["config"] is True:
        config_info = dict()
        config_info["method"] = hologram.method
        config_info["iteration"] = hologram.iter
        config_info["alg_settings"] = hologram.flags
        config_info["computational_shape"] = str(list(hologram.shape))
        config_info["slm_shape"] = str(list(hologram.slm_shape))
        config_info["save_options"] = save_options
        if extra_data is not None:
            config_info["extra_data"] = extra_data
        # Now perform the actual save:
        full_path = save_options["path"] + os.sep + save_options["prefix"] + save_options["name"] + "_config.yml"
        fhdl = open(full_path, 'w')
        yaml.dump(config_info, fhdl)
        
    npy_data = dict()
    if save_options["slm_pattern"] is True:
        # Here, we save the amplitude pattern that is assumed, and the phase pattern for the SLM
        # If crop is True, then we resize the amplitude and phase pattern 
        try:
            if isinstance(hologram.amp, float):
                amp = hologram.amp
                npy_data["slm_amp"] = amp
            else:
                amp = hologram.amp.get()
                if save_options["crop"] is True:
                    npy_data["slm_amp"] = toolbox.unpad(amp, hologram.slm_shape)
                else:
                    npy_data["slm_amp"] = amp
        except:
            amp = hologram.amp
            npy_data["slm_amp"] = amp

        npy_data["raw_slm_amp"] = hologram.amp

        if save_options["crop"] is True:
            npy_data["slm_phase"] = toolbox.unpad(hologram.extract_phase(), hologram.slm_shape)
        else:
            npy_data["slm_phase"] = hologram.extract_phase()

        npy_data["raw_slm_phase"] = hologram.phase
            

    if save_options["ff_pattern"] is True:
        # Here, we save the farfield pattern. There is no cropping here.
        npy_data["ff_amp"] = hologram.extract_farfield()

    if save_options["target"] is True:
        # Here, we save the target. The target is in the computational space
        npy_data["target"] = hologram.target
        npy_data["weights"] = hologram.weights

    if (save_options["slm_pattern"] is True) or (save_options["ff_pattern"] is True) or (save_options["target"] is True):
        full_path2 = save_options["path"] + os.sep + save_options["prefix"] + save_options["name"] + "_data"
        np.savez_compressed(full_path2, **npy_data)
        full_path2 = full_path2 + ".npz" # Add file extension

    return full_path, full_path2

def load_slm_calculation(base_path, bConfig, bPhase):
    config = None
    data = None
    if bConfig:
        config_path = base_path + "_config.yml"
        with open(config_path, 'r') as fhdl:
            config = yaml.load(fhdl, Loader=yaml.FullLoader)
    if bPhase:
        data_path = base_path + "_data.npz"
        data = np.load(data_path)
    return config,data

def save_add_phase(phase_mgr, save_options, extra_data=None):
    """

    """
    if not "config" in save_options:
        save_options["config"] = False
    if not "phase" in save_options:
        save_options["phase"] = False
    if not "path" in save_options:
        save_options["path"] = os.getcwd()
    if not "name" in save_options:
        save_options["name"] = ""
    if not "prefix" in save_options:
        now = datetime.now()
        save_options["prefix"] = now.strftime("%Y%m%d_%H%M%S")
    full_path = None 
    full_path2 = None
    # Now, we gather all the information to save in the config
    if save_options["config"] is True:
        config_info = dict()
        config_info["shape"] = str(list(phase_mgr.shape))
        config_info["log"] = str(phase_mgr.add_log)
        if extra_data is not None:
            config_info["extra_data"] = extra_data
        # Now perform the actual save:
        full_path = save_options["path"] + os.sep + save_options["prefix"] + save_options["name"] + "_add_phase_config.yml"
        fhdl = open(full_path, 'w')
        yaml.dump(config_info, fhdl)

    npy_data = dict()
    if save_options["phase"] is True:
        npy_data["phase"] = phase_mgr.additional
        full_path2 = save_options["path"] + os.sep + save_options["prefix"] + save_options["name"] + "_add_phase_data"
        np.savez_compressed(full_path2, **npy_data)
        full_path2 = full_path2 + ".npz" # Add file extension
    return full_path, full_path2

def load_add_phase(base_path, bConfig, bPhase):
    config = None
    data = None
    if bConfig:
        config_path = base_path + "_add_phase_config.yml"
        with open(config_path, 'r') as fhdl:
            config = yaml.load(fhdl, Loader=yaml.FullLoader)
    if bPhase:
        data_path = base_path + "_add_phase_data.npz"
        data = np.load(data_path)
    return config,data

def get_target(data_dict):
    idxs = np.nonzero(data_dict["target"])
    ntargets = len(idxs[0])
    target = np.zeros((2, ntargets))
    target[0,:] = idxs[1]
    target[1,:] = idxs[0]
    return target

def get_target_from_file(path):
    _, data_dict = load_slm_calculation(path, 0, 1)
    return get_target(data_dict)

# Callback for hologram feedback
def feedback_client_callback(client):
    def func(hologram):
        hologram.cameraslm.slm.write(hologram.extract_phase(), settle=True)
        spot_amps = client.get_spot_amps()
        if np.array_equal(spot_amps, np.array([-1.0])):
            nspots = len(hologram.spot_amp)
            hologram.external_spot_amp = np.array([1.0 for i in range(nspots)])
        else:
            print("setting external_spot_amp to " + str(spot_amps))
            hologram.external_spot_amp = spot_amps
    return func

## Pattern generation
## Return 2D target array 'targets' such that targets[0,:] = x coords and targets[1,:] = y coords
## Returns also target_amps which is a 1D array of ones the length of the array size
#Center at 256 since coordinates are 512x512

def gen_square_targets2(side_length, pixel_spacing, rot_angle, offset):
    # WARNING: This does not generate an array with pixel spacing = pixel_spacing
    targets = np.zeros((2, side_length**2))
    min_x = -(side_length*pixel_spacing)/2
    min_y = min_x
    x_targets = np.array([min_x + i*(side_length * pixel_spacing) / (side_length - 1) for i in range(0,side_length)])
    y_targets = np.array([min_y + i*(side_length * pixel_spacing) / (side_length - 1) for i in range(0,side_length)])
    targets_mesh = np.array(np.meshgrid(x_targets, y_targets)).T.reshape(-1,2).T
    x_targets = targets_mesh[0,:]
    y_targets = targets_mesh[1,:]
    if rot_angle != 0:
        rot_x_targets = x_targets * np.cos(rot_angle) - y_targets * np.sin(rot_angle)
        rot_y_targets = x_targets * np.sin(rot_angle) + y_targets * np.cos(rot_angle)
    else:
        rot_x_targets = x_targets
        rot_y_targets = y_targets
    x_targets = rot_x_targets + offset[0]
    y_targets = rot_y_targets + offset[1]
    targets[0,:] = x_targets
    targets[1,:] = y_targets
    target_amps = np.ones(side_length**2)
    return targets, target_amps

def gen_square_targets(side_length, pixel_spacing, rot_angle, offset):
    if isinstance(side_length, (int, float)):
        side_length = np.array([side_length, side_length])
    if isinstance(pixel_spacing, (int, float)):
        pixel_spacing = np.array([pixel_spacing, pixel_spacing])
    targets = np.zeros((2, np.prod(side_length)))
    min_x = -(side_length[0] - 1) * pixel_spacing[0] / 2
    min_y = -(side_length[1] - 1) * pixel_spacing[1] / 2
    max_x = (side_length[0] - 1) * pixel_spacing[0] / 2
    max_y = (side_length[1] - 1) * pixel_spacing[1] / 2
    x_targets = np.linspace(min_x, max_x, side_length[0])
    #y_targets = np.linspace(min_y, max_y, side_length[1])
    y_targets = np.linspace(max_y, min_y, side_length[1])
    #targets_mesh = np.array(np.meshgrid(x_targets, y_targets)).T.reshape(-1,2).T
    targets_mesh = np.array(np.meshgrid(y_targets, x_targets)).T.reshape(-1,2).T
    x_targets = targets_mesh[1,:]
    y_targets = targets_mesh[0,:]
    if rot_angle != 0:
        rot_x_targets = x_targets * np.cos(rot_angle) - y_targets * np.sin(rot_angle)
        rot_y_targets = x_targets * np.sin(rot_angle) + y_targets * np.cos(rot_angle)
    else:
        rot_x_targets = x_targets
        rot_y_targets = y_targets
    x_targets = rot_x_targets + offset[0]
    y_targets = rot_y_targets + offset[1]
    #targets[0,:] = x_targets
    #targets[1,:] = y_targets
    targets[0,:] = x_targets
    targets[1,:] = y_targets
    target_amps = np.ones(np.prod(side_length))
    return targets, target_amps
