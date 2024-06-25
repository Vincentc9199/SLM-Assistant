import numpy as np
import slmsuite.holography.toolbox.phase
import utils
from PIL import Image

class PhaseManager(object):
    def __init__(self, slm):
        self.slm = slm
        self.shape = slm.shape
        self.base = np.zeros(self.shape,dtype=np.float32)
        self.base_source = ''
        self.additional = np.zeros(self.shape,dtype=np.float32)
        self.add_log = [] # log for everything that has been added to this additional phase
        self.aperture = None
        self.mask = None

    def set_base(self, base, source = ''):
        self.base = base
        self.base_source = source

    def get(self):
        phase = self.base + self.additional
        if self.mask is not None:
            phase = np.multiply(phase, self.mask)
        return phase

    def reset_base(self):
        self.base = np.zeros(self.shape)
        self.base_source = ''

    def reset_additional(self):
        self.additional = np.zeros(self.shape)
        self.add_log = []

    def set_aperture(self, aperture_size):
        self.aperture = aperture_size
        # calculate mask
        center = [(elem - 1)/2 for elem in self.shape]
        mask = np.zeros(self.shape,dtype=np.float32)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if (i - center[0])**2 / (self.aperture[0]**2) + (j - center[1])**2 / (self.aperture[1]**2) < 1:
                    mask[i,j] = 1
        self.mask = mask

    def get_aperture(self):
        return self.aperture, self.mask

    def reset_aperture(self):
        self.aperture = None
        self.mask = None

    def add_fresnel_lens(self, focal_length):
        phase= slmsuite.holography.toolbox.phase.lens(self.slm, focal_length)
        self.additional = self.additional + phase
        self.add_log.append(["fresnel_lens", np.array2string(focal_length, separator=',')])
    
    def add_zernike_poly(self, zernike_list):
        phase = slmsuite.holography.toolbox.phase.zernike_sum(self.slm, zernike_list, aperture="cropped")
        self.additional = self.additional + phase
        self.add_log.append(["zernike", str(zernike_list)])
    def add_offset(self, offset_data):
        phase = slmsuite.holography.toolbox.phase.blaze(self.slm, vector = offset_data)
        self.additional = self.additional + phase
        self.add_log.append(["offset", str(offset_data)])

    def save_to_file(self, save_options, extra_info=None):
        full_path, full_path2 = utils.save_add_phase(self, save_options, extra_info)
        return full_path, full_path2

    def add_from_file(self, fname):
        #with Image.open(fname) as image:
        #    image_array = np.array(image)
        _,data = utils.load_add_phase(fname, 0, 1)
        self.additional = self.additional + data["phase"]
        self.add_log.append(["file", fname])

    def add_pattern_to_additional(self, fname):
        _,data = utils.load_slm_calculation(fname, 0, 1)
        self.additional = self.additional + data["slm_phase"]
        self.add_log.append(["file", fname])

    def add_correction(self, fname, bitdepth, scale):
        with Image.open(fname) as image:
            image_array = np.array(image)
        image_array = image_array / (2**bitdepth - 1) * 2 * np.pi * scale
        act_image_array = np.zeros((1024,1280))
        act_image_array[:,4:1276] = image_array 
        self.additional = self.additional + act_image_array
        self.add_log.append(["file_correction", fname])






