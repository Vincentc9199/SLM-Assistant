import zmq
import numpy as np
from slmsuite.hardware.cameras.camera import Camera

class CameraClient(Camera):
    """
    Template for adding a new camera to :mod:`slmsuite`. Replace :class:`Template`
    with the desired subclass name. :class:`~slmsuite.hardware.cameras.camera.Camera` is the
    superclass that sets the requirements for :class:`Template`.

    Attributes
    ----------
    sdk : object
        Many cameras have a singleton SDK class which handles all the connected cameras
        of a certain brand. This is generally implemented as a class variable.
    cam : object
        Most cameras will wrap some handle which connects to the the hardware.
    """

    def recreate_sock(self):
        if self.__sock is not None:
            self.__sock.close()
        self.__sock = self.__ctx.socket(zmq.REQ)
        self.__sock.connect(self.__url)

    # Class variable (same for all instances of Template) pointing to a singleton SDK.
    sdk = None

    def __init__(
        self, 
        url,
        **kwargs
    ):
        """
        Initialize camera and attributes.

        Parameters
        ----------
        serial : str
            Most SDKs identify different cameras by some serial number or string.
        verbose : bool
            Whether or not to print extra information.
        kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """
        # TODO: Insert code here to initialize the camera hardware, load properties, etc.

        # Mandatory functions:
        # - Opening a connection to the device

        # Other possibilities to consider:
        # - Loading a connection to the SDK, if applicable.
        # - Gathering parameters such a width, height, and bitdepth.

        # network
        self.__url = url
        self.__ctx = zmq.Context()
        self.__sock = None
        self.recreate_sock()
        self.timeout = 500
        self.connected = True # Start off true.
        self.width = self.get_width()
        self.height = self.get_height()
        # Finally, use the superclass constructor to initialize other required variables.
        super().__init__(
            self.width,
            self.height,
            bitdepth= self.get_depth(),
            name= self.get_serial(),
            **kwargs
        )

        # ... Other setup.
        self.info = self._info

    # decorators for polling
    # recv_type = 1 is a string receive
    def poll_recv(recv_type = [1], timeout=1000, flag=0, default_val=None):
        def deco(func):
            def f(self, *args): #timeout in milliseconds
                try:
                    if self.connected:
                        func(self, *args)
                    else:
                        return default_val
                except Exception as e:
                    print('Error in client function: ' + str(e))
                    return None
                rep = []
                for i in recv_type:
                    if self.__sock.poll(timeout) == 0:
                        self.connected=False
                        print("Warning: CameraClient is not connected")
                        rep.append(default_val[i])
                    else:
                        if i == 0:
                            rep.append(self.__sock.recv(flag))
                        else:
                            rep.append(self.__sock.recv_string(flag))
                return rep
            return f
        return deco

    def recv1(func):
        def f(*args, **kwargs):
            rep = func(*args, **kwargs)
            data = None
            if rep is not None:
                data = rep[0]
            return data
        return f

    def recv1int(func):
        def f(*args, **kwargs):
            rep = func(*args, **kwargs)
            data = None
            if rep is not None:
                data = int.from_bytes(rep[0], 'little')
            return data
        return f

    def recv1float(func):
        def f(*args, **kwargs):
            rep = func(*args, **kwargs)
            data = None
            if rep is not None:
                data = np.frombuffer(rep[0])
                data = data[0]
            return data
        return f

    def recv1arr(reshape_dims=None, dtype=float):
        def deco(func):
            def f(*args, **kwargs):
                rep = func(*args, **kwargs)
                data = None
                if rep is not None:
                    data = np.frombuffer(rep[0], dtype=dtype)
                    if reshape_dims is not None:
                        data = np.reshape(data, reshape_dims)
                return data
            return f
        return deco

    @recv1
    @poll_recv([1], default_val=["Not connected"])
    def close(self):
        self.__sock.send_string("close")

    @staticmethod
    def info(verbose=True):
        """
        Discovers all cameras detected by the SDK.
        Useful for a user to identify the correct serial numbers / etc.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of str
            List of serial numbers or identifiers.
        """
        return "Please use the info method of the instance, since a socket is necessary"

    @recv1
    @poll_recv([1], default_val=["Not connected"])
    def _info(self, verbose=True):
        self.__sock.send_string("info")

    ### Property Configuration ###

    @recv1float
    @poll_recv([0], default_val=[np.array([1.0]).tobytes()])
    def get_exposure(self):
        self.__sock.send_string("get_exposure")
    
    @recv1
    @poll_recv([1], default_val=["Not connected"])
    def set_exposure(self, exposure_s):
        self.__sock.send_string("set_exposure", zmq.SNDMORE)
        self.__sock.send(exposure_s.astype(np.float64).tobytes())

    @recv1
    @poll_recv([1], default_val=["Not connected"])
    def _set_woi(self, woi):
        self.__sock.send_string("set_woi", zmq.SNDMORE)
        self.__sock.send(woi.astype(np.float64).tobytes())
        self.width = int(woi[1])
        self.height = int(woi[3])

    def set_woi(self, woi=None):
        if woi is None:
            pass
        else:
            self._set_woi(woi)
            

    def get_image(self, timeout_ms=60000):
        @CameraClient.recv1arr((self.height, self.width), np.int32)
        @CameraClient.poll_recv([0], timeout=timeout_ms)
        def _get_image(self):
            self.__sock.send_string("get_image")
        if self.connected:
            return _get_image(self)
        else:
            return np.zeros((self.height, self.width), dtype=np.int32)

    @recv1
    @poll_recv([1], default_val=["Not connected"])
    def flush(self):
        self.__sock.send_string("flush")

    @recv1int
    @poll_recv([0], default_val=[int(512).to_bytes(4, 'little')])
    def get_width(self):
        self.__sock.send_string("get_width")

    @recv1int
    @poll_recv([0], default_val=[int(512).to_bytes(4, 'little')])
    def get_height(self):
        self.__sock.send_string("get_height")

    @recv1int
    @poll_recv([0], default_val=[int(512).to_bytes(4, 'little')])
    def get_depth(self):
        self.__sock.send_string("get_depth")

    @recv1
    @poll_recv([1], default_val=["Not connected"])
    def get_serial(self):
        self.__sock.send_string("get_serial")