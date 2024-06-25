"""
Template for writing a subclass for SLM hardware control in :mod:`slmsuite`.
Outlines which SLM superclass functions must be implemented.
"""
from slmsuite.hardware.slms.slm import SLM

class CorrectedSLM(SLM):
    """
    Wraps a SLM to include a PhaseManager so corrections are always projected. 
    """

    def __init__(
        self,
        slm,
        phase_mgr,
        **kwargs
    ):
        r"""
        Initialize SLM and attributes.

        Parameters
        ----------
        width : int
            Width of the SLM in pixels.
        height : int
            Height of the SLM in pixels.
        wav_um : float
            Wavelength of operation in microns.
        pitch_um : float
            Pitch of SLM pixels in microns.
        bitdepth : int
            Bits of phase resolution (e.g. 8 for 256 phase settings.)
        kwargs
            See :meth:`.SLM.__init__` for permissible options.

        Note
        ~~~~
        These arguments, which ultimately are used to instantiate the :class:`.SLM` superclass,
        may be more accurately filled by calling the SLM's SDK functions.
        See the other implemented SLM subclasses for examples.
        """

        # TODO: Insert code here to initialize the SLM hardware, load properties, etc.

        # Mandatory functions:
        # - Opening a connection to the device.

        # Other possibilities to consider:
        # - Setting the SLM's operating wavelength (wav_um).
        # - Updating the SLM's phase table if necessary, and/or setting the design
        #   wavelength (wav_design_um).
        # - Setting the SLM's default settle time (abstract class SLM uses
        #   settle_time_s=0.3 seconds). This is important for experimental feedback to
        #   allow the SLM to settle before viewing the result on a camera.
        # - Checking for and saving the SLM parameters (height, width, etc).

        self.slm = slm
        self.phase_mgr = phase_mgr
        # Instantiate the superclass
        super().__init__(
            self.slm.shape[1],
            self.slm.shape[0],
            bitdepth=self.slm.bitdepth,
            wav_um=self.slm.wav_um,
            dx_um=self.slm.dx_um,
            dy_um=self.slm.dy_um,
            **kwargs
        )

        # Zero the display using the superclass `write()` function.
        # self.slm.write(None)

    @staticmethod
    def info(verbose=True):
        """
        Discovers all SLMs detected by an SDK.
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
        return self.slm.info()

    def write(self, base, name="from_function_call", **kwargs):
        """

        """
        print("Calling to write to hardware")
        self.phase_mgr.set_base(base, name)
        self.slm.write(self.phase_mgr.get(), **kwargs)

    def _write_hw(self, phase):
        """
        Low-level hardware interface to write ``phase`` data onto the SLM.
        When the user calls the :meth:`.SLM.write` method of
        :class:`.SLM`, ``phase`` is error checked before calling
        :meth:`_write_hw()`. See :meth:`.SLM._write_hw` for further detail.
        """
        self.slm._write_hw(phase)
