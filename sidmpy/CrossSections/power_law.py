from sidmpy.CrossSections.cross_section import InteractionCrossSection

class PowerLaw(InteractionCrossSection):

    def __init__(self, norm, v_ref, v_dep):

        """
        This class implements a velocity-dependent cross section of the form

        sigma(v) = cross0 * (v_ref / v) ^ v_dep

        i.e. a power law in velocity normalized to cross0 at v = v_ref with a logarithmic slope v_dep

        :param cross0: the cross section normalization in cm^2 / gram
        :param v_ref: a reference velocity (dwarf scale is ~ 30 km/sec)
        :param v_dep: the logarithmic slope of the velocity dependence, negative values are unphysical
        """

        if v_dep < 0:
            raise Exception('you have specified v_dep = '+str(v_dep) + ', this is unphysical.')
        self.v_pow = v_dep
        self.v_ref = v_ref

        super(PowerLaw, self).__init__(norm, self._velocity_dependence_kernel)

    @property
    def kwargs(self):
        """
        Returns the keyword arguments for this cross section model
        """
        return {'norm': self.norm, 'v_ref': self.v_ref, 'v_dep': self.v_pow}

    def _velocity_dependence_kernel(self, v):

        """
        evaluates the velocity dependence of the cross section
        """
        r = self.v_ref / v
        return r ** self.v_pow

