from sidmpy.CrossSections.cross_section import InteractionCrossSection

class VelocityIndependentCrossSection(InteractionCrossSection):

    def __init__(self, norm):
        """
        This class implements a velocity-independent cross section with a constant value specified by norm

        :param norm: the cross section normalization in cm^2 / gram
        """

        super(VelocityIndependentCrossSection, self).__init__(norm, self._velocity_dependence_kernel)

    def _velocity_dependence_kernel(self, v):

        return 1.

