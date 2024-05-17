import itertools

from ema_workbench.em_framework.samplers import *
from ema_workbench.util import EMAError
class PartialFactorialDesigns(object):

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, value):
        self._kind = value
        self.ff_designs.kind = value
        self.other_designs.kind = value

    def __init__(self, ff_designs, other_designs, parameters, n):
        self.ff_designs = ff_designs
        self.other_designs = other_designs

        self.parameters = parameters
        self.params = [p.name for p in parameters]

        self._kind = None
        self.n = n

    def __iter__(self):
        designs = itertools.product(self.ff_designs, self.other_designs)
        return partial_designs_generator(designs)


def partial_designs_generator(designs):
    """generator which combines the full factorial part of the design
    with the non full factorial part into a single dict

    Parameters
    ----------
    designs: iterable of tuples

    Yields
    ------
    dict
        experimental design dict

    """

    for design in designs:
        try:
            ff_part, other_part = design
        except ValueError:
            ff_part = design
            other_part = {}

        design = ff_part.copy()
        design.update(other_part)

        yield design


class PartialFactorialSampler(AbstractSampler):
    """
    generates a partial factorial design over the parameters. Any parameter
    where factorial is true will be included in a factorial design, while the
    remainder will be sampled using LHS or MC sampling.

    Parameters
    ----------
    sampling: {PartialFactorialSampler.LHS, PartialFactorialSampler.MC}, optional
              the desired sampling for the non factorial parameters.

    Raises
    ------
    ValueError
        if sampling is not either LHS or MC

    """

    LHS = 'LHS'
    MC = 'MC'

    def __init__(self, sampling='LHS'):
        super(PartialFactorialSampler, self).__init__()

        if sampling == PartialFactorialSampler.LHS:
            self.sampler = LHSSampler()
        elif sampling == PartialFactorialSampler.MC:
            self.sampler = MonteCarloSampler()
        else:
            raise ValueError(('invalid value for sampling type, should be LHS '
                              'or MC'))
        self.ff = FullFactorialSampler()

    def _sort_parameters(self, parameters):
        """sort parameters into full factorial and other

        Parameters
        ----------
        parameters : list of parameters

        """
        ff_params = []
        other_params = []

        for param in parameters:
            if param.pff:
                ff_params.append(param)
            else:
                other_params.append(param)

        if not ff_params:
            raise EMAError("no parameters for full factorial sampling")
        if not other_params:
            raise EMAError("no parameters for normal sampling")

        return ff_params, other_params

    def generate_designs(self, parameters, nr_samples):
        """external interface to sampler. Returns the computational experiments
        over the specified parameters, for the given number of samples for each
        parameter.

        Parameters
        ----------
        parameters : list
                        a list of parameters for which to generate the
                        experimental designs
        nr_samples : int
                     the number of samples to draw for each parameter

        Returns
        -------
        generator
            a generator object that yields the designs resulting from
            combining the parameters
        int
            the number of experimental designs

        """

        ff_params, other_params = self._sort_parameters(parameters)

        # generate a design over the factorials
        # TODO update ff to use resolution if present
        ff_designs = self.ff.generate_designs(ff_params, nr_samples)

        # generate a design over the remainder
        # for each factorial, run the MC design
        other_designs = self.sampler.generate_designs(other_params,
                                                      nr_samples)

        nr_designs = other_designs.n * ff_designs.n

        designs = PartialFactorialDesigns(ff_designs, other_designs,
                                          ff_params + other_params, nr_designs)

        return designs
