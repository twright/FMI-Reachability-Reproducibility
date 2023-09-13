from PyCosimLibrary.runner import CosimRunner
from src import SensitivityScenario
from fmpy.fmi2 import fmi2OK


class SensitivityJacobiRunner(CosimRunner):
    """
    This class implements the jacobi co-simulation algorithm.
    """
    def run_cosim_step(self, time, scenario: SensitivityScenario):
        for f in scenario.fmus:
            res = f.doStep(time, scenario.step_size)
            assert res == fmi2OK, "Step failed."
        self.propagate_outputs(scenario.connections)
        scenario.computeVariationalMatrix(scenario.step_size)        