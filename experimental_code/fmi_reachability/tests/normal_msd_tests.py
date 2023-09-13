import unittest

from PyCosimLibrary.gauss_seidel_runner import GaussSeidelRunner
from PyCosimLibrary.jacobi_runner import JacobiRunner
from PyCosimLibrary.jacobit_it_runner import JacobiIterativeRunner
from PyCosimLibrary.scenario import Connection, VarType, SignalType, OutputConnection, CosimScenario
from src.fmus import *
import matplotlib.pyplot as plt


class NormalMSDTestSuite(unittest.TestCase):
    """Basic test cases."""

    def build_double_msd_scenario(self):
        msd1 = MSD1("msd1")
        msd1.instantiate()
        msd2 = MSD2("msd2")
        msd2.instantiate()

        msd1_out = Connection(value_type=VarType.REAL,
                              signal_type=SignalType.CONTINUOUS,
                              source_fmu=msd1,
                              target_fmu=msd2,
                              source_vr=[msd1.x, msd1.v],
                              target_vr=[msd2.xe, msd2.ve])
        msd1_in = Connection(value_type=VarType.REAL,
                             signal_type=SignalType.CONTINUOUS,
                             source_fmu=msd2,
                             target_fmu=msd1,
                             source_vr=[msd2.fe],
                             target_vr=[msd1.fe])
        msd2_out = OutputConnection(value_type=VarType.REAL,
                                    signal_type=SignalType.CONTINUOUS,
                                    source_fmu=msd2,
                                    source_vr=[msd2.x, msd2.v])

        connections = [msd1_out, msd1_in]
        out_connections = [msd1_out, msd1_in, msd2_out]
        scenario = CosimScenario(
            fmus=[msd1, msd2],
            connections=connections,
            step_size=0.01,
            print_interval=0.1,
            stop_time=7.0,
            record_inputs=True,
            outputs=out_connections)
        return scenario

    def test_run_dmsd_jacobiIt(self):
        scenario = self.build_double_msd_scenario()

        runner = JacobiIterativeRunner(100, 1e-4)

        results = runner.run_cosim(scenario, lambda t: None)

        for f in scenario.fmus:
            f.terminate()

        msd1 = next(f for f in scenario.fmus if f.instanceName == "msd1")
        msd2 = next(f for f in scenario.fmus if f.instanceName == "msd2")

        self.assertTrue(results.timestamps[-1] > 6.0)
        self.assertTrue(results.out_signals[msd1.instanceName][msd1.x][-1] > -1.0)

    def test_run_jacobi(self):
        scenario = self.build_double_msd_scenario()

        jacobi = JacobiRunner()

        results = jacobi.run_cosim(scenario, lambda t: None)

        for f in scenario.fmus:
            f.terminate()

        msd1 = next(f for f in scenario.fmus if f.instanceName == "msd1")
        msd2 = next(f for f in scenario.fmus if f.instanceName == "msd2")

        self.assertTrue(results.timestamps[-1] > 6.0)
        self.assertTrue(results.out_signals[msd1.instanceName][msd1.x][-1] > -1.0)

    def test_run_gauss_seidal(self):
        scenario = self.build_double_msd_scenario()

        runner = GaussSeidelRunner()

        results = runner.run_cosim(scenario, lambda t: None)

        for f in scenario.fmus:
            f.terminate()

        msd1 = next(f for f in scenario.fmus if f.instanceName == "msd1")
        msd2 = next(f for f in scenario.fmus if f.instanceName == "msd2")

        self.assertTrue(results.timestamps[-1] > 6.0)
        self.assertTrue(results.out_signals[msd1.instanceName][msd1.x][-1] > -1.0)


if __name__ == '__main__':
    unittest.main()
