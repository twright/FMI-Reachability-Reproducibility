import unittest

from src.SensitivityRunner import SensitivityJacobiRunner
from src.SensitivityScenario import SensitivityScenario
from src.fmus import MSD1, MSD2
from PyCosimLibrary.scenario import Connection, VarType, SignalType, OutputConnection
from src.fmus import *
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np


def msd_system(t, state):
    x1, v1, x2, v2, dx1x1, dx1v1, dx1x2, dx1v2, dv1x1, dv1v1, dv1x2, dv1v2, dx2x1, dx2v1, dx2x2, dx2v2, dv2x1, dv2v1, dv2x2, dv2v2 = state

    dx1 = v1
    dv1 = -x1 - v1 + x1 - x2 + v2 - v1
    dx2 = v2
    dv2 = -x2 + x1 - x2 + v1 - v2
    ddx1x1 = dv1x1
    ddx1v1 = dv1v1
    ddx1x2 = dv1x2
    ddx1v2 = dv1v2
    ddv1x1 = -2*dx1x1 -2*dv1x1 + dx2x1 + dv2x1
    ddv1v1 = -2*dx1v1 -2*dv1v1 + dx2v1 + dv2v1
    ddv1x2 = -2*dx1x2 -2*dv1x2 + dx2x2 + dv2x2
    ddv1v2 = -2*dx1v2 -2*dv1v2 + dx2v2 + dv2v2
    ddx2x1 = dv2x1
    ddx2v1 = dv2v1
    ddx2x2 = dv2x2
    ddx2v2 = dv2v2
    ddv2x1 = dx1x1 + dv1x1 - 2*dx2x1 - dv2x1
    ddv2v1 = dx1v1 + dv1v1 - 2*dx2v1 - dv2v1
    ddv2x2 = dx1x2 + dv1x2 - 2*dx2x2 - dv2x2
    ddv2v2 = dx1v2 + dv1v2 - 2*dx2v2 - dv2v2

    return [dx1, dv1, dx2, dv2,
            ddx1x1, ddx1v1, ddx1x2, ddx1v2,
            ddv1x1, ddv1v1, ddv1x2, ddv1v2,
            ddx2x1, ddx2v1, ddx2x2, ddx2v2,
            ddv2x1, ddv2v1, ddv2x2, ddv2v2]



def msd_variational_numerical(scale):
    y0 = [1, 0, 1, 0,
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1]
    return solve_ivp(msd_system,t_span=[0,7],t_eval=scale, y0=y0, method='LSODA')




class SensitivityMSDTestSuite(unittest.TestCase):
    """Basic test cases."""
    def build_double_msd_sensitivity_scenario(self, step_time):
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
        scenario = SensitivityScenario(
            fmus=[msd1, msd2],
            connections=connections,
            step_size=step_time,
            print_interval=0.1,
            stop_time=7.0,
            record_inputs=True,
            outputs=out_connections)
        
        return scenario
    
    def test_graph_builder(self):
        scenario = self.build_double_msd_sensitivity_scenario(0.01)
        msd2 = next(f for f in scenario.fmus if f.instanceName == "msd2")
        self.assertTrue(len(scenario.dependency_graph[msd2][msd2.dv])==3)
        self.assertTrue(scenario.dependency_graph[msd2][msd2.dv][0].vr in [msd2.v,msd2.x,msd2.fe])
        self.assertTrue(scenario.dependency_graph[msd2][msd2.dv][1].vr in [msd2.v,msd2.x,msd2.fe])
        self.assertTrue(scenario.dependency_graph[msd2][msd2.dv][2].vr in [msd2.v,msd2.x,msd2.fe])

    def test_path_builder(self):
        scenario = self.build_double_msd_sensitivity_scenario(0.01)
        msd2 = next(f for f in scenario.fmus if f.instanceName == "msd2")
        paths = scenario.getDifferialPaths(msd2,msd2.dv,msd2,msd2.x)
        self.assertTrue(paths[0][0].fmu == msd2 and paths[0][0].vr == msd2.dv)
        self.assertTrue(paths[0][1].fmu == msd2 and paths[0][1].vr == msd2.x)
        self.assertTrue(paths[1][0].fmu == msd2 and paths[1][0].vr == msd2.dv)
        self.assertTrue(paths[1][1].fmu == msd2 and paths[1][1].vr == msd2.fe)
        self.assertTrue(paths[1][2].fmu == msd2 and paths[1][2].vr == msd2.x)

    def test_jacobian(self):
        scenario = self.build_double_msd_sensitivity_scenario(0.01)
        self.assertTrue(scenario.getJacobian()[1][1]==-2)
        self.assertTrue(scenario.getJacobian()[3][2]==-2)

    def test_run_sensitivity_dmsd_variational(self):
        scenario = self.build_double_msd_sensitivity_scenario(0.01)

        runner = SensitivityJacobiRunner()

        runner.run_cosim_step(0, scenario) # TODO: use runner.runcosim()
        numerical_sim = msd_variational_numerical([1e-3, 2e-3])

        
        self.assertTrue(abs(scenario.variational_matrix[1][0][0]-numerical_sim.y[:,0][4])<1e-2)
        self.assertTrue(abs(scenario.variational_matrix[1][0][1]-numerical_sim.y[:,0][5])<1e-2)


        runner.run_cosim_step(0, scenario)

        self.assertTrue(abs(scenario.variational_matrix[2][0][0]-numerical_sim.y[:,1][4])<1e-2)
        self.assertTrue(abs(scenario.variational_matrix[2][0][1]-numerical_sim.y[:,1][5])<2e-2)
    
    def get_error_by_step_size(self, step_size):
        scenario = self.build_double_msd_sensitivity_scenario(step_time=step_size)
        runner = SensitivityJacobiRunner()
        runner.run_cosim(scenario, lambda t: None)
        scale = []
        t = 0
        for m in scenario.variational_matrix:
            scale.append(t)
            t = t + step_size
        
        msd_num = msd_variational_numerical(scale)
        error = []
    
        for i in range(len(scenario.variational_matrix)):
            sub = np.subtract(scenario.variational_matrix[i],msd_num.y[:,i][4:].reshape((4,4)))
            error.append(np.linalg.norm(sub,2))


        return [error,scale]

    def test_run_validation(self):
        error_scale2 = self.get_error_by_step_size(0.5)
        error_scale3 = self.get_error_by_step_size(0.1)
        error_scale4 = self.get_error_by_step_size(0.05)
        error_scale5 = self.get_error_by_step_size(0.01)

        f = open("data/error05.txt", "w")
        for n in error_scale2[0]:
            f.write(f"{n}\n")
        f.close

        f = open("data/error01.txt", "w")
        for n in error_scale3[0]:
            f.write(f"{n}\n")
        f.close

        f = open("data/error005.txt", "w")
        for n in error_scale4[0]:
            f.write(f"{n}\n")
        f.close

        f = open("data/error001.txt", "w")
        for n in error_scale5[0]:
            f.write(f"{n}\n")
        f.close


        #plt.plot(error_scale1[1],error_scale1[0])
        plt.plot(error_scale2[1],error_scale2[0], label='Step size 0.5')
        plt.plot(error_scale3[1],error_scale3[0], label='Step size 0.1')
        plt.plot(error_scale4[1],error_scale4[0], label='Step size 0.05')
        plt.plot(error_scale5[1],error_scale5[0], label='Step size 0.01')

        leg = plt.legend(loc='upper center')
        plt.xlabel('Time (s)', fontsize=13)
        plt.ylabel('error(t)', fontsize=13)

        plt.savefig('figures/error_comparison.pdf')

if __name__ == '__main__':
    unittest.main()
