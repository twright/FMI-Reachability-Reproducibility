from src.SensitivityScenario import SensitivityScenario
from src.fmus import MSD1, MSD2
from src.SensitivityRunner import SensitivityJacobiRunner
from PyCosimLibrary.scenario import Connection, VarType, SignalType, OutputConnection
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


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



def msd_variational_numerical(t_array):
    y0 = [1, 0, 1, 0,
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1]
    return solve_ivp(msd_system,t_span=[0,7],t_eval=t_array, y0=y0, method='LSODA')



if __name__ == '__main__':
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
        step_size=0.01,
        print_interval=0.1,
        stop_time=7.0,
        record_inputs=True,
        outputs=out_connections)

    #paths = scenario.getDifferialPaths(msd2, msd2.dv, msd2, msd2.x)
    #print(scenario.dependency_graph)
    #for e in scenario.dependency_graph[msd2][msd2.dv]:
    #    print(e)
    # print(paths)
    #for p in paths:
    #    for l in p:
    #        print(l)
    #jacobian = scenario.getJacobian()
    #print(jacobian)
    #sol = msd_variational_numerical()
    #print(sol.y[:,0])
    #scenario.computeVariationalMatrix()
    #print(scenario.variational_matrix)
    

    runner = SensitivityJacobiRunner()
    runner.run_cosim(scenario, lambda t: None)
    #plt.plot(scenario.variational_matrix[0:][0][0])
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    scale = []
    t = 0
    for m in scenario.variational_matrix:
        scale.append(t)
        list1.append(m[0][0])
        list2.append(m[0][1])
        list3.append(m[0][2])
        list4.append(m[0][3])
        t = t + 0.01
    #plt.show()
    f1 = open("data/dx1x1fmu.txt", "w")
    for n in list1:
        f1.write(f"{n}\n")
    f1.close
    print(len(scale))
    print(scale[-1])
    msd_num = msd_variational_numerical(scale)
    f2 = open("data/dx1x1solve_ivp.txt", "w")
    for n in msd_num.y[4]:
        f2.write(f"{n}\n")
    f2.close
    plt.figure(1)
    plt.plot(scale,msd_num.y[4], label='solve_ivp')
    plt.plot(scale, list1, label='Co-Simulation')
    leg = plt.legend(loc='upper center')
    plt.xlabel('time (s)')
    plt.ylabel('Value')
    plt.savefig('figures/sensitivity_comparison.pdf')

    