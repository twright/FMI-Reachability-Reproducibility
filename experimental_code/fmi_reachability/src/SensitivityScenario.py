from PyCosimLibrary.scenario import CosimScenario
from src.SensitivityVirtualFMU import SensitivityVirtualFMU
import copy
from numpy import identity, matmul, asarray, add



class FMUElementPair():
    fmu: SensitivityVirtualFMU
    vr: int

    def __init__(self, fmu, vr):
        self.fmu=fmu
        self.vr=vr

    def __str__(self):
        return '(' + str(self.fmu) + ',' + str(self.vr) + ')'

class SensitivityScenario(CosimScenario):
    
    def __init__(self, **args):
        super().__init__(**args)
        self.dependency_graph = self.create_dependency_graph()
        dim = 0
        for f in self.fmus:
            dim+=len(f.getStateVariables())
        self.variational_matrix = []
        self.variational_matrix.append(identity(dim))
        

    def create_dependency_graph(self):
        dependency_graph={}
        for fmu in self.fmus:
            # First we create a dictionary for each fmu
            # This is done due to variables being stored as an int in each fmu
            dependency_graph[fmu]={}

            for var in fmu.getRelevantVariables():
                # Each variable in the dictionary represents a node of the graph with a list of pairs (fmu, vr) as vertexes
                # Add the vertexes from explicit dependencies on the Jacobian
                dependency_graph[fmu][var]=list()
                # If the variable has a time derivative relevant, check which variables influence on it
                if var in fmu.getDifferentiableVariables():    
                    for dependent_var in fmu.getRelevantVariables():
                        if fmu.hasDependency(var, dependent_var):
                            dependency_graph[fmu][var].append(FMUElementPair(fmu,dependent_var))
                # Add the vertexes from connections to other fmus 
                if var in fmu.getInputVariables():
                    for con in self.connections:
                        if var in con.target_vr:
                            dependency_graph[fmu][var].append(FMUElementPair(con.source_fmu,con.source_vr[con.target_vr.index(var)]))
            
        return dependency_graph
        

    def getDifferialPaths(self, source_fmu, vr, target_fmu, dvr):
        # Initialize the queue that stores the nodes that we need to check
        queue = [FMUElementPair(source_fmu, vr)]
        # List of all correct paths
        paths = []
        # Path that we are checking
        temp_path = []
        # Visited nodes to avoid loops
        visited = []
        # Flag to check if a path is a dead end
        isDeadEnd = True
        while len(queue) != 0:
            # We get the first element of the queue
            candidate = queue[len(queue)-1]
            # Add it to the temporal path
            temp_path.append(candidate)
            # Add it to the list of visited nodes
            visited.append(candidate)
            # Remove it from the queue
            queue.pop()

            # Check if the path has reached the destination
            if candidate.fmu==target_fmu and candidate.vr == dvr:
                paths.append(copy.copy(temp_path))
                # If it has we do not add its connections to the list
            else:
                # Add non-visited connections of candidate element to the queue
                for e in self.dependency_graph[candidate.fmu][candidate.vr]:
                    if e not in visited:
                        queue.append(e)
                        isDeadEnd=False
            
            # If we reach a dead end prune the prath
            if isDeadEnd==True and len(queue) !=0:
                # We remove the last element of the temporal path
                temp_path.pop()
                # We check if we need to remove more
                n_candidate = queue[len(queue)-1]
                while n_candidate not in self.dependency_graph[temp_path[len(temp_path)-1].fmu][temp_path[len(temp_path)-1].vr]:
                    temp_path.pop()
            else:
                # Reset the dead end flag
                isDeadEnd=True

        return paths

    def getJacobianElement(self, source_fmu, vr, target_fmu, dvr):
        # We get all the paths between the variables
        paths=self.getDifferialPaths(source_fmu, vr, target_fmu, dvr)
        sum = 0
        temp_product = 1
        for p in paths:
            # Multiply derivatives of elements in the same path
            for i in range(len(p)-1):
                # If two nodes are from different fmus, their derivative is 1
                if p[i].fmu == p[i+1].fmu:
                    temp_product = temp_product * p[i].fmu.getPartialDerivative(p[i].vr,p[i+1].vr)
            # Add for every path
            sum = sum + temp_product
            temp_product = 1
        
        return sum

    def getJacobian(self):
        state_variables = []
        time_derivatives = []
        jacobian = []
        temp_list = []

        # We get a list with all state variables
        for fmu in self.fmus:
            for v in fmu.getStateVariables():
                state_variables.append(FMUElementPair(fmu, v))
        
        # We get a list with all time derivatives from state variables
        for v in state_variables:
                time_derivatives.append(FMUElementPair(v.fmu, v.fmu.getTimeDerivative(v.vr)))

        # We get the jacobian matrix
        for dv in time_derivatives:    
            for v in state_variables:
                temp_list.append(self.getJacobianElement(dv.fmu, dv.vr, v.fmu, v.vr))
            jacobian.append(copy.copy(temp_list))
            temp_list = []

        return asarray(jacobian)

    def computeVariationalMatrix(self,time_step):
        m = add(self.variational_matrix[-1],(matmul(self.getJacobian(),self.variational_matrix[-1])*time_step))
        self.variational_matrix.append(m.copy())