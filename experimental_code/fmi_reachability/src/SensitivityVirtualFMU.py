from PyCosimLibrary.virtual_fmus import VirtualFMU

class SensitivityVirtualFMU(VirtualFMU):
    def getPartialDerivative(self, vr, difvr):
        pass

    def getTimeDerivative(self, vr):
        pass
    
    # Inputs from other FMUs
    def getInputVariables(self):
        pass

    # Time derivative of state variables (for jacobian) and variables whose directional derivative are important
    def getDifferentiableVariables(self):
        pass
    
    # State variables
    def getStateVariables(self):
        pass
    
    def getRelevantVariables(self):
        return self.getDifferentiableVariables() + self.getInputVariables() + self.getStateVariables()
    
    # Returns true if variable vr depends on dvr
    def hasDependency(self, vr, dvr):
        pass
    # Required for dictionary keys
    def __hash__(self):
        return hash(self.instanceName)
    
    # Debugging
    def __str__(self):
        return self.instanceName