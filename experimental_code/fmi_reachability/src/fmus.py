from fmpy.fmi2 import fmi2True, fmi2OK

from src.SensitivityVirtualFMU import SensitivityVirtualFMU

from fmpy.fmi2 import fmi2True, fmi2OK

class MSD1(SensitivityVirtualFMU):
    def __init__(self, instanceName):
        ref = 0
        self.x = ref;
        ref += 1
        self.v = ref;
        ref += 1
        self.m = ref;
        ref += 1
        self.c = ref;
        ref += 1
        self.cf = ref;
        ref += 1
        self.fe = ref;
        ref += 1
        self.dv = ref;
        ref += 1
        self.dx = ref;
        ref += 1

        super().__init__(instanceName, ref)

    def reset(self):
        super().reset()

        self.state[self.x] = 1.0
        self.state[self.v] = 0.0
        self.state[self.m] = 1.0
        self.state[self.c] = 1.0
        self.state[self.cf] = 1.0
        self.state[self.fe] = 0.0
        self.state[self.dv] = (1.0 / self.state[self.m]) * (- self.state[self.c] * self.state[self.x]
                                                  - self.state[self.cf] * self.state[self.v]
                                                  + self.state[self.fe])
        self.state[self.dx] = self.state[self.v]

    def doStep(self, currentCommunicationPoint, communicationStepSize, noSetFMUStatePriorToCurrentPoint=fmi2True):
        n = 10
        h = communicationStepSize / n
        for i in range(n):
            # Why not use v directly instead of creating a temp variable? More efficient
            self.state[self.dx] = self.state[self.v]
            self.state[self.dv] = (1.0 / self.state[self.m]) * (- self.state[self.c] * self.state[self.x]
                                                  - self.state[self.cf] * self.state[self.v]
                                                  + self.state[self.fe])

            self.state[self.x] = self.state[self.x] + self.state[self.dx] * h
            self.state[self.v] = self.state[self.v] + self.state[self.dv] * h

        return fmi2OK
    
    def getTimeDerivative(self, vr):
        match vr:
            case self.x:
                return self.dx
            case self.v:
                return self.dv
            case _:
                raise Exception("Variable with no time derivative")

    def getPartialDerivative(self, vr, difvr):
        if vr == self.dx:
            if difvr == self.x:
                return 0
            elif difvr == self.v:
                return 1
            elif difvr == self.fe:
                return 0
            else:
                raise Exception("Partial Derivative not Stored for" + str(vr) + " and " + str(difvr))
        elif vr == self.dv:
            if difvr == self.x:
                return -(self.state[self.c]/self.state[self.m])
            elif difvr == self.v:
                return - (self.state[self.cf]/self.state[self.m])
            elif difvr == self.fe:
                return (1.0 / self.state[self.m])
            else:
                raise Exception("Partial Derivative not Stored for" + str(vr) + " and " + str(difvr))
        else:
            raise Exception("Partial Derivative not Stored for" + str(vr) + " and " + str(difvr))

    def getInputVariables(self):
        return [self.fe]

    def getDifferentiableVariables(self):
        return list((self.dx,self.dv))
    
    def getStateVariables(self):
        return list((self.x, self.v))
    
    def hasDependency(self, vr, dvr):
        if vr == self.dx:
            match dvr:
                case self.v:
                    return True
                case _:
                    return False 
        elif vr == self.dv:
            match dvr:
                case self.v:
                    return True
                case self.x:
                    return True
                case self.fe:
                    return True
                case _:
                    return False
    

class MSD2(SensitivityVirtualFMU):
    def __init__(self, instanceName):
        ref = 0
        self.x = ref;
        ref += 1
        self.v = ref;
        ref += 1
        self.m = ref;
        ref += 1
        self.c = ref;
        ref += 1
        self.cf = ref;
        ref += 1
        self.ce = ref;
        ref += 1
        self.cef = ref;
        ref += 1
        self.fe = ref;
        ref += 1
        self.xe = ref;
        ref += 1
        self.ve = ref;
        ref += 1
        self.dv = ref;
        ref += 1
        self.dx = ref;
        ref += 1

        super().__init__(instanceName, ref)

    def reset(self):
        super().reset()

        self.state[self.x] = 0.0
        self.state[self.v] = 0.0
        self.state[self.m] = 1.0
        self.state[self.c] = 1.0
        self.state[self.cf] = 0.0
        self.state[self.ce] = 1.0
        self.state[self.cef] = 1.0
        self.state[self.fe] = 0.0
        self.state[self.xe] = 0.0
        self.state[self.ve] = 0.0
        self.state[self.dv] = (1.0 / self.state[self.m]) * (- self.state[self.c] * self.state[self.x]
                                                  - self.state[self.cf] * self.state[self.v]
                                                  - self.state[self.fe])
        self.state[self.dx] = self.state[self.v]

    def doStep(self, currentCommunicationPoint, communicationStepSize, noSetFMUStatePriorToCurrentPoint=fmi2True):
        n = 10
        h = communicationStepSize / n
        for i in range(n):
            self.state[self.fe] = self.state[self.ce] * (self.state[self.x] - self.state[self.xe]) \
                                  + self.state[self.cef] * (self.state[self.v] - self.state[self.ve])

            self.state[self.dx] = self.state[self.v]
            self.state[self.dv] = (1.0 / self.state[self.m]) * (- self.state[self.c] * self.state[self.x]
                                                  - self.state[self.cf] * self.state[self.v]
                                                  - self.state[self.fe])

            self.state[self.x] = self.state[self.x] + self.state[self.dx] * h
            self.state[self.v] = self.state[self.v] + self.state[self.dv] * h

        return fmi2OK
    
    def getTimeDerivative(self, vr):
        match vr:
            case self.x:
                return self.dx
            case self.v:
                return self.dv
            case _:
                raise Exception("Variable with no time derivative")
    
    def getPartialDerivative(self, vr, difvr):
        if vr == self.dx:
            if difvr == self.x:
                return 0
            elif difvr == self.v:
                return 1
            elif difvr == self.fe:
                return 0
            elif difvr == self.xe:
                return 0
            elif difvr == self.ve:
                return 0
            else:
                raise Exception("Partial Derivative not Stored for" + vr + " and " + difvr)
        elif vr == self.dv:
            if difvr == self.x:
                return -(self.state[self.c]/self.state[self.m])
            elif difvr == self.v:
                return - (self.state[self.cf]/self.state[self.m])
            elif difvr == self.fe:
                return (-1.0 / self.state[self.m])
            elif difvr == self.xe:
                return 0
            elif difvr == self.ve:
                return 0
            else:
                raise Exception("Partial Derivative not Stored for" + str(vr) + " and " + str(difvr))
        elif vr == self.fe:
            if difvr == self.x:
                return self.state[self.ce]
            elif difvr == self.xe:
                return -self.state[self.ce]
            elif difvr == self.v:
                return self.state[self.cef]
            elif difvr == self.ve:
                return -self.state[self.cef]
            elif difvr == self.fe:
                return 1
            else:
                raise Exception("Partial Derivative not Stored for" + str(vr) + " and " + str(difvr))
        else:
            raise Exception("Partial Derivative not Stored for" + str(vr) + " and " + str(difvr))
        
    def getInputVariables(self):
        return list((self.ve,self.xe))
    
    def getStateVariables(self):
        return list((self.x,self.v))
    
    def getDifferentiableVariables(self):
        return list((self.dx,self.dv,self.fe))

    def hasDependency(self, vr, dvr):
        if vr == self.dx:
            match dvr:
                case self.v:
                    return True
                case _:
                    return False
        elif vr == self.dv:
            match dvr:
                case self.v:
                    return True
                case self.x:
                    return True
                case self.fe:
                    return True
                case _:
                    return False
        elif vr == self.fe:
            match dvr:
                case self.v:
                    return True
                case self.x:
                    return True
                case self.xe:
                    return True
                case self.ve:
                    return True
                case _:
                    return False
        else:
            return False