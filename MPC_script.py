import numpy as np
from scipy import linalg

class MPCDesign:
    def __init__(self, Np, n, m, p):
        
        self.Np = [7.0, 7.0]
        self.F  = np.zeros([p,n])
        self.G  = np.zeros([p,m])
        self.H  = np.zeros([p,p])
        

class eMPC(MPCDesign):
    def __init__(self, Np, n, m, p):
        super().__init__(Np, n, m, p)
        
    def calculateFGMatrices(self, A, B, C, Np, n):
        
        [S,D] = linalg.eig(A) #eigen values of A matrix
        MID = np.zeros(n, dtype=complex) # initializing diagonal matrix
        
        for y in range(n):
            MID[y] = np.exp(S[y]*Np)
        
        phi = D @ np.diag(MID) @ linalg.inv(D) #calculating phi matrix
        F = np.real(C @ phi)
        G = np.real(C @ linalg.inv(A) @ (phi - np.identity(n)) @ B)
        
        return F,G
    
    def calculateGain(self, Q, R):
        
        self.H = self.G.transpose() @ Q @ self.G + R
        self.H_inv = linalg.inv(self.H)
        self.K_eMPC = linalg.inv(self.H) @ self.G.transpose() @ Q
        
    def setConstraints(self,M_con,g_con):
        self.M_con = M_con
        self.g_con = g_con
        
    def assignFGMatrices(self, F, G):
        self.F = F
        self.G = G
        
    def constraintsSatisfied(self, U, M, g):
        
        result = True
        LHS = M @ U
        for i in range(len(g)):
            if (LHS[i] >= g[i]):
                result = False
                break
        return result