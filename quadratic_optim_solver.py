import numpy as np
import copy
import math
import matplotlib.pyplot as plt

class QuadProg:
    def __init__(self, M_con, H_inv, g_con, f):
        self.W = (M_con @ H_inv @ M_con.T) * 0.5
        self.Z = np.squeeze(M_con @ H_inv @ f + g_con.transpose())
        self.r = np.shape(self.Z)[0]
        self.i_t = 0
        self.err = np.empty(1)
        self.err[self.i_t] = 1
        self.lam = np.zeros(self.r)
        
    def plotResults(self, U):
        print(f"\nConstrained Inputs:", U[0],U[1], "after number of iterations:", self.i_t)
        x = list(range(0, self.i_t))
        y = self.err[:-1]
        plt.plot(x,y)
        plt.show()
        
class PQP(QuadProg):
    def __init__(self, M_con, H_inv, g_con, f):
        super().__init__(M_con, H_inv, g_con,f)
        self.lam = np.ones(self.r)
        self.Zm = np.maximum(-self.Z,0)
        self.Zp = np.maximum(self.Z,0)
        r_vec = np.maximum(-self.W,0) @ np.zeros(self.r)
        self.Wm = np.maximum(-self.W,0) + np.diag(r_vec)
        self.Wp = np.maximum(self.W,0) + np.diag(r_vec)
        
    def optimize(self):
        while self.err[self.i_t] > 1e-6:
            old = copy.deepcopy(self.lam)
            for i in range(self.r):
                km = self.Wm @ self.lam
                kp = self.Wp @ self.lam
                denominator = self.Zp[i] + kp[i]
                if denominator != 0:
                    self.lam[i] = self.lam[i] * ((self.Zm[i] + km[i]) / denominator)

            err = np.dot((self.lam - old).T, (self.lam - old))
            self.i_t += 1
            self.err = np.append(self.err, err)
