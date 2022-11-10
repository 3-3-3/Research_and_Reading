import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as pc

class Collision:
    def __init__(self, t_f, r_1, r_2, v_1, v_2, q_1=-pc.e, q_2=pc.e, m_1=pc.m_e, m_2=pc.m_p):
        self.step = 0
        self.init = [[r_1, v_1], [r_2, v_2]]
        self.t_f = t_f

        self.r_1 = r_1; self.r_2 = r_2
        self.v_1 = v_1; self.v_2 = v_2

        self.q_1 = q_1; self.q_2 = q_2
        self.m_1 = m_1; self.m_2 = m_2

        self.m = (m_1 * m_2) / (m_1 + m_2)

        self.v_inf = np.sqrt(np.abs(np.dot(v_2 - v_1, v_2 - v_1)))
        self.b_pi_2 =  np.abs((q_1 * q_2) / (4 * np.pi * pc.epsilon_0 * self.m * self.v_inf))
        self.dt = self.b_pi_2 / (10 * self.v_inf)

        t = np.array([])
        self.r_1_hist = np.empty([500,3]); self.r_2_hist = np.empty([500,3])
        self.v_1_hist = np.empty([500,3]); self.v_2_hist = np.empty([500,3])

    def sigma_1(self):
        #Normalized E-field at q_1 due to q_2
        R = self.r_1 - self.r_2
        R_mag = np.sqrt(np.abs(np.dot(R, R)))
        return (self.q_1 / self.m_1) * self.q_2 * R / (4 * np.pi * pc.epsilon_0 * R_mag ** 3)

    def sigma_2(self):
        #Flippoty Flop
        R = self.r_2 - self.r_1
        R_mag = np.sqrt(np.abs(np.dot(R, R)))
        return (self.q_2 / self.m_2) * self.q_1 * R / (4 * np.pi * pc.epsilon_0 * R_mag ** 3)


    def next(self):
        self.v_1 = self.v_1 + self.sigma_1() * self.dt
        self.v_2 = self.v_2 + self.sigma_2() * self.dt

        self.r_1 = self.r_1 + self.v_1 * self.dt
        self.r_2 = self.r_2 + self.v_2 * self.dt

        self.r_1_hist[self.step] = self.r_1
        self.r_2_hist[self.step] = self.r_2
        self.v_1_hist[self.step] = self.v_1
        self.v_2_hist[self.step] = self.v_2

        self.step += 1

    def plot(self):
        plt.plot(self.r_1_hist[:self.step-1,1],self.r_1_hist[:self.step-1,2])
        plt.plot(self.r_2_hist[:self.step-1,1],self.r_2_hist[:self.step-1,2], color = 'red')
        plt.show()
