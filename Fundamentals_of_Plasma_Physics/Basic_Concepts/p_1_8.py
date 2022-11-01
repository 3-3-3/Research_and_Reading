import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as pc

def E(r_1, r_2, q_1):
    #E from relative positions of particles, charge
    r = np.sqrt(np.dot(r_1, r_2))
    r_hat = (r_2 - r_1) / r #unit vector in the direction of r_2 from r_1
    return q_1 / (4 * np.pi * pc.epsilon_0 * r ** 2) * r_hat

def Sigma(q,m,E):
    #Normalized electric field vector
    return q * E / m

def v_next(dt, v_old, Sigma):
    return v_old + Sigma * dt

def simulation(t_f, r_1, r_2, v_1, v_2, q_1, q_2, m_1, m_2):
    m = (m_1 * m_2) / (m_1 + m_2) #reduced mass of the system
    v_inf = v_2 - v_1 #characteristic speed; speeds are in +/- z
    b_pi_2 =  (q_1 * q_2) / (4 * np.pi * pc.epsilon_0 * m * v_inf) #characteristic length; 90 degree impact paramter, as found in 1.1
    dt = b_pi_2 / v_inf #characteristic time, time it takes particle to move characteristic length at characteristic speed

    steps = int(t_f / dt)
    t = np.linspace(0, t_f, steps) #times to calculate next position, velocity; multiples of characteristic time
    p_1 = (np.array(r_1),np.array(v_1)) #particle 1 info; position, velocity
    p_2 = (np.array(r_2),np.array(v_2)) #particle 2 info; position, velocity

    for dt in t:
        r_1_old = p_1[0][-1]; v_1_old = p_1[1][-1]
        r_2_old = p_2[0][-1]; v_2_old = p_2[1][-1]

        E_1_2 = E(r_2_old, r_1_old, q_2)
        E_2_1 = E(r_1_old, r_2_old, q_1)

        v_1_new = v_next(dt, v_2_old, Sigma(q_1, m_1, E_1_2))
        v_2_new = v_next(dt, v_1_old, Sigma(q_2, m_2, E_2_1))

        r_1_new = r_1_old + v_1_new * dt
        r_2_new = r_2_old + v_2_new * dt

        p_1[0] = np.append(p_1[0], r_1_new, axis=0); p_1[1] = np.append(p_1[1], v_1_new, axis=0)
        p_2[0] = np.append(p_2[0], r_2_new, axis=0); p_2[1] = np.append(p_2[1], v_2_new, axis=0)

    return (p_1, p_2)
