import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as pc


def A(dt, c_freq):
    return c_freq * dt / 2

def C(dt, v_old, n_E, c_freq):
    return v_old + dt * (n_E + np.cross(v_old, c_freq / 2))

def v_next(A, C):
    return (C + A * (np.dot(A, C)) - np.cross(A, C)) / (1 + np.dot(A, A))

def simulation(r_0, v_0, dt=1e-2, B=1, E=1, steps=100):
    c_freq = np.array([0, 0, B]) #choose units where e/m_e = 1 :-)
    n_E = np.array([E, 0, 0])

    t_out = np.array([0])
    v_out = np.array([v_0])
    r_out = np.array([r_0])



    for n in range(steps):
        #Old values are the last value in the out arrays
        r_old = r_out[-1]
        v_old = v_out[-1]
        t_old = t_out[-1]

        #Calculate new values
        t_new = t_old + dt
        a = A(dt, c_freq)
        c = C(dt, v_old, n_E, c_freq)
        v_new = v_next(a, c)
        r_new = r_old + v_new * dt

        r_out = np.append(r_out, [r_new], axis=0)
        v_out = np.append(v_out, [v_new], axis=0)
        t_out = np.append(t_out, t_new)

        print(f'Completed step {n}. r: {r_new}, v: {v_new}, t: {t_new}')

    return (t_out, v_out, r_out)


def analytic(t, B=1, E=1): #analytic solution for r_0 = v_0 = 0
    c_freq = B #choose units where e/m_e = 1 :-)
    n_E = E

    x = n_E / (c_freq ** 2) * (1 - np.cos(c_freq * t))
    y = -n_E / c_freq * (t - 1 / c_freq * np.sin(c_freq * t))
    z = np.zeros(x.size)

    r_out = np.column_stack((x, y, z))

    return (t, r_out)

if __name__ == '__main__':
    dt = 0.01
    steps = 1000
    t_f = dt * steps
    t = np.linspace(0, t_f, steps + 1)

    sim = simulation(np.array([0,0,0]), np.array([0,0,0]), B=5, E=1, dt=dt,steps=steps)
    anal = analytic(t,B=5,E=1)

    #Plot numeric solution
    r = sim[2] #position vector
    print(r.shape)
    r_x = r[:,0]
    r_y = r[:,1]
    plt.plot(r_x,r_y,color='red',label='Numerical')

    #Plot analytic solution
    r_anal = anal[1]
    r_x_anal = r_anal[:,0]
    print(r_anal.shape)
    r_y_anal = r_anal[:,1]
    plt.plot(r_x_anal, r_y_anal, color='blue', label='Analytic')

    plt.grid()
    plt.legend()
    plt.title('Numerical and Analytic Motion (Indistinguishable)')
    plt.savefig('p_1_7.png')
    plt.show()

    #Take a look at the error function
    error = r_anal - r
    print(error.shape)
    plt.plot(t,error[:,0],label='x Error')
    plt.plot(t,error[:,1],label='y Error')
    plt.grid()
    plt.legend()
    plt.title('Error')
    plt.savefig('p_1_7_error.png')
    plt.show()
