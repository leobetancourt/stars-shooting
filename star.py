import constants as const
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from opal import Opal


class Star:
    def __init__(self, M=1, composition=(0.7, 0.28, 0.02), m_fit=0.3, res=10000):
        # composition
        self.X, self.Y, self.Z = composition
        self.X_CNO = self.Z
        self.mu = 1 / (2 * self.X + (3 * self.Y / 4) + self.Z / 2)

        # opacity
        self.opal = Opal(composition)
        self.kappa_thomson = 0.2 * (1 + self.X)

        # set initial parameters
        self.m_c, self.m_s = 1e-6, (2e33)*M
        self.r_c, self.r_s = 1e-6, 7e10*(M**0.8)
        self.P_c, self.P_s = 2.5e17, self.P_photo(self.m_s, self.r_s, self.kappa_thomson)
        self.L_c, self.L_s = 1e-6, 4e33
        self.T_c, self.T_s = 1.5e7, self.T_eff(self.L_s, self.r_s)
        if M > 5:
            self.r_s = 2e11
            self.P_c = 4.7e16
            self.L_s = 7e36
            self.T_c = 2.8e7

        # shooting method parameters
        self.m_fit, self.res = m_fit, res
        self.m_o = np.linspace(self.m_c, (self.m_s) * self.m_fit, self.res)
        self.m_i = np.linspace(self.m_s, (self.m_s) * self.m_fit, self.res)

    """ helper methods """

    def rho(self, P, T):
        return (self.mu / const.R_g) * (P / T)

    def P_photo(self, M, R, kappa=0.4):
        return (2 * const.G * M) / (3 * (R ** 2) * kappa)

    @staticmethod
    def T_eff(L, R):
        return (L / (4 * np.pi * (R ** 2) * const.sigma_sb)) ** (1/4)

    def schwarzschild(self, P, T, dT_rad, dP):
        nabla_ad = (const.gamma - 1) / const.gamma
        return ((P / T) * (dT_rad / dP)) < nabla_ad

    def epsilon_pp(self, rho, T):
        # if not isinstance(T, np.ndarray) and T < 1e6:
        #     return 0
        T_6 = 1e6 / T
        return (2.4e6) * rho * (self.X ** 2) * (T_6 ** (2/3)) * np.exp(-33.8 * (T_6 ** (1/3)))

    def epsilon_cno(self, rho, T):
        # if not isinstance(T, np.ndarray) and T < 1e6:
        #     return 0
        T_6 = 1e6 / T
        return (8.7e27) * self.X * self.X_CNO * rho * (T_6 ** (2/3)) * np.exp(-152.3 * (T_6 ** (1/3)))

    def radiative(self, r, T, L, kappa):
        return - (3 * L * kappa) / (64 * (np.pi ** 2) * const.a * const.c * (T ** 3) * (r ** 4))

    def convective(self, P, T, dP):
        return ((const.gamma - 1) / const.gamma) * (T / P) * dP

    """ stellar structure equations """

    def stellar_structure(self, m, Y):
        P, r, T, L = Y
        _rho = self.rho(P, T)

        kappa = self.kappa_thomson # self.opal.get_opacity(_rho, T)

        dP = - (const.G * m) / (4 * np.pi * r ** 4)
        dr = 1 / (4 * np.pi * (r ** 2) * _rho)
        dT_rad = self.radiative(r, T, L, kappa)
        dT_conv = self.convective(P, T, dP)
        dL = self.epsilon_pp(_rho, T) + self.epsilon_cno(_rho, T)

        stable = self.schwarzschild(P, T, dT_rad, dP)
        dT = dT_rad if stable else dT_conv

        return dP, dr, dT, dL

    # set initial boundary conditions to solve
    # (central pressure, radius, central temperature, luminosity)
    def set_initial(self, P, R, T, L):
        self.P_c, self.r_s, self.T_c, self.L_s = P, R, T, L

    """ shooting method """

    # returns outward and inward integrations of the stellar structure equations
    def shooting(self):
        int_o = solve_ivp(self.stellar_structure, (self.m_c, (self.m_s)*self.m_fit),
                          (self.P_c, self.r_c, self.T_c, self.L_c), t_eval=self.m_o)
        int_i = solve_ivp(self.stellar_structure, (self.m_s, (self.m_s)*self.m_fit),
                          (self.P_s, self.r_s, self.T_s, self.L_s), t_eval=self.m_i)

        return int_o.y, int_i.y

    def solve(self, log=False):
        fit_mass = self.m_fit * self.m_s
        tol = 1e-6

        frac_error, error = np.ones(4), np.ones(4)
        max_iter = 100000
        n = 0
        E = np.array([self.P_c, self.r_s, self.T_c, self.L_s])
        while np.max(np.abs(frac_error)) > tol and n < max_iter:
            # outward and inward integrations
            Y_o, Y_i = self.shooting()
            Y_o_f, Y_i_f = Y_o[:, -1], Y_i[:, -1]
            error = Y_i_f - Y_o_f  # error at fitting point
            frac_error = error / E
            if log:
                print("max error:", np.max(np.abs(frac_error)), E, end='\r')

            dYdE = np.zeros((4, 4))
            # 'wiggle' each parameter to calculate resulting change in Y
            for i in range(len(E)):
                outward = i % 2 == 0
                e = np.copy(E)
                delta_e = e[i] * 0.01
                e[i] += delta_e

                m_span = (self.m_c, fit_mass) if outward else (
                    self.m_s, fit_mass)
                Y_0 = (e[0], self.r_c, e[2], self.L_c) if outward else (
                    self.P_s, e[1], self.T_s, e[3])
                Y_f_new = solve_ivp(self.stellar_structure,
                                    m_span, Y_0).y[:, -1]
                delta_Y = Y_f_new - Y_o_f if outward else Y_i_f - Y_f_new
                dYdE[i] = delta_Y / delta_e

            # solve system of equations for delta_E
            dYdE = np.transpose(dYdE)
            delta_E = np.linalg.solve(dYdE, error)
            E += (delta_E * (0.1))
            self.P_c, self.r_s, self.T_c, self.L_s = E

            # update photospheric boundary conditions
            kappa = self.opal.get_opacity(
                self.rho(self.P_s, self.T_s), self.T_s)
            
            if math.isnan(kappa):
                kappa = self.kappa_thomson

            self.P_s = self.P_photo(self.m_s, self.r_s, kappa)
            self.T_s = self.T_eff(self.L_s, self.r_s)

            n += 1
        if log:
            print()
            print("Solution:", E)

        return E

    """ plotting """

    def plot(self, radius=False):
        y_labels = np.array([f"$P$", f"$r$", f"$T$", f"$L$"])

        fig, axes = plt.subplots(figsize=(12, 12), ncols=2, nrows=2)
        Y_o, Y_i = self.shooting()

        # if not int_o.success or not int_i.success:
        #     print("Integration failed")
        #     return

        for i, ax in enumerate(axes.ravel()):
            ax.set_ylabel(y_labels[i])
            if radius:
                if i == 1:  # plot r vs m
                    ax.plot(Y_o[1], self.m_o, c="b")
                    ax.plot(Y_i[1], self.m_i, c="orange")
                    ax.set_ylabel(f"$m$")
                else:
                    ax.plot(Y_o[1], Y_o[i], c="b")
                    ax.plot(Y_i[1], Y_i[i], c="orange")
                ax.set_xlabel(f"$r$")
            else:
                ax.plot(self.m_o, Y_o[i], c="b")
                ax.plot(self.m_i, Y_i[i], c="orange")
                ax.set_xlabel(f"$m$")
        plt.show()

        self.plot_schwarzschild()

        # opacity
        rho_o = self.rho(Y_o[0], Y_o[2])
        kappa_o = [self.opal.get_opacity(rho_o[i], Y_o[2][i])
                   for i in range(len(rho_o))]
        rho_i = self.rho(Y_i[0], Y_i[2])
        kappa_i = [self.opal.get_opacity(rho_i[i], Y_i[2][i])
                   for i in range(len(rho_i))]
        plt.plot(self.m_o, kappa_o)
        plt.plot(self.m_i, kappa_i)
        plt.xlabel(r"$m$")
        plt.ylabel(r"$\kappa$")
        plt.show()

        # nuclear burning
        eps_cno_o = self.epsilon_cno(rho_o, Y_o[2])
        eps_cno_i = self.epsilon_cno(rho_i, Y_i[2])
        eps_pp_o = self.epsilon_pp(rho_o, Y_o[2])
        eps_pp_i = self.epsilon_pp(rho_i, Y_i[2])
        logT_o = np.log10(Y_o[2])
        logT_i = np.log10(Y_i[2])
        plt.scatter(logT_o, np.log10(eps_cno_o), s=0.1, color="b", label=r"$\epsilon_{CNO}$")
        plt.scatter(logT_i, np.log10(eps_cno_i), s=0.1, color="b")
        plt.scatter(logT_o, np.log10(eps_pp_o), s=0.1, color="orange", label=r"$\epsilon_{pp}$")
        plt.scatter(logT_i, np.log10(eps_pp_i), s=0.1, color="orange")
        plt.xlabel(r"log $T$")
        plt.ylabel(r"log $\epsilon$")
        plt.ylim(-10, 5)
        plt.legend()
        ax = plt.gca()
        ax.invert_xaxis()
        plt.show()

        # log rho vs. log T
        rho_total = np.concatenate((rho_o, np.flip(rho_i)), axis=0)
        T_bf = (self.kappa_thomson / ((3e25) * (1 - self.X - self.Y)
                * (1 + self.X + (3/4)*self.Y) * rho_total))**(-2/7)
        T_ff = (self.kappa_thomson / ((4e22) * (self.X+self.Y)
                * (1+self.X) * rho_total))**(-2/7)
        plt.scatter(np.log10(rho_o), np.log10(Y_o[2]), c="black", s=0.5)
        plt.scatter(np.log10(rho_i), np.log10(Y_i[2]), c="black", s=0.5)
        plt.plot(np.log10(rho_total), np.log10(T_bf),
                 label=r"$\kappa_{bf} = \kappa_{T}$")
        plt.plot(np.log10(rho_total), np.log10(T_ff),
                 label=r"$\kappa_{ff} = \kappa_{T}$")
        plt.xlabel(r"log $\rho$")
        plt.ylabel(r"log $T$")
        plt.legend()
        plt.show()

    def plot_schwarzschild(self, logP=True):
        Y_o, Y_i = self.shooting()
        P_o, r_o, T_o, L_o = Y_o
        P_i, r_i, T_i, L_i = Y_i

        nabla_ad = (const.gamma - 1) / (const.gamma)
        dP_o = - (const.G * self.m_o) / (4 * np.pi * r_o ** 4)
        rho_o = self.rho(P_o, T_o)
        kappa_o = [self.opal.get_opacity(rho_o[i], T_o[i])
                   for i in range(len(rho_o))]
        dT_rad_o = self.radiative(r_o, T_o, L_o, kappa_o)
        criterion_o = (P_o / T_o) * (dT_rad_o / dP_o)

        dP_i = - (const.G * self.m_i) / (4 * np.pi * r_i ** 4)
        rho_i = self.rho(P_i, T_i)
        kappa_i = [self.opal.get_opacity(rho_i[i], T_i[i])
                   for i in range(len(rho_i))]
        dT_rad_i = self.radiative(r_i, T_i, L_i, kappa_i)
        criterion_i = (P_i / T_i) * (dT_rad_i / dP_i)
        criterion_vals = np.concatenate((criterion_o[1:], criterion_i[1:]))
        ymin = min(criterion_vals)
        ymax = max(criterion_vals)
        plt.plot(self.m_o, criterion_o, c="b",
                 label="outward (P/T)(dT_rad/dP)")
        plt.plot(self.m_i, criterion_i, c="orange",
                 label="inward (P/T)(dT_rad/dP)")
        plt.axhline(y=nabla_ad, c="g", label="nabla_ad")
        plt.xlabel(r"$m$")
        # plt.ylim(ymin, ymax)
        plt.title("schwarzschild criterion")
        plt.legend()
        plt.show()
