#!/usr/bin/python
# -*- coding: utf-8 -*-

# Adapted from https://github.com/sheroze1123/MatScatPy/blob/a17a2d76280c3a79f4b83183c41439094a7fbc86/clampedspline.py
import torch
torch.set_default_dtype(torch.float64)

class Clamped_Cubic_Spline:
    def __init__(self, t_data, x_data, clamped=False, dx0=None, dxn=None):
        super(Clamped_Cubic_Spline, self).__init__()
        self.t_data = t_data
        self.x_data = x_data
        self.clamped = clamped
        self.dx0 = dx0
        self.dxn = dxn
        self.coeffs = None

    def update_coeffs(self):
        self.coeffs = self.cubic_spline_get_coeffs(X=self.t_data, Y=self.x_data, clamped=self.clamped, dx0=self.dx0,
                                              dxn=self.dxn) if self.clamped else self.cubic_spline_get_coeffs(X=self.t_data,
                                                                                                         Y=self.x_data)
    def eval(self, t):
        y, yp, ypp = self.forward(t), self.derivative(t), self.derivative(t, order=2)
        return y, yp, ypp
    def forward(self, t):
        return self.cubic_spline_evaluate(x_in=t, S=self.coeffs, xx=self.t_data)
    def derivative(self, t, order=1):
        return self.cubic_spline_evaluate(x_in=t, S=self.coeffs, xx=self.t_data, order=order)
    
    def cubic_spline_get_coeffs(self,X,Y,clamped=False, dx0=0.0, dxn=0.0):
        N = len(X) - 1
        H = X[1:] - X[:-1]
        D = (Y[1:] - Y[:-1]) / H
        A = H[1:N]
        B = 2 * (H[:N-1] + H[1:])
        C = H[1:]
        U = 6 * (D[1:] - D[:-1])
        if (clamped):
            # Clamped cubic spline endpoint conditions
            B[0] = B[0]- H[0] * 0.5
            U[0] = U[0]-3.0 * (D[0] - dx0)
            B[N-2] = B[N-2] - H[N-1] * 0.5
            U[N-2] = U[N-2] - 3.0 * (dxn - D[N-1])

        # Solve the tridiagonal system of equations
        # elimination phase
        for k in range(1, N - 1):
            temp = A[k - 1] / B[k - 1]
            B[k] = B[k] - temp * C[k - 1]
            U[k] = U[k] - temp * U[k - 1]
        #
        # M = np.zeros(N+1, dtype="float")   #   numpy --> torch
        M = torch.zeros(N + 1, dtype=X.dtype, device=X.device)
        #
        # Solve and back-substitute
        M[N - 1] = U[N - 2] / B[N - 2]
        #
        k = N - 3
        while k >= 0:
            M[k + 1] = (U[k] - C[k] * M[k + 2]) / B[k]
            k -= 1
        #
        if (clamped):
            # Clamped cubic spline endpoint conditions
            M[0] = 3.0 * (D[0] - dx0) / H[0] - M[1] * 0.5
            M[N] = 3.0 * (dxn - D[N-1]) / H[N-1] - M[N-1] * 0.5
        #  else:
        #    # Free spline endpoint conditions
        #    M[0]=0
        #    M[N+1]=0
        #
        # The spline coefficients
        # S = np.zeros((N, 4), dtype="float")  #   numpy --> torch
        S = torch.zeros((N, 4), dtype=X.dtype, device=X.device)
        for k in range(N):
            S[k, 0] = (M[k + 1] - M[k])/(6 * H[k])
            S[k, 1] = M[k] * 0.5
            S[k, 2] = D[k] - H[k] * (2 * M[k] + M[k + 1]) / 6
            S[k, 3] = Y[k]
        return S
    
    def cubic_spline_evaluate (self, x_in, S, xx, order=0):
        if len(x_in.shape)==1:  # x_in has multiple time points, i.e., torch.tensor([1.0, 2.0, 3.0])
            x = x_in
            isvector = True
        else:                   # x_in: a single time point,     i.e., torch.tensor(1.0)
            x = x_in.unsqueeze(0)  # tensor(a) --> tensor([a])
            isvector = False
        m = len(x)
        n = len(xx)
        y = torch.zeros(m, dtype=xx.dtype, device=xx.device)
        yprime = torch.zeros(m, dtype=xx.dtype, device=xx.device)
        yprimeprime = torch.zeros(m, dtype=xx.dtype, device=xx.device)
        for j in range(m):
            # lookup x(j)
            # find integer k such that xx(k) <= x(j) <= xx(k+1)
            if (x[j] < xx[1]):
                # first interval
                k = 0
                u = 1
            elif (x[j] > xx[n-2]):
                # last interval
                k = n - 2
                u = n-1
            else:
                # start bisection search algorithm
                k = 1
                u = n-1
                while ((u - k) > 1):
                    mp = int((u + k) / 2)
                    if (x[j] > xx[mp]):
                        k = mp
                    else:
                        u = mp
            # evaluate the cubic in nested form
            w = x[j] - xx[k]
            y[j] = ((S[k, 0] * w + S[k, 1]) * w + S[k, 2]) * w + S[k, 3]
            yprime[j] = (3 * S[k, 0] * w + 2 * S[k, 1]) * w + S[k, 2]
            yprimeprime[j] = 6 * S[k, 0] * w + 2 * S[k, 1]
        if isvector:
            ans = (y, yprime, yprimeprime)
        else:
            ans = (y[0], yprime[0], yprimeprime[0])
        return ans[order]

class CSpline():
    def __init__(self, t_data, x_data, clamped=False, dx0=0.0, dxn=0.0):
        self.t_data = t_data
        self.x_data = x_data
        self.spline = Clamped_Cubic_Spline(t_data, x_data, clamped=clamped, dx0=dx0, dxn=dxn)
        self.spline.update_coeffs()
    
    def fit(self,t):
        y_fit, dy_fit, ddy_fit = self.spline.eval(t)
        y_fit = y_fit.reshape(-1,1)
        return(y_fit)

'''
if __name__ == '__main__':
    #True: clamped cubic spline,  f'(t_0) = dx0, f'(t_n) = dxn
    #False: natural cubic spline, f''(t_0) = 0, f''(t_n) = 0
    clamped = False

    t_data = torch.linspace(0, 5, 10)
    y_data = torch.sin(t_data)
    dx0, dxn = torch.cos(t_data[0]), torch.cos(t_data[-1])

    #spline = Clamped_Cubic_Spline(t_data, y_data, clamped=clamped, dx0=dx0, dxn=dxn)
    #spline.update_coeffs()  # get the coeffs of the cubic spline!
    cspline = CSpline(t_data, y_data, clamped=clamped, dx0=dx0, dxn=dxn)
    
    # Dense plot
    t = torch.linspace(0, 5, 1000)
    y = torch.sin(t)
    dy = torch.cos(t)
    ddy = -torch.sin(t)

    #y_fit, dy_fit, ddy_fit = spline.eval(t)
    y_fit, dy_fit, ddy_fit = cspline.fit(t)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 5), facecolor='white')
    ax_y = fig.add_subplot(131)
    ax_dy = fig.add_subplot(132)
    ax_ddy = fig.add_subplot(133)
    ax_y.plot(t_data, y_data, 'go', label='data')
    ax_y.plot(t, y, 'b-', label='True $y(t)$')
    ax_y.plot(t, y_fit, 'r--', label='Pred $y(t)$')
    ax_y.set_xlabel(r'$t$')
    ax_y.legend()

    ax_dy.plot(t, dy, 'b-', label="True $y'(t)$")
    ax_dy.plot(t, dy_fit, 'r--', label="Pred $y'(t)$")
    ax_dy.set_xlabel(r'$t$')
    ax_dy.legend()

    ax_ddy.plot(t, ddy, 'b-', label="True $y''(t)$")
    ax_ddy.plot(t, ddy_fit, 'r--', label="Pred $y''(t)$")
    ax_ddy.set_xlabel(r'$t$')
    ax_ddy.legend()

    fig.suptitle('clamped:{}'.format(clamped))
    plt.show()
    '''