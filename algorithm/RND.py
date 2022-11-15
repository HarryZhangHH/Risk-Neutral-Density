import numpy as np
from scipy.stats import norm
from scipy import special
from sklearn.metrics import mean_squared_error

class RiskNeutralDensityFunctions:
    def __init__(self, config=None, real_data=False):
        if config is None:
            # default config
            config = {
                'polynomial': 'hermite',
                'rule': 'trapezoidal',
                'transformation': None,
                'k': 200,
                'T': 30/252,
                'rate': 0,
                'y': np.arange(1,9.05,0.05),
                'K': np.arange(1,9.05,0.05),
                'F0': 5,
                'S0': 5,
                'sigma': 0.1,
                'a': None,
                'b': None
            }
        self.parse_config(**config)
        self.N = norm.cdf
        if not real_data:
            self.baseline()
            self.calculate_BS()
    
    def parse_config(self, polynomial, rule, transformation, k, T, rate, K, y, F0, S0, sigma, a, b):
        self.polynomial = polynomial
        self.rule = rule
        self.transformation = transformation
        self.k = k
        self.rate = rate
        self.T = T
        self.y = y
        self.K = K
        self.F0 = F0
        self.S0 = S0
        self.sigma = sigma
        self.delta = T
        if transformation == 'normal':
            self.a = a
            self.b = b
    
    def baseline(self):
        # only for simulation, real world data don't have baseline
        s = np.sqrt(self.sigma**2*self.T)
        mu = np.log(self.S0)+(self.rate-1/2*(self.sigma**2))*self.T
        if self.transformation == 'log':
            self.pdf = 1 / (s*np.sqrt(2*np.pi)) * np.exp(-1/2*((np.log(self.y)-mu)/s)**2)
        else:
            self.pdf = 1 / (s*self.y*np.sqrt(2*np.pi)) * np.exp(-1/2*((np.log(self.y)-mu)/s)**2)

    def transformed(self, x):
        return (x-self.a)/self.b
    
    def transform(self, f):
        return (1/np.sqrt(self.b)) * f
        
    def hermite_k(self, x, h_k_coef, hermite):
        return h_k_coef*np.exp(-(x**2)/2)*hermite(x)

    # apply hermite polynomial
    def hermite_element(self):
        H_kF0, H_ky, H_kK, H_k2K = [], [], [], []      
        for k in range(0, self.k+1):
            h_k_coef = 1/(((np.pi)**(1/4))*(2**(k/2))*np.sqrt(special.factorial(k)))
            hermite = special.hermite(k)
            if self.transformation == 'normal':
                # normal transformation
                H_kF0.append(self.transform(self.hermite_k(self.transformed(self.F0), h_k_coef, hermite)))
                H_ky.append(self.transform(self.hermite_k(self.transformed(self.y), h_k_coef, hermite)))
                H_k2K.append(-1/(self.b**2) * (2*k+1-self.transformed(self.K)**2) * self.transform(self.hermite_k(self.transformed(self.K), h_k_coef, hermite)))
            elif self.transformation == 'log':
                # log transformation
                H_kF0.append(self.hermite_k(np.log(self.F0), h_k_coef, hermite))
                H_ky.append(self.hermite_k(np.log(self.y), h_k_coef, hermite))
                H_kK.append(self.hermite_k(np.log(self.K), h_k_coef, hermite))
                H_k2K.append(-(2*k+1-(np.log(self.K))**2) * self.hermite_k(np.log(self.K), h_k_coef, hermite))
            else:
                # no transformation 
                H_kF0.append(self.hermite_k(self.F0, h_k_coef, hermite))
                H_ky.append(self.hermite_k(self.y, h_k_coef, hermite))
                H_k2K.append(-(2*k+1-(self.K**2))*self.hermite_k(self.K, h_k_coef, hermite))
        # only for log transformation
        if self.transformation == 'log':
            H_kK = np.array(H_kK)
            H_k1K, A_K = np.zeros((H_kK.shape[0],H_kK.shape[1])), np.zeros((H_kK.shape[0],H_kK.shape[1]))
            # insert 0 to the first row because of k-1
            H_kK = np.insert(H_kK, len(H_kK), values=0, axis=0)
            for k in range(0, self.k):
                H_k1K[k] = np.sqrt(k/2)*H_kK[k-1]-np.sqrt((k+1)/2)*H_kK[k+1]
            H_k1K, H_k2K = np.array(H_k1K), np.array(H_k2K)
            for i in range(len(self.K)):
                A_K[:,i] = (-H_k1K[:,i]+H_k2K[:,i])/(self.K[i]**2)        
            return np.array(H_kF0), np.array(H_ky), A_K
        else:
            # print(f'kF0: {H_kF0},\nky: {H_ky},\nk2K: {H_k2K}')
            return np.array(H_kF0), np.array(H_ky), np.array(H_k2K)

    def laguerre_k(self, x, laguerre):
        return np.exp(-x/2)*laguerre(x)
    
    # the second derivative of laguerre
    def laguerre_k_second(self, k, x, Laguerre_kK):
        return -k/(x**2)*(Laguerre_kK[k]-Laguerre_kK[k-1]) + (1/4-k/x)*Laguerre_kK[k]

    # apply laguerre polynomial
    def laguerre_element(self):
        L_kF0, L_ky, Laguerre_kK = [], [], []  # for all three transformation
        for k in range(0, self.k+1):
            laguerre = special.genlaguerre(k,0)
            if self.transformation == 'normal':
                # normal transformation
                L_kF0.append(self.transform(self.laguerre_k(self.transformed(self.F0), laguerre)))
                L_ky.append(self.transform(self.laguerre_k(self.transformed(self.y), laguerre)))
                Laguerre_kK.append(self.laguerre_k(self.transformed(self.K), laguerre))
            elif self.transformation == 'log':
                # log transformation
                L_kF0.append(self.laguerre_k(np.log(self.F0), laguerre))
                L_ky.append(self.laguerre_k(np.log(self.y), laguerre))
                Laguerre_kK.append(self.laguerre_k(np.log(self.K), laguerre))
            else:
                # no transformation 
                L_kF0.append(self.laguerre_k(self.F0, laguerre))
                L_ky.append(self.laguerre_k(self.y, laguerre))
                Laguerre_kK.append(self.laguerre_k(self.K, laguerre)) 
        Laguerre_kK = np.array(Laguerre_kK)
        L_k1K, L_k2K = np.zeros((Laguerre_kK.shape[0],Laguerre_kK.shape[1])), np.zeros((Laguerre_kK.shape[0],Laguerre_kK.shape[1]))
        # insert 0 to the last row because of k-1
        Laguerre_kK = np.insert(Laguerre_kK, len(Laguerre_kK), values=0, axis=0)
        # only for log transformation
        if self.transformation == 'log':
            for k in range(0, self.k+1):
                L_k1K[k] = k/np.log(self.K) * (Laguerre_kK[k]-Laguerre_kK[k-1]) - Laguerre_kK[k]/2
                L_k2K[k] = self.laguerre_k_second(k, np.log(self.K), Laguerre_kK)
            A_K = np.zeros((L_k1K.shape[0],L_k1K.shape[1]))
            for i in range(len(self.K)):
                A_K[:,i] = (-L_k1K[:,i]+L_k2K[:,i])/(self.K[i]**2)
            return np.array(L_kF0), np.array(L_ky), A_K
        else:
            for k in range(0,self.k+1):
                if self.transformation == 'normal':
                    # normal transformation
                    L_k2K[k] = self.transform(self.laguerre_k_second(k, self.transformed(self.K), Laguerre_kK))/(self.b**2)
                else:
                    # no transformation 
                    L_k2K[k] = self.laguerre_k_second(k, self.K, Laguerre_kK)
            return np.array(L_kF0), np.array(L_ky), L_k2K

    # the first derivative of legendre
    def legendre_k_first (self, k, x, P_kK):
        return k/(x**2-1) * (x*P_kK[k] - np.sqrt((k+0.5)/(k-0.5))*P_kK[k-1])

    # the second derivative of legendre
    def legendre_k_second(self, k, x, P_kK):
        return (np.sqrt((k+0.5)/(k-0.5))*2*k*x*P_kK[k-1] + k*((k-1)*x**2-k-1)*P_kK[k]) / (x**2-1)**2

    # apply legendre polynomial
    def legendre_element(self):
        P_kF0, P_ky, P_kK = [], [], []
        for k in range(0, self.k+1):
            p_k_coef = np.sqrt(k+0.5)
            legendre = special.legendre(k)
            p_ky, p_kK = [], []
            if self.transformation == 'normal':
                # normal transformation
                P_kF0.append(self.transform(p_k_coef*legendre(self.transformed(self.F0))))
                for j in range(len(self.y)):
                    p_ky.append(self.transform(p_k_coef*legendre(self.transformed(self.y[j]))))
                for i in range(len(self.K)):
                    p_kK.append(self.transform(p_k_coef*legendre(self.transformed(self.K[i]))))
            elif self.transformation == 'log':
                # log transformation
                P_kF0.append(p_k_coef*legendre(np.log(self.F0)))
                for j in range(len(self.y)):
                    p_ky.append(p_k_coef*legendre(np.log(self.y[j])))
                for i in range(len(self.K)):
                    p_kK.append(p_k_coef*legendre(np.log(self.K[i])))
            else:
                # no transformation 
                print(k, legendre(self.F0))
                P_kF0.append(p_k_coef*legendre(self.F0))
                for j in range(len(self.y)):
                    p_ky.append(p_k_coef*legendre(self.y[j]))
                for i in range(len(self.K)):
                    p_kK.append(p_k_coef*legendre(self.K[i]))
            # for all three transformation
            P_ky.append(p_ky)
            P_kK.append(p_kK)
        P_kK = np.array(P_kK)
        P_k1K, P_k2K = np.zeros((P_kK.shape[0],P_kK.shape[1])), np.zeros((P_kK.shape[0],P_kK.shape[1]))
        # only for log transformation
        if self.transformation == 'log':
            P_k1K, P_k2K = np.zeros((P_kK.shape[0],P_kK.shape[1])), np.zeros((P_kK.shape[0],P_kK.shape[1]))
            # insert 0 to the first row because of k-1
            P_kK = np.insert(P_kK, 0, values=0, axis=0)
            for k in range(1, self.k+2):
                P_k1K[k-1] = self.legendre_k_first (k, np.log(self.K), P_kK)
                P_k2K[k-1] = self.legendre_k_second(k, np.log(self.K), P_kK)
            A_K = np.zeros((P_k1K.shape[0],P_k1K.shape[1]))
            for i in range(len(self.K)):
                A_K[:,i] = (-P_k1K[:,i]+P_k2K[:,i])/(self.K[i]**2)
            return np.array(P_kF0), np.array(P_ky), A_K
        else:
            P_kK = np.insert(P_kK, 0, values=0, axis=0)
            for k in range(1,self.k+2):
                if self.transformation == 'normal':
                    # normal transformation
                    P_k2K[k-1] = self.transform(self.legendre_k_second(k, self.transformed(self.K), P_kK))/(self.b**2)
                else:
                    # no transformation 
                    P_k2K[k-1] = self.legendre_k_second(k, self.K, P_kK)
            return np.array(P_kF0), np.array(P_ky), P_k2K

    def BS_CALL(self, K):
        d1 = (np.log(self.S0/K) + (self.rate + self.sigma**2/2)*self.T) / (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return self.S0 * self.N(d1) - K * np.exp(-self.rate*self.T)* self.N(d2)

    def BS_PUT(self, K):
        d1 = (np.log(self.S0/K) + (self.rate + self.sigma**2/2)*self.T) / (self.sigma*np.sqrt(self.T))
        d2 = d1 - self.sigma* np.sqrt(self.T)
        return K*np.exp(-self.rate*self.T)*self.N(-d2) - self.S0*self.N(-d1)

    def calculate_BS(self):
        # only for simuation, in real world data the BS is provided or obtained using different method
        BS = []
        for i in range(len(self.K)):
            if self.K[i] <= self.S0:
                BS.append(self.BS_PUT(self.K[i]))
            else:
                BS.append(self.BS_CALL(self.K[i]))
        self.BS = np.array(BS)

    def calculate_I_hat(self, k2K):
        I_hat = []
        for k_idx in range(0, self.k+1):
            I = []
            if self.rule == 'trapezoidal':
                for i in range(len(self.K)-1):
                    I.append((self.K[i+1]-self.K[i])*(k2K[k_idx,i]*self.BS[i] + k2K[k_idx,i+1]*self.BS[i+1])/2)
            elif self.rule == 'simpson':
                for i in range(1, int(len(self.K)/2)):
                    I.append((self.K[2*i]-self.K[2*i-2])*(k2K[k_idx,2*i-2]*self.BS[2*i-2] + 4*k2K[k_idx,2*i-1]*self.BS[2*i-1] + k2K[k_idx,2*i]*self.BS[2*i])/6)
            elif self.rule == 'irresimpson':
                for i in range(1, int(len(self.K)/2)):
                    I.append((self.K[2*i]-self.K[2*i-2])*( (2-(self.K[2*i]-self.K[2*i-1])/(self.K[2*i-1]-self.K[2*i-2])) * k2K[k_idx,2*i-2]*self.BS[2*i-2]
                            + (self.K[2*i]-self.K[2*i-2])**2/((self.K[2*i]-self.K[2*i-1])*(self.K[2*i-1]-self.K[2*i-2])) * k2K[k_idx,2*i-1]*self.BS[2*i-1] 
                            + (2-(self.K[2*i-1]-self.K[2*i-2])/(self.K[2*i]-self.K[2*i-1])) * k2K[k_idx,2*i]*self.BS[2*i]) / 6)
            else:
                print('--- Wrong Numerical Method!!! ---\n--- Select rule from (trapezoidal, simpson, irresimpson) ---\
                    /n--- RESTART!!! ---')
                exit()
            I = np.array(I)
            I_hat.append(np.exp(self.rate*self.delta)*np.sum(I[~np.isnan(I)]))
        return I_hat
    
    def RND_f_slice(self, F_kF0, F_ky, I_hat, k):
        # find the RND of specific k
        A = F_kF0[:,:k+1]@F_ky[:k+1,:]
        B = I_hat[:,:k+1]@F_ky[:k+1,:]
        density = A+B
        return density

    def RND_f(self, idx: list = None):
        # example code of RND procedure
        if self.polynomial == 'hermite':
            F_kF0, F_ky, F_k2K = self.hermite_element()
        elif self.polynomial == 'laguerre':
            F_kF0, F_ky, F_k2K = self.laguerre_element()
        elif self.polynomial == 'legendre':
            F_kF0, F_ky, F_k2K = self.legendre_element()
        # print(f'kF0: {F_kF0},\nky: {F_ky},\nk2K: {F_k2K}')
        
        F_kF0 = F_kF0.reshape((1,-1))
        print(f'Using {self.polynomial}, we can get --> kF0 shape: {F_kF0.shape}, ky shape: {F_ky.shape}, k2K shape: {F_k2K.shape}')
        I_hat = np.array(self.calculate_I_hat(F_k2K)).reshape((1,-1))
        density = []
        if idx is None:
            density.append(self.RND_f_slice(F_kF0, F_ky, I_hat, self.k))
        else:
            for k in idx:
                density.append(self.RND_f_slice(F_kF0, F_ky, I_hat, k))
        F_k_mean = F_kF0 + I_hat
        return density, F_k_mean.reshape((-1))
    
    def find_best_error(self, density, idx):
        # find the best k which has the min MSE
        index, best_idx, min_error = 0, 0, np.inf
        for i in range(len(idx)):
            mse = mean_squared_error(self.pdf, density[i])
            error = self.pdf-density[i]
            print(f'N={idx[i]}      Max error is {np.max(error)}      MSE is {mse}')
            if mse < min_error:
                min_error = mse
                best_idx = idx[i]
                index = i
        return best_idx, min_error, index


