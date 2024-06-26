
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):

        """ 
        defines the paramterspace for the parameters and solutions      
        """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilonM = 1.0
        par.epsilonF = 1.0
        par.omega = 0.5 
        

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ 
        LM: male labor
        HM: male home production
        LF: female labor
        HF: female home production
        
        calculates the total utility given the parmeters LM, HM, LF and HF 
        """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma==1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma==0:
            H = np.min(np.array([HM,HF]))
        else:
            HM = np.fmax(HM,1e-8)
            HF = np.fmax(HF,1e-8)
            sigma_ = (par.sigma-1)/par.sigma
            H = ((1-par.alpha)*HM**sigma_ + (par.alpha)*HF**sigma_)**((sigma_)**-1)
            
        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilonM_ = 1+1/par.epsilonM
        epsilonF_ = 1+1/par.epsilonF
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilonM_/epsilonM_+TF**epsilonF_/epsilonF_)
        
        return utility - disutility

    def solve_discrete(self):
        """ solves the model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        return opt
        

    def solve(self):
        """ solves the model continously """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. value function given the paremters (in order) LM, HM, LF and HF
        def v(x):
            value = -self.calc_utility(x[0],x[1],x[2],x[3])
            if x[0]+[1]>24:
                value = -np.inf
            elif x[2]+x[3]>24:
                value = -np.inf
            return value

        # b. optimize the valuefunction w.r.t LM, HM, LF and HF
        solution = optimize.minimize(v,[1,1,1,1],method='Nelder-Mead')

        # c. save the optimal allocation of LM, HM, LF and HF
        opt.LM = solution.x[0]
        opt.HM = solution.x[1]
        opt.LF = solution.x[2]
        opt.HF = solution.x[3]
        
        return opt

    def solve_wF_vec(self,discrete=False):
        """ 
        discrete: True or False. False is standard.

        solve model for vector of female wages """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # a. solve the model (discretly or continously) for a given female wage
        for i, wF in enumerate(par.wF_vec):
            par.wF = wF #set wF value
            
            if discrete==False:
                opt = self.solve() #Optimal allocation solution (continous)
            elif discrete==True:
                opt = self.solve_discrete() #Optimal allocation solution (discrete)
            else:
                print("discrete must be True or False")

            # saves solution for each female wage 
            sol.LM_vec[i] = opt.LM
            sol.HM_vec[i] = opt.HM
            sol.LF_vec[i] = opt.LF
            sol.HF_vec[i] = opt.HF
            
        return sol


    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        # a. define log values
        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)

        # b. run regression 
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]

        return sol
    
    def estimate(self,alpha=0.5,sigma=0.5):
        """ 
        alpha, sigma: set starting guess (standard=0.5) 

        estimate optimal values for alpha and sigma """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        # a. defines error function
        def error(x):
            alpha, sigma = x.ravel()
            par.alpha = alpha # sets alpha value
            par.sigma = sigma # sets sigma value
            
            self.solve_wF_vec() # finds optimal household production 
            sol = self.run_regression() # calculates beta0 and beta1
            error = (sol.beta0 - par.beta0_target)**2 +(sol.beta1 - par.beta1_target)**2 #calculates error
            return error
        
        # b. minimizes the error using 'Nelder-Mead' with bounds
        solution = optimize.minimize(error,[alpha,sigma],method='Nelder-Mead', bounds=[(0.0001,0.999), (0.0001,10)])
        
        # c. saves optimal value for alpha and beta
        opt.alpha = solution.x[0]
        opt.sigma = solution.x[1]
        error = (sol.beta0 - par.beta0_target)**2 +(sol.beta1 - par.beta1_target)**2 #calculates error
        opt.error = error
        
        return opt
        
    def est_alphacons(self,sigma=0.5,epsilonM=1,epsilonF=1,extended=True):
        """ 
        sigma, epsilonM, epsilonF: set starting guess (default=1) 
        extended: True if epsilonM and epsilonF should be optimized. False otherwise
        
        estimate optimal values for coefficients given alpha=0.5"""

        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        par.alpha = 0.5

        if extended==True: 
            # a.i defines error function
            def error(x):
                sigma, epsilonM, epsilonF = x.ravel()
                par.sigma = sigma # sets sigma value
                par.epsilonM = epsilonM
                par.epsilonF = epsilonF
                
                self.solve_wF_vec() # finds optimal household production 
                sol = self.run_regression() # calculates beta0 and beta1
                error = (sol.beta0 - par.beta0_target)**2 +(sol.beta1 - par.beta1_target)**2 #calculates error
                return error
            
            # b.i minimizes the error using 'Nelder-Mead' with bounds
            solution = optimize.minimize(error,[sigma, epsilonM, epsilonF],method='Nelder-Mead', bounds=[(0,2),(0.5,2),(0.5,2)])
            
            # c.ii saves optimal coefficients
            opt.sigma = solution.x[0]
            opt.epsilonM = solution.x[1]
            opt.epsilonF = solution.x[2]
        
        elif extended==False:
            # a.ii defines error function
            def error(x):
                par.sigma = x # sets sigma value
                
                self.solve_wF_vec() # finds optimal household production 
                sol = self.run_regression() # calculates beta0 and beta1
                error = (sol.beta0 - par.beta0_target)**2 +(sol.beta1 - par.beta1_target)**2 #calculates error
                return error

            # b.ii minimizes the error using 'Nelder-Mead' with bounds            
            solution = optimize.minimize(error,[sigma],method='Nelder-Mead', bounds=[(0,2)])

            # c.ii saves optimal coefficients
            opt.sigma = solution.x

        else:
            print('extended must be either True or False')

        # d. saves error value  
        error = (sol.beta0 - par.beta0_target)**2 +(sol.beta1 - par.beta1_target)**2 #calculates error
        opt.error = error

        return opt