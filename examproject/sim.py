from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize

class simClass():

    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
    
    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. parameters
        par.eta = 0.5
        par.w = 1.0
        par.kappa = 1.5 # chosen at random
        par.rho = 0.9
        par.iota = 0.01
        par.sigma_eps = 0.1
        par.R = (1+0.01)**(1/12)
        par.delta = 0.0

        # b. misc
        par.simT = 119
        par.l_ini = 0
        par.kappa_ini = 1

    
    def allocate(self):
        """ allocate arrays for simulation """
        
        par = self.par
        sim = self.sim

        # a. list of variables
        variables = ['t','kappa','log_kappa','log_kappa_lag','l','epsilon','h_con']

        # b. allocate
        for varname in variables:
            sim.__dict__[varname] = np.nan*np.ones(par.simT)


    def simulate(self,delta=0.0,K=1):
        """ simulate model """

        t0 = time.time()

        par = self.par
        sim = self.sim

        sim.delta = delta
        sim.K = K

        # a. initial values
        sim.log_kappa_lag[0] = np.log(par.kappa_ini)
        sim.l[0] = par.l_ini
        H_con = np.zeros((1,K))
        
        # b. iterate with K simulations        
        for k in range(K):
            
            for t in range(par.simT):
                
                if t>0:
                    sim.log_kappa_lag[t] = sim.log_kappa[t-1]

                # i. demand shock
                sim.epsilon[t] = np.random.normal(-0.5*par.sigma_eps**2,par.sigma_eps)
                sim.log_kappa[t] = par.rho*sim.log_kappa_lag[t] + sim.epsilon[t]
                sim.kappa[t] = np.exp(sim.log_kappa[t])
                
                # ii. optimal labor
                sim.l[t] = (((1-par.eta)*sim.kappa[t])/par.w)**(1/par.eta)
                
                if delta!=0:
                    # model with delta
                    if np.abs(sim.l[t-1]-sim.l[t]) > delta:
                        sim.l[t]
                    else:
                        sim.l[t-1]

                # iii. h contribution
                if sim.l[t] != sim.l[t-1]:
                    sim.h_con[t] = par.R**(-t)*(sim.kappa[t]*sim.l[t]**(1-par.eta) - par.w*sim.l[t] - par.iota)
                else:
                    sim.h_con[t] = par.R**(-t)*(sim.kappa[t]*sim.l[t]**(1-par.eta) - par.w*sim.l[t])
            
            # c. h (aggregate)
            sim.h = np.sum(sim.h_con)

            # d. H contribution
            H_con[0,k] = sim.h

        # d. H (aggregate)
        sim.H = np.average(H_con)

        return sim


