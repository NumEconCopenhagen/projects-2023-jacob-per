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
        self.sol = SimpleNamespace()

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
        par.kappa = 1.5 # chosen
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

        # a. misc
        sim.delta = delta
        sim.K = K
        H_con = np.zeros((1,K))
        
        # b. simulations K times        
        for k in range(K):
            
            # i. simulating model
            self.iterate(delta)

            # ii. h (aggregate)
            sim.h = np.sum(sim.h_con)

            # iii. H contribution
            H_con[0,k] = sim.h
        
        # c. H (aggregate)
        sim.H = np.average(H_con)

        return sim

    def iterate(self,delta):

        par = self.par
        sim = self.sim

        # a. initial values
        sim.log_kappa_lag[0] = np.log(par.kappa_ini)
        sim.l[0] = par.l_ini

        # b. iterating
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
                if np.absolute(sim.l[t-1]-sim.l[t]) <= delta:
                    sim.l[t] = sim.l[t-1]
            
            # iii. h contribution
            if sim.l[t] == sim.l[t-1]:
                sim.h_con[t] = par.R**(-t)*(sim.kappa[t]*sim.l[t]**(1-par.eta) - par.w*sim.l[t])
            else:
                sim.h_con[t] = par.R**(-t)*(sim.kappa[t]*sim.l[t]**(1-par.eta) - par.w*sim.l[t] - par.iota)

    def optimizer(self,value_function,n_guess=1,seeds=1,K=100, do_print=False):
               
        sol = self.sol
        
        # a. random guesses
        guess_draw = np.random.uniform(0.01,0.19,n_guess)
        guess = np.sort(guess_draw)

        # b. setup
        best_H = 0
        best = np.zeros((2,n_guess))

        # optimizer guesses
        for n in range(n_guess):

            # i. optimal delta for a given guess (note: change to SLSQP to increase speed)
            now_delta = optimize.minimize(value_function,guess[n], method='SLSQP',bounds=[(0,0.25)]).x[0]
                
            # ii. calculates H
            now_H = self.simulate(now_delta,K).H

            # iii. store results 
            best[0,n] = now_delta
            best[1,n] = now_H

            # iv. change best if better
            if now_H > best_H:
                best_delta = now_delta
                best_H = now_H
            
            if do_print==True:
                print(f'initial guess={guess[n]:.3f}: optimal guess to H={best_H:.3f} and delta={best_delta:.3f}')
        
        return best, best_delta, best_H
            


