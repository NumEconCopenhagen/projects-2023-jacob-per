from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize

class OLGModelClass():

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

        # a. household
        par.sigma = 2.0 # CRRA coefficient
        par.beta = 1/1.40 # discount factor
        par.E_ini = 1.0 # initial employment

        # b. firms
        par.production_function = 'cobb-douglas'
        par.alpha = 0.30 # capital weight
        par.theta = 0.0 # substitution parameter
        par.delta = 1.0 # depreciation rate

        # c. government
        par.tau = 0.25 # labor income tax

        # d. misc
        par.K_lag_ini = 1.0 # initial capital stock
        par.simT = 50 # length of simulation
        par.w_lag_ini = 1.0 # initial wage
    
    def allocate(self):
        """ allocate arrays for simulation """
        
        par = self.par
        sim = self.sim

        # a. list of variables
        household = ['C1','C2']
        firm = ['K','Y','K_lag','E']
        prices = ['w','r']

        # b. allocate
        allvarnames = household + firm + prices
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan*np.ones(par.simT)

    def simulate(self,do_print=True, shock=False, PAYG=False):
        """ simulate model """

        t0 = time.time()

        par = self.par
        sim = self.sim
        
        # a. initial values
        sim.K_lag[0] = par.K_lag_ini
        #sim.w_lag[0] = par.w_lag_ini

        # b. iterate
        for t in range(par.simT):
            
            # i. simulate before s
            simulate_before_s(par,sim,t,shock,PAYG)

            if t == par.simT-1: continue          

            # i. find bracket to search
            s_min,s_max = find_s_bracket(par,sim,t)

            # ii. find optimal s
            obj = lambda s: calc_euler_error(s,par,sim,t=t)
            result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect')
            s = result.root

            # iii. simulate after s
            simulate_after_s(par,sim,t,s)

            # iv. utility
            utility = (sim.C1[t]**(1-par.sigma))/(1-par.sigma) + par.beta*((sim.C2[t+1]**(1-par.sigma))/(1-par.sigma))

        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs')

def find_s_bracket(par,sim,t,maxiter=500,do_print=False):
    """ find bracket for s to search in """

    # a. maximum bracket
    s_min = 0.0 + 1e-8 # save almost nothing
    s_max = 1.0 - 1e-8 # save almost everything

    # b. saving a lot is always possible 
    value = calc_euler_error(s_max,par,sim,t)
    sign_max = np.sign(value)
    if do_print: print(f'euler-error for s = {s_max:12.8f} = {value:12.8f}')

    # c. find bracket      
    lower = s_min
    upper = s_max

    it = 0
    while it < maxiter:
                
        # i. midpoint and value
        s = (lower+upper)/2 # midpoint
        value = calc_euler_error(s,par,sim,t)

        if do_print: print(f'euler-error for s = {s:12.8f} = {value:12.8f}')

        # ii. check conditions
        valid = not np.isnan(value)
        correct_sign = np.sign(value)*sign_max < 0
        
        # iii. next step
        if valid and correct_sign: # found!
            s_min = s
            s_max = upper
            if do_print: 
                print(f'bracket to search in with opposite signed errors:')
                print(f'[{s_min:12.8f}-{s_max:12.8f}]')
            return s_min,s_max
        elif not valid: # too low s -> increase lower bound
            lower = s
        else: # too high s -> increase upper bound
            upper = s

        # iv. increment
        it += 1

    #raise Exception('cannot find bracket for s')

def calc_euler_error(s,par,sim,t):
    """ target function for finding s with bisection """

    # a. simulate forward
    simulate_after_s(par,sim,t,s)
    simulate_before_s(par,sim,t+1) # next period

    # c. Euler equation
    LHS = sim.C1[t]**(-par.sigma)
    RHS = par.beta*(1+sim.r[t+1])*sim.C2[t+1]**(-par.sigma)

    return LHS-RHS

def simulate_before_s(par, sim, t, shock=False, PAYG=False):
    """ simulate forward """

    # a. shock to employment
    sim.E[t]=(par.E_ini+sim.E[t-1])*0.5
    sim.E[0]=par.E_ini
    if shock==True:
        sim.E[1]=par.E_ini*1.5
        

    if t > 0:
        sim.K_lag[t] = sim.K[t-1]
    
    # b. production and factor prices 
    if par.production_function == 'ces': # (kept from lectures, but not used)

        # i. production
        sim.Y[t] = ( par.alpha*sim.K_lag[t]**(-par.theta) + (1-par.alpha)*(1.0)**(-par.theta) )**(-1.0/par.theta)

        # ii. factor prices
        sim.r[t] = par.alpha*sim.K_lag[t]**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)
        sim.w[t] = (1-par.alpha)*(1.0)**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)

    elif par.production_function == 'cobb-douglas':

        # i. production
        sim.Y[t] = sim.K_lag[t]**par.alpha * sim.E[t]**(1-par.alpha)

        # ii. factor prices
        sim.r[t] = par.alpha * sim.K_lag[t]**(par.alpha-1) * sim.E[t]**(1-par.alpha)
        sim.w[t] = (1-par.alpha) * sim.K_lag[t]**(par.alpha) * sim.E[t]**(-par.alpha)

    else:

        raise NotImplementedError('unknown type of production function')

    # if t > 0:
    #     sim.w_lag[t] = sim.w[t-1]

    # d. consumption
    # if PAYG==False: #FF
    #     sim.w[-1]=par.w_lag_ini
    #     sim.C2[t] = (1+sim.r[t])*(sim.K_lag[t]+par.tau*sim.w[t-1])
    if PAYG==True: #PAYG
        sim.C2[t] = (1+sim.r[t])*sim.K_lag[t]+sim.E[t]*par.tau*sim.w[t]
 
def simulate_after_s(par,sim,t,s):
    """ simulate forward """

    # a. consumption of young
    sim.C1[t] = (1-par.tau)*sim.w[t]*(1.0-s)

    # b. end-of-period stocks
    I = sim.Y[t] - sim.C1[t] - sim.C2[t]
    sim.K[t] = (1-par.delta)*sim.K_lag[t] + I