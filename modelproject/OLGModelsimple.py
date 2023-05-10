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

        # b. firms
        par.production_function = 'cobb-douglas'
        par.alpha = 0.30 # capital weight
        par.theta = 0.0 # substitution parameter
        par.delta = 1 # depreciation rate

        # c. government
        par.tau_w = 0.0 # labor income tax
        par.tau_r = 0.0 # capital income tax

        # d. misc
        par.K_lag_ini = 1 # initial capital stock
        par.w_ini = 1 # initial government debt
        par.simT = 50 # length of simulation
        par.E_ini = 1.0
    
    def allocate(self):
        """ allocate arrays for simulation """
        
        par = self.par
        sim = self.sim

        # a. list of variables
        household = ['C1','C2']
        firm = ['K','Y','K_lag','E']
        prices = ['w','rk','rb','r','rt']
        government = ['T']

        # b. allocate
        allvarnames = household + firm + prices + government
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan*np.ones(par.simT)

    def simulate(self,do_print=True,shock=False,regime='PAYG'):
        """ simulate model """

        t0 = time.time()

        par = self.par
        sim = self.sim
        
        # a. initial values
        sim.K_lag[0] = par.K_lag_ini        
   
        # b. iterate
        for t in range(par.simT):
            
            # i. simulate before s
            simulate_before_s(par,sim,t,shock,regime)

            if t == par.simT-1: continue          

            # i. find bracket to search
            s_min,s_max = find_s_bracket(par,sim,t)

            # ii. find optimal s
            obj = lambda s: calc_euler_error(s,par,sim,t=t)
            result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect')
            s = result.root

            # iii. simulate after s
            simulate_after_s(par,sim,t,s)

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

    raise Exception('cannot find bracket for s')

def calc_euler_error(s,par,sim,t):
    """ target function for finding s with bisection """

    # a. simulate forward
    simulate_after_s(par,sim,t,s)
    simulate_before_s(par,sim,t+1) # next period

    # c. Euler equation
    LHS = sim.C1[t]**(-par.sigma)
    RHS = (1+sim.rt[t+1])*par.beta * sim.C2[t+1]**(-par.sigma)

    return LHS-RHS

def simulate_before_s(par,sim,t,shock=False,regime='PAYG'):
    """ simulate forward """

    sim.E[t]=par.E_ini
    if shock==True:
        sim.E[1]=par.E_ini*1.5
    
    if t > 0:
        sim.K_lag[t] = sim.K[t-1]

    # a. production and factor prices 
    if par.production_function == 'ces': # (kept from lectures, but not used)

        # i. production
        sim.Y[t] = ( par.alpha*sim.K_lag[t]**(-par.theta) + (1-par.alpha)*(1.0)**(-par.theta) )**(-1.0/par.theta)

        # ii. factor prices
        sim.rk[t] = par.alpha*sim.K_lag[t]**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)
        sim.w[t] = (1-par.alpha)*(1.0)**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)

    elif par.production_function == 'cobb-douglas':

        # i. production
        sim.Y[t] = sim.K_lag[t]**par.alpha * sim.E[t]**(1-par.alpha)

        # ii. factor prices
        sim.rk[t] = par.alpha * sim.K_lag[t]**(par.alpha-1) * sim.E[t]**(1-par.alpha)
        sim.w[t] = (1-par.alpha) * sim.K_lag[t]**(par.alpha) * sim.E[t]**(-par.alpha)

    else:

        raise NotImplementedError('unknown type of production function')

    # b. no-arbitrage and after-tax return
    sim.r[t] = sim.rk[t]-par.delta # after-depreciation return
    sim.rt[t] = (1-par.tau_r)*sim.r[t] # after-tax return

    # c. consumption
    # sim.C2[t] = (1+sim.rt[t])*(sim.K_lag[t])+par.tau_w*sim.w[t]*sim.E[t]

    if regime=='FF': #FF
        if t>0:
            sim.C2[t] = (1+sim.rt[t])*(sim.K_lag[t]+par.tau_w*sim.w[t-1])
        elif t==0:
            sim.C2[t] = (1+sim.rt[t])*sim.K_lag[t]+sim.E[t]*par.tau_w*sim.w[t]

    if regime=='PAYG': #PAYG
        sim.C2[t] = (1+sim.rt[t])*sim.K_lag[t]+sim.E[t]*par.tau_w*sim.w[t]

    # d. government
    sim.T[t] = par.tau_r*sim.r[t]*(sim.K_lag[t]) + par.tau_w*sim.w[t]
   
def simulate_after_s(par,sim,t,s):
    """ simulate forward """

    # a. consumption of young
    sim.C1[t] = (1-par.tau_w)*sim.w[t]*(1.0-s)

    # b. end-of-period stocks
    I = sim.Y[t] - sim.C1[t] - sim.C2[t] - sim.T[t]
    sim.K[t] = (1-par.delta)*sim.K_lag[t] + I