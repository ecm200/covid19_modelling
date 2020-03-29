
import numpy as np, pandas as pd, scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from time import strptime

def compute_SEIR_model(N = 1000, T_inc = 10, T_rec = 14, T_die = 10, R0 = 5,
                       tau = 0.2, p_live = 0.95, p_die = 0.05, ndays = 100, 
                       interventions = False, 
                       categories = ['Susceptible','Exposed','Infected','Recovered','Deceased'],
                       figsize=(10,4)):

    # ODE parameters
    alpha1 = tau*p_live/N # average % of susceptible people who get infected by survivor
    alpha2 = tau*p_die/N # average % of susceptible people who get infected by non-survivor
    beta = 1/T_inc # transition rate of incubation to infection
    gamma = p_live/T_rec # transition rate of infection to recovery
    psi = p_die/T_die # transition rate of infection to mortality

    y0 = [N-1, 1, 0, 0, 0] # initial conditions
    t_span = [0, ndays] # dayspan to evaluate
    t_eval = np.arange(ndays) # days to evaluate

    # Here we look at the evolution given no intervention

    print('[INFO] R0::', R0)
    print('[INFO] Alpha1::', alpha1)
    print('[INFO] Alpha2::', alpha2)
    print('[INFO] Beta::', beta)
    print('[INFO] Gamma::', gamma)
    print('[INFO] Psi::',psi)

    if interventions == False:

        solution = solve_ivp(fun = ODE_model, t_span = t_span, t_eval = t_eval, y0 = y0, args = (R0, alpha1, alpha2, beta, gamma, psi))

    else:

        solution = solve_ivp(fun = ODE_model_with_intervention, t_span = t_span, t_eval = t_eval, y0 = y0, 
                             args = (R0, reproduction, alpha1, alpha2, beta, gamma, psi, interventions))

    Y = np.maximum(solution.y,0)

    multi_plot(Y, labels=categories, interventions=interventions, figsize=figsize)

    CC0 = Y[2,0]+Y[3,0]+Y[4,0]
    CCases = np.diff(Y[2]+Y[3]+Y[4], prepend=CC0).cumsum()

    print('')
    print('')
    print('************************************************')
    print('***          SEIR MODEL RUN RESULTS          ***')
    print('************************************************')
    print('')
    print('Total number of fatalities:: {0:8.0f}'.format(np.max(Y[4])))
    print('Total number of confirmed cases:: {0:8.0f} '.format(np.max(CCases)))
    print('')
    print('Finished.')


def ODE_model(t, y, Rt, alpha1, alpha2, beta, gamma, psi):

    if callable(Rt):
        reproduction_t = Rt(t)
    else:
        reproduction_t = Rt
    
    S, E, I, R, D = y
    St = dS_dt(S, I, reproduction_t, alpha1, alpha2)
    Et = dE_dt(S, I, E, reproduction_t, alpha1, alpha2, beta)
    It = dI_dt(E, I, beta, gamma, psi)
    Rt = dR_dt(I, gamma)
    Dt = dD_dt(I, psi)
    return [St, Et, It, Rt, Dt]

def ODE_model_with_intervention(t, y, R0, Rt, alpha1, alpha2, beta, gamma, psi, interventions):

    if callable(Rt):
        reproduction_t = Rt(t, R0, interventions)
    else:
        reproduction_t = Rt
    
    S, E, I, R, D = y
    St = dS_dt(S, I, reproduction_t, alpha1, alpha2)
    Et = dE_dt(S, I, E, reproduction_t, alpha1, alpha2, beta)
    It = dI_dt(E, I, beta, gamma, psi)
    Rt = dR_dt(I, gamma)
    Dt = dD_dt(I, psi)

    return [St, Et, It, Rt, Dt]


def reproduction(t, R0, interventions):
    intervention_days = [interventions[kk]['day'] for kk in list(interventions.keys())]
    reproduction_rates = [interventions[kk]['reproduction_rate'] for kk in list(interventions.keys())]
    ix=np.where(np.array(intervention_days)<t)[0]
    
    if len(ix)==0:
        return R0
    else:
        return reproduction_rates[ix.max()]


def dS_dt(S, I, reproduction_t, alpha1, alpha2):
    return -alpha1*reproduction_t*S*I -alpha2*reproduction_t*S*I

def dE_dt(S, I, E, reproduction_t, alpha1, alpha2, beta):
    return alpha1*reproduction_t*S*I + alpha2*reproduction_t*S*I - beta*E

def dI_dt(E, I, beta, gamma, psi):
    return beta*E - gamma*I - psi*I

def dR_dt(I, gamma):
    return gamma*I

def dD_dt(I, psi):
    return psi*I


def multi_plot(M, figsize=(10,4), susceptible = True, labels=False, interventions=False):
    n = M.shape[0]
    CC0 = M[2,0]+M[3,0]+M[4,0]
    CCases = np.diff(M[2]+M[3]+M[4], prepend=CC0).cumsum()
    Deaths = M[4]
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_axes([0.1, 1, 1.25, 1], ylabel='# of People')
    ax2 = fig.add_axes([0.1, 0, 1.25, 1], ylabel='# of People')    
    if susceptible == True:
        rows=range(0,n)
    else:
        rows=range(1,n)
    for ii in rows:
        if labels == False:
            ax1.plot(M[ii])
        else:
            ax1.plot(M[ii], label = labels[ii])
    if interventions==False:
        ax1.set_title('Time Evolution without intervention')
    else:
        ax1.set_title('Time Evolution with intervention')
        for action, day in zip(list(interventions.keys()), [interventions[kk]['day'] for kk in list(interventions.keys())]):
            ax1.axvline(x=day,label=action, linestyle='--')
            ax2.axvline(x=day,label=action, linestyle='--')
    ax1.legend(loc='best')
    ax2.plot(CCases, label='ConfirmedCases', color='brown')
    ax2.plot(Deaths, label='Deaths', color='black')
    ax2.legend(loc='best')
    ax2.set_xlabel('Days')
    ax1.grid(b=True, which='major', c='k', lw=0.5, ls='--')
    ax2.grid(b=True, which='major', c='k', lw=0.5, ls='--')
    plt.show()