import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def SIR_model(N, D, R_0, CaseFatalityRate, max_days, figsize=(10,4), ylim=(0,1.2), log_y_axis=False, ylimit=None):
    '''
    N: total population
    D, R_0, CaseFatalityRate: see texts above
    '''
    I0, R0 = 1, 0  # Initial number of infected and recovered individuals (1 infected, 0 recovered) [this R0 has nothing to do with the basic reproduction number R0]
    S0 = N - I0 - R0 # Initial number of susceptible (everyone else)

    gamma = 1.0 / D  # see texts above
    beta = R_0 * gamma  # see texts above
    alpha = CaseFatalityRate

    t = np.linspace(0, max_days, max_days) # Grid of time points (in days)
    
    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    
    # Adding deaths (see text above)
    X = np.zeros(max_days)
    for day in range(12, max_days):
        X[day] = sum(I[:day-12])
    # X[13:] = I[:-13]
    X = alpha * X * gamma
    
    print('[STATS] Fatalities:: {0:8.0f}'.format(np.max(X)))
    print('[STATS] Infected:: {0:8.0f}'.format(np.max(I)))
    print('[STATS] Recovered:: {0:8.0f}'.format(np.max(R)))

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    f, ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, X, 'r', alpha=0.7, linewidth=2, label='Dead')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')

    ax.set_xlabel('Time (days)')
    ax.title.set_text('SIR-Model. Total Population: ' + str(N) + ", Days Infectious: " + str(D) + ", R_0: " + str(R_0) + ", CFR: " + str(CaseFatalityRate*100))
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    if log_y_axis:
        ax.set_yscale('log')
    if ylimit is not None:
        ax.set_ylim(ylimit[0],ylimit[1])
    ax.grid(b=True, which='major', c='k', lw=0.5, ls='--')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show();


def SIR_model_with_lockdown(N, D, R_0, CaseFatalityRate, max_days, L, R_0_2, figsize=(10,4), ylim=(0,1.2), log_y_axis=False, ylimit=None):
    '''
    N: total population
    D, R_0, CaseFatalityRate, ...: see texts above
    '''
    # BEFORE LOCKDOWN (same code as first model)
    I0, R0 = 1, 0  # Initial number of infected and recovered individuals (1 infected, 0 recovered) [this R0 has nothing to do with the basic reproduction number R0]
    S0 = N - I0 - R0 # Initial number of susceptible (everyone else)

    gamma = 1.0 / D  # see texts above
    beta = R_0 * gamma  # see texts above
    alpha = CaseFatalityRate

    t = np.linspace(0, L, L)  # Grid of time points (in days)
    
    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    
    
    # AFTER LOCKDOWN
    I0_2, R0_2, S0_2 = I[-1], R[-1], S[-1]  # beginning of lockdown -> starting Infected/Susceptible/Recovered numbers are the numbers at the end of no-lockdown period

    gamma = 1.0 / D  # constant
    beta_2 = R_0_2 * gamma
    alpha = CaseFatalityRate  # constant

    t_2 = np.linspace(0, max_days - L + 1, max_days - L + 1)
    
    # Initial conditions vector
    y0_2 = S0_2, I0_2, R0_2
    # Integrate the SIR equations over the time grid, t.
    ret_2 = odeint(deriv, y0_2, t_2, args=(N, beta_2, gamma))
    S_2, I_2, R_2 = ret_2.T

    
    # COMBINING PERIODS
    S_full = np.concatenate((S, S_2[1:]))
    I_full = np.concatenate((I, I_2[1:]))
    R_full = np.concatenate((R, R_2[1:]))
    t_full = np.linspace(0, max_days, max_days)
    
    # Adding deaths (see text above)
    X = np.zeros(max_days)
    for day in range(12, max_days):
        X[day] = sum(I_full[:day-12])
    # X[13:] = I[:-13]
    X = alpha * X * gamma

    print('[STATS] Fatalities:: {0:8.0f}'.format(np.max(X)))
    print('[STATS] Infected:: {0:8.0f}'.format(np.max(I_full)))
    print('[STATS] Recovered:: {0:8.0f}'.format(np.max(R_full)))

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    f, ax = plt.subplots(1,1,figsize=figsize)
    ax.plot(t_full, S_full, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t_full, I_full, 'y', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t_full, X, 'r', alpha=0.7, linewidth=2, label='Dead')
    ax.plot(t_full, R_full, 'g', alpha=0.7, linewidth=2, label='Recovered')

    ax.set_xlabel('Time (days)')
    ax.title.set_text('SIR-Model with Lockdown. Total Population: ' + str(N) + 
                      ", Days Infectious: " + str(D) + ", R_0: " + str(R_0) + 
                      ", CFR: " + str(CaseFatalityRate*100) + " R_0_2: " + str(R_0_2) + 
                      ", L: " + str(L) + " days")
    if log_y_axis:
        ax.set_yscale('log')
    if ylimit is not None:
        ax.set_ylim(ylimit[0],ylimit[1])
    plt.text(L,N/20,'Lockdown')
    plt.plot(L, 0, marker='o', markersize=6, color="red")
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='k', lw=0.5, ls='--')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show();


def SIR_model_with_lockdown_deaths(x, N, D, R_0, CaseFatalityRate, max_days, L, R_0_2, 
                                    figsize=(10,4), ylim=(0,1.2), log_y_axis=False, ylimit=None):
    '''
    We now try to fit the SIR-Model's Dead Curve to real data by tweaking the variables. Some of them are constant:
        - max_days is set to `len(train.groupby("Date").sum().index)` so that we can compare against all available data
        - N is fixed for each country, that's just the total population
        - L is fixed for each country (the date it went into lockdown)
        - D is set to vary from 5 to 20 (according to [this study](https://www.ncbi.nlm.nih.gov/pubmed/32150748), it takes on avg. 5 days to show symptoms, at most 14; according to [this source (German)](https://www.rki.de/DE/Content/InfAZ/N/Neuartiges_Coronavirus/Steckbrief.html#doc13776792bodyText5), people are infectious up to 5 days after onset of symptoms).
        - CFR set to vary from $0.1\% - 10\%$ (according to [this study](https://wwwnc.cdc.gov/eid/article/26/6/20-0320_article))
        - R_0 and R_0_2 are set to vary from 0.1 to 3.5
    
    '''
    # BEFORE LOCKDOWN (same code as first model)
    I0, R0 = 1, 0  # Initial number of infected and recovered individuals (1 infected, 0 recovered) [this R0 has nothing to do with the basic reproduction number R0]
    S0 = N - I0 - R0 # Initial number of susceptible (everyone else)

    gamma = 1.0 / D  # see texts above
    beta = R_0 * gamma  # see texts above
    alpha = CaseFatalityRate

    t = np.linspace(0, L, L)  # Grid of time points (in days)
    
    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    
    
    # AFTER LOCKDOWN
    I0_2, R0_2, S0_2 = I[-1], R[-1], S[-1]  # beginning of lockdown -> starting Infected/Susceptible/Recovered numbers are the numbers at the end of no-lockdown period

    gamma = 1.0 / D  # constant
    beta_2 = R_0_2 * gamma
    alpha = CaseFatalityRate  # constant

    t_2 = np.linspace(0, max_days - L + 1, max_days - L + 1)
    
    # Initial conditions vector
    y0_2 = S0_2, I0_2, R0_2
    # Integrate the SIR equations over the time grid, t.
    ret_2 = odeint(deriv, y0_2, t_2, args=(N, beta_2, gamma))
    S_2, I_2, R_2 = ret_2.T

    
    # COMBINING PERIODS
    S_full = np.concatenate((S, S_2[1:]))
    I_full = np.concatenate((I, I_2[1:]))
    R_full = np.concatenate((R, R_2[1:]))
    t_full = np.linspace(0, max_days, max_days)
    
    # Adding deaths (see text above)
    X = np.zeros(max_days)
    for day in range(12, max_days):
        X[day] = sum(I_full[:day-12])
    # X[13:] = I[:-13]
    X = alpha * X * gamma

    print('[STATS] Fatalities:: {0:8.0f}'.format(np.max(X)))
    print('[STATS] Infected:: {0:8.0f}'.format(np.max(I_full)))
    print('[STATS] Recovered:: {0:8.0f}'.format(np.max(R_full)))

    return X[x]

from lmfit import Model

def fit_SIR(country_name, lockdown_date, region_name=None):
    """
    y_data: the fatalities data of one country/region (array)
    population: total population of country
    lockdown_date: format YYYY-MM-DD
    """
    if region_name:
        y_data = train[(train["Country_Region"] == country_name) & (train["Region"] == region_name)].Fatalities.values
    else:
        if len(train["Country_Region"] == country_name) > len(train["Country_Region"] == "Germany"):  # country with several regions and no region provided
            y_data = train[(train["Country_Region"] == country_name) & (train["Region"].isnull())].Fatalities.values
        else:
            y_data = train[train["Country_Region"] == country_name].Fatalities.values
        
    max_days = len(train.groupby("Date").sum().index) # constant for all countries

    # country specific values
    N = world_population.loc[world_population['Country (or dependency)'] == country_name]["Population (2020)"].values[0]
    L = train.groupby("Date").sum().index.tolist().index(lockdown_date)  # index of the lockdown date

    # x_data is just [0, 1, ..., max_days] array
    x_data = np.linspace(0, max_days - 1, max_days, dtype=int)
    
    # curve fitting from here
    mod = Model(SIR_model_with_lockdown_deaths)

    # initial values and bounds
    mod.set_param_hint('N', value=N)
    mod.set_param_hint('max_days', value=max_days)
    mod.set_param_hint('L', value=L)
    mod.set_param_hint('D', value=10, min=2, max=25)
    mod.set_param_hint('CaseFatalityRate', value=0.01, min=0.0001, max=0.1)
    mod.set_param_hint('R_0', value=2.0, min=0.1, max=5.0)
    mod.set_param_hint('R_0_2', value=2.0, min=0.1, max=5.0)

    params = mod.make_params()

    # fixing constant parameters
    params['N'].vary = False
    params['max_days'].vary = False
    params['L'].vary = False

    result = mod.fit(y_data, params, x=x_data)
    
    return result, country_name


def fitted_plot(result, country_name, region_name=None):
    if region_name:
        y_data = train[(train["Country_Region"] == country_name) & (train["Region"] == region_name)].Fatalities.values
    else:
        if len(train["Country_Region"] == country_name) > len(train["Country_Region"] == "Germany"):  # country with several regions and no region provided
            y_data = train[(train["Country_Region"] == country_name) & (train["Region"].isnull())].Fatalities.values
        else:
            y_data = train[train["Country_Region"] == country_name].Fatalities.values

    max_days = len(train.groupby("Date").sum().index)
    x_data = np.linspace(0, max_days - 1, max_days, dtype=int)
    x_ticks = train[train["Country_Region"] == "Germany"].Date.values  # same for all countries
    
    plt.figure(figsize=(10,5))
    
    real_data, = plt.plot(x_data, y_data, 'bo', label="real data")
    SIR_fit = plt.plot(x_data, result.best_fit, 'r-', label="SIR model")
    
    plt.xlabel("Day")
    plt.xticks(x_data[::10], x_ticks[::10])
    plt.ylabel("Fatalities")
    plt.title("Real Data vs SIR-Model in " + country_name)
    plt.legend(numpoints=1, loc=2, frameon=None)
    plt.show()