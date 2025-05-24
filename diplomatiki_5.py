from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
import matplotlib.pyplot as plt
import numpy as np
No=2e-11
def dbm_to_mw(dbm):
    y=10**(dbm/10)
    return y


def fitness(individual):
    
    Pc_dbm, Pd_dbm, htx2rx, dtx2rx, dd2d, a, dc2d, hd2d, hc2d = individual

    Pd = dbm_to_mw(Pd_dbm)
    Pc = dbm_to_mw(Pc_dbm)

    signal = Pd * abs(htx2rx) ** 2 * dtx2rx ** (-a)
    interference = Pc * abs(hc2d) ** 2 * dc2d ** (-a)+Pd * abs(hd2d) ** 2 * dd2d ** (-a)
    sinr = signal / (interference + No)
    sinr_db = 10 * np.log10(sinr)

    return sinr_db

problem={'num_vars':9,
         'names':['Pc_dbm','Pd_dbm','htx2rx','dtx2rx','dd2d','a','dc2d','hd2d','hc2d'],
         'bounds':[
             [10,24],
             [10,24],
             [0.1,1.5],
             [5,100],
             [5,100],
             [2.75,3],
             [5,100],
             [0.1,1.5],
             [0.1,1.5]
             ]
    }
param_values = morris_sample.sample(problem, 1024)

Y = np.array([fitness(x) for x in param_values])
si= morris_analyze.analyze(problem, param_values, Y, conf_level=0.95, print_to_console=True)

#plot
params = problem['names']
mu_star = si['mu_star']


plt.figure(figsize=(12, 6))
plt.bar(params, mu_star, color='skyblue')
plt.xlabel('Παράμετρος')
plt.ylabel('Mu_star')
plt.title('Mu_star ανά Παράμετρο')
plt.xticks(rotation=60)
plt.grid(axis='y', linestyle='--')
plt.show()
