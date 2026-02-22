# -*- coding: utf-8 -*-
"""
Created on Sun May  4 18:59:05 2025

@author: elcha
"""

import numpy as np # math functions
import matplotlib.pyplot as plt # for plotting figures and setting the figures’ properties

#%%

N_gpus = np.linspace(start=100, stop=1000, num=100)
N_hbi = 8
N_spines = 60 # TODO - how many spines we want to put per N_gpus
N_segments = np.round(N_gpus / 1024)
N_x = np.round((N_gpus / N_hbi) **0.5)

P_HBI_tot = 0
P_switch_ToR = 0 #10000
P_switch_agg = 0 #40000
P_link = 1
P_link_expensive = P_link #850
P_link_cheap = P_link #400

#TODO: add to all the topologies the price of the clusters.

P_all_HBIs = P_HBI_tot * N_gpus/N_hbi

def Rail_Only_Price():
    return P_all_HBIs + N_hbi * P_switch_ToR + N_gpus * P_link_cheap

def Fat_Tree_Price():
    return P_all_HBIs + Rail_Only_Price() + N_spines * (P_switch_agg + N_hbi * P_link_expensive)

def HPN_Price():
    return P_all_HBIs + 2*(N_spines * P_switch_agg + N_hbi * N_segments * (P_switch_ToR + N_spines * P_link_expensive) + N_gpus * P_link_cheap)

def HyperX_Price(N_dim1):
    N_dim2 = (N_gpus/N_dim1)**0.5
    return 0.5 * P_link * N_gpus * (N_dim1 + 2*N_dim2)

def HyperX_Price_old():
    return P_all_HBIs + N_gpus / N_hbi * (P_switch_ToR + 0.5*(N_x + N_gpus / (N_hbi * N_x)) * P_link_cheap)

def HyperX_3D_Price():
    N_classters = N_gpus / N_hbi 
    return P_all_HBIs + N_classters * (P_switch_ToR + 0.5*3*N_classters**(1/3)*P_link_cheap)

def DragonFlyP_Price():
    P_hbi = N_hbi*(P_switch_ToR + N_hbi)
    num_of_hbis = N_gpus/N_hbi
    P_click = 0.5 * num_of_hbis**2
    return num_of_hbis*P_hbi + N_hbi * P_click

plt.figure(dpi = 500)
plt.plot(N_gpus, Fat_Tree_Price(), label = "Fat Tree price")
plt.plot(N_gpus, Rail_Only_Price(), label = "Rail Only price")
# plt.plot(N_gpus, HPN_Price(), label = "HPN_price")
plt.plot(N_gpus, HyperX_Price(2), label = "HyperX price dim1=2")
plt.plot(N_gpus, HyperX_Price(8), label = "HyperX price dim1=8")
# plt.plot(N_gpus, HyperX_3D_Price(), label = "HyperX_3D_price")
plt.plot(N_gpus, DragonFlyP_Price(), label = "DragonFly+ price")
plt.xlabel("Number of PGUs")
plt.ylabel("number of links")#("Price")
# plt.title( f"N_hbi = {N_hbi} ; N_spines = {N_spines} ; \n\n \
           # P_switch_ToR = {P_switch_ToR} ; P_switch_agg = {P_switch_agg}")
plt.legend()
plt.grid()