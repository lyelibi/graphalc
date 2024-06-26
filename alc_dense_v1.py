#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:59:48 2020

@author: lionel

revised ALC dense with a cleaner and faster runtime.
The present code is very close to the sparse version too,
which makes their runtime complexity not too different
the goal would be to essentially demonstrate alc robustness to missing
data.
"""

import numpy as np
from numba import vectorize


@vectorize('float64(float64, float64)', nopython=True)
def lc(corr, size):

    if size <=1.0:
        return 0.0
    if corr <= size:
        return 0.0
    if size**2 - corr <= 0.0:
        corr-=1e-3
    return 0.5*( np.log(size/corr) +  (size - 1.0)*np.log( (size**2 - size) / ( size**2 - corr) )  )


@vectorize('float64(float64, float64, float64, float64)', nopython=True)
def merge(corr_com, corr_neighbors, cross_neighbors , size):
    
    corr = corr_com + corr_neighbors + 2.0 * cross_neighbors
    if corr <= 0:
        return 0.0
    if size <=1.0:
        return 0.0
    if corr <= size:
        return 0.0
    if size**2 - corr <= 0.0:
        corr-=1e-3
        
    return 0.5*( np.log(size/corr) +  (size - 1.0)*np.log( (size**2 - size) / ( size**2 - corr) )  )

def alc(G):
    
    
    N = len(G)
    communities = np.arange(N)
    tracker = np.arange(N)
    ns = np.ones(N)
    all_seen = False
    while not all_seen:
        all_seen = True
        index = 0
        while index < len(tracker):            
            community = tracker[index]
            community_neighbors = np.nonzero(G[community])[0]
            community_neighbors = community_neighbors[community_neighbors!=community]
            A = lc(G[community,community], ns[community])
            B = lc(G[community_neighbors,community_neighbors], ns[community_neighbors])
            C = merge(G[community, community],
                      G[community_neighbors, community_neighbors],
                      G[community, community_neighbors],
                      ns[community] + ns[community_neighbors])
            scores = C - ( A + B )      
            
            if np.all( scores <= 0 ):
                index+=1
                continue
            
            pick = community_neighbors[np.argmax(scores)]
            pick_neighbors = np.nonzero(G[pick])[0]
            pick_neighbors = pick_neighbors[pick_neighbors!=community]
            pick_neighbors = pick_neighbors[pick_neighbors!=pick]
            G[community, pick_neighbors]+=G[pick, pick_neighbors]
            G[pick_neighbors, community]+=G[pick_neighbors, pick]
            G[community, community] = G[community,community] + G[pick, pick] + 2*G[pick, community]
            G[pick] = 0
            G[:,pick] = 0
            communities[ pick ] = community
            ns[ community ] += ns[ pick ]
            ns[ pick ] =  0
            all_seen = False
    
            tracker = np.unique(communities)
            index = 0 #Reset the index 
            
            break
            
        if all_seen:
            return communities