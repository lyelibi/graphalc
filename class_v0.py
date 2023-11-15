''' Prototype two which is fast and merges using the method merge all.'''


import numpy as np
from sklearn.base import BaseEstimator
import numba
from umap_contructor import build_graph
from warnings import warn
import networkx as nx
import graphblas_algorithms as ga
import graphblas as gb

DISCONNECTION_DISTANCES = {
    "correlation": 0.6,
    "cosine": 0.6,
    "hellinger": 1,
    "jaccard": 1,
    "dice": 1,
}



''' the cluster function:
    compute the likelihood which occurs when two objects are clustered or
    the object own likelihood. By design if a cluster only containts one object
    its likelihood will be 0'''


@numba.vectorize([numba.float64(numba.float64, numba.float64)], nopython=True)
def Likelihood(omega_comm, epsilon_comm):
    '''
    This here is the graph giada-marsili likelihood, but the function could
    replace with any cluster quality function.

    Parameters
    ----------
    omega_comm : TYPE
        DESCRIPTION.
    epsilon_comm : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    nu = 1e-3

    if epsilon_comm == 1:
        return 0
    if omega_comm <= epsilon_comm:
        return 0
    A = np.log(epsilon_comm / (omega_comm - nu))
    B = (epsilon_comm - 1) * np.log((epsilon_comm**2 - epsilon_comm) / (epsilon_comm**2 - (omega_comm - nu)))
    output = 0.5*(A + B)
    return output








class GRAPHALC(BaseEstimator):


    def __init__(
        self,
        n_neighbors=100,
        max_merge = 50,
        metric="correlation",
        metric_kwds=None,
        # low_memory=True,
        n_jobs=-1,
        # set_op_mix_ratio=1.0,
        # local_connectivity=1.0,
        # random_state=None,
        # angular_rp_forest=False,
        verbose=False,
        # tqdm_kwds=None,
        disconnection_distance=None,
    ):
        '''
        

        Parameters
        ----------
        n_neighbors : TYPE, optional
            DESCRIPTION. The default is 100.
        max_merge : TYPE, optional
            DESCRIPTION. The default is 50.
        metric : TYPE, optional
            DESCRIPTION. The default is "correlation".
        metric_kwds : TYPE, optional
            DESCRIPTION. The default is None.
        # low_memory : TYPE, optional
            DESCRIPTION. The default is True.
        n_jobs : TYPE, optional
            DESCRIPTION. The default is -1.
        # set_op_mix_ratio : TYPE, optional
            DESCRIPTION. The default is 1.0.
        # local_connectivity : TYPE, optional
            DESCRIPTION. The default is 1.0.
        # random_state : TYPE, optional
            DESCRIPTION. The default is None.
        # angular_rp_forest : TYPE, optional
            DESCRIPTION. The default is False.
        verbose : TYPE, optional
            DESCRIPTION. The default is False.
        # tqdm_kwds : TYPE, optional
            DESCRIPTION. The default is None.
        disconnection_distance : TYPE, optional
            DESCRIPTION. The default is None.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.n_neighbors = n_neighbors
        self.max_merge = max_merge
        self.metric = metric
        self.metric_kwds = metric_kwds
        # self.low_memory = low_memory
        # self.set_op_mix_ratio = set_op_mix_ratio
        # self.local_connectivity = local_connectivity
        # self.random_state = random_state
        # self.angular_rp_forest = angular_rp_forest
        self.verbose = verbose
        # self.tqdm_kwds = tqdm_kwds
        self.disconnection_distance = disconnection_distance

        self.n_jobs = n_jobs
        
    def _validate_parameters(self):

        
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 1")
        if self.metric_kwds is None:
            self._metric_kwds = {}
        else:
            self._metric_kwds = self.metric_kwds
            
        
        if self.disconnection_distance is None:
            self._disconnection_distance = DISCONNECTION_DISTANCES.get(
                self.metric, np.inf
            )
        elif isinstance(self.disconnection_distance, int) or isinstance(
            self.disconnection_distance, float
        ):
            self._disconnection_distance = self.disconnection_distance
        else:
            raise ValueError("disconnection_distance must either be None or a numeric.")


    def fit(self, X):
        
        self._validate_parameters()

        if self.verbose:
            print(str(self))
        
        G = build_graph(X, self._disconnection_distance, metric=self.metric, n_neighbors = self.n_neighbors, verbose=self.verbose)
        N = G.shape[0]

        if nx.is_connected(ga.Graph(gb.Matrix.from_csr(
                  G.indptr,
                  G.indices,
                  G.data,
                  nrows=N, ncols=N, dtype='float32'
                ))):
            warn('Graph is Connected')
        else:
            warn('Graph is not Connected')

        
        E = G.copy()
        E.data = E.data - E.data + 1
        G = G.tolil()
        E = E.tolil()
        del X
                
        
        ''' the cluster function:
            compute the likelihood which occurs when two objects are clustered or
            the object own likelihood. By design if a cluster only containts one object
            its likelihood will be 0'''
        
        
        
        
        
        ''' aspc only requires a correlation matrix as input:
            here we convert the correlation to a dictionary for convenience. adding new
            entries in a dict() is much faster than editing a numpy matrix'''
        
        
        
        ''' tracker is dictionary which stores the objects member of the same clusters.
            the data is stored as strings: i.e. cluster 1234 contains objects 210 & 890
            which results in tracker['1234'] == '210_890' '''
        
          
        weights = np.ones(N)
        ns_vec = np.ones(N)
        edges = np.zeros(N)
        communities = np.arange(G.shape[0]).astype(int)
        
        
        ''' Create a list of object indices 'other_keys': at every iteration one object
         is clustered and removed from the list. It is also removed if no suitable
         optimal destination is found.'''
        
        ''' the operation stocks once there is only one object left to cluster as we
        need two objects at the very least.'''
        stopper = 0
        ckeys = range(N)
        n_communities = N
        label = 0
        
        
        while stopper <= n_communities:
            k =  ckeys[label]    
            improvement = True
        
            
            while improvement == True:
                community_neighbors = np.unique( communities[ G.rows[ communities[k] ] ] )                
                nsk = ns_vec[ communities[k] ]
                omega_k = weights[ communities[k] ]
                epsilon_k = np.sqrt( 2*edges[ communities[k] ] + nsk )        
                L_k = Likelihood(omega_k, epsilon_k)
                
                nsix = ns_vec[ community_neighbors ]         
                ns =  nsk + nsix.sum()
                epsilon_comm = np.sqrt( 2*edges[ communities[ community_neighbors ] ].sum() + 2*edges[ communities[k] ] + ns + 2 * E[communities[ k ] , communities[ community_neighbors ] ].sum() )
                omega_comm = weights[ communities[ community_neighbors ] ].sum() + weights[ communities[k] ]  + 2 * G[communities[k], communities[ community_neighbors ]].sum()
                cost = Likelihood(omega_comm, epsilon_comm) - ( L_k + sum( Likelihood(weights[ communities[ community_neighbors ] ], np.sqrt( 2*edges[ communities[ community_neighbors ] ] + nsix))) )

                if cost <= 0:
                    stopper+=1
                    improvement = False
                    label+=1
                    if label >= len(ckeys):
                        label = 0
                    continue
                
                neighbors_to_merge = community_neighbors
        
                ''' update arrays'''
                edges[ communities[k]  ] += (edges[ neighbors_to_merge  ] + E[communities[ k ] , neighbors_to_merge ]).sum()    
                weights[ communities[k]  ] += (weights[ neighbors_to_merge  ]  + 2 * G[ communities[k], neighbors_to_merge ]).sum()      
                    
                ''' need to get the communities of the rows not just the rows'''
                
        
                for new_community in neighbors_to_merge:
        
                    rowg = np.unique( communities[ G.rows[ new_community ]] )
                    rowg = rowg[ rowg != communities[k] ]                
                    G[ communities[k], rowg ]+= G[ new_community , rowg ]
                    E[ communities[k], rowg ]+= E[ new_community , rowg ]
                    
                upg =  G[ communities[k] ].copy()
                upe =  E[ communities[k] ].copy()
        
                # for new_community in neighbors_to_merge:
        
                G.rows[neighbors_to_merge] = [[]] * len(neighbors_to_merge)
                G.data[neighbors_to_merge] = [[]] * len(neighbors_to_merge)
                E.rows[neighbors_to_merge] = [[]] * len(neighbors_to_merge)
                E.data[neighbors_to_merge] = [[]] * len(neighbors_to_merge)
                G[ communities[k]]= upg
                E[ communities[k] ]= upe

                mask = np.isin(communities, neighbors_to_merge)
                communities[mask] = communities[k]
                stopper = 0
                ckeys = np.unique(communities)
        
                n_communities = ckeys.shape[0]
                ns_vec[ communities[k] ] = ns
        unique_values = np.unique(communities)
        relabel_dict = {old: new for new, old in enumerate(unique_values)}
        # relabel the array
        new_arr = np.array([relabel_dict[val] for val in communities])   
        self.communities = new_arr     
    
    
    
    def fit_transform(self, X):
        self.fit(X)
        return self.communities
