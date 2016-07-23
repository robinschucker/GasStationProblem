'''
Author: Robin Schucker (schucker.robin[at]gmail.com)
date: 05/11/16
----------------------------------------------------
Gas Station Problem:
    Given a set of gas stations with GPS coordinates and the price per gallon
    compute the optimal path from a start point to end point. From any point
    it is possible to reach any other point in a straight line
    Given constraints:
        - do not run out of gas
        - minimise length of trip
        - minimise cost
        
    Minimising the length and minimising the cost are two different objectives
    and so I chose to optimise both at the same time using a weighted cost
    function:
        C = cost of gas + cost of time
        where 'cost of time' = distance * ALPHA, and ALPHA some constant
        ALPHA = 1/6 corresponds to 10$/h at 60mph
        ALPHA = 0 corresponds to the minimum cost solution
        and a big ALPHA (in practice 10) corresponds to the minimum length 
        solution
        
    This algorithm is largely inspired by:
    [1] "To Fill or not to Fill: The Gas Station Problem", Samir Khuller,
            Azarakhsh Malekian and Julian Mestre
        http://www.cs.umd.edu/projects/gas/gas-station.pdf
        
    It recursively search all possible options starting from the finish point
    with a limited allowed number of stops.
    
    It has a runtime of O(MAX_STOP * n^2 * log(n)) where n is the number of
    gas stations in file. With 8000 data points and MAX_STOP = 10
    it takes about 60 minutes on a standard mac laptop

Input file:
    The csv input file must have the following format:
    ,fuelPrice,latitude,longitude
    0,1.46,32.3717248,-112.8607099
    .
    .
    .
    last_index,1.48,39.5383008,-110.2207786
    start,0.0,30.0,-99.0
    finish,x,40.,-122.0
    
Possible performance gain:
    - With a big maximum range, not many stops are required and the algorithm
    only choses cheap gas stations. In it possible to seriously reduce the
    number of gas stations to consider in the first place by perfomring
    a clustering algorithm (k-means for example) and only keep a fraction of 
    the cheapest gas stations within the cluster. The number of clusters to
    choose is highly dependent on the maximum range of the car (thus the number
    of stops to make). 
    
    - If only interested in the minimum distance, a simple A* search on the
    graph is probably more efficient
'''


import numpy as np
import sys

from CSVData import CSVData

MPG = 25.0
START_GAS = 20
MAX_GAS = 40
START_RANGE = START_GAS * MPG;
MAX_RANGE = MAX_GAS * MPG
MAX_STOP = 10 # limit on how many times driver can stop

ALPHA = 0.3 / 6 # cost of driving distance (time), 1/6 $/mi <=> 10$/h at 60 mph

FILENAME = 'GasStations.csv' # use GasStations_tiny.csv or _small to debug

''' Function that computes the distance matrix in miles between a list of
    points in GPS coordinates
    Input: numpy arrays of shape (N,)
    Output: numpy array of shpae (N,N)
'''
def distance(lat, long):
    EARTH_RADIUS = 3959 # miles
    N = long.shape[0]
    
    # convert to radians
    rlong = np.radians(long)
    rlat = np.radians(lat)
    
    dlat = rlat.reshape((N,1)) - rlat
    dlong = rlong.reshape((N,1)) - rlong
    a = (np.sin(dlat / 2) ** 2 + np.cos(rlat).reshape((N,1)) * np.cos(rlat) 
            * np.sin(dlong / 2) ** 2)
    miles = EARTH_RADIUS * 2 * np.arcsin(np.sqrt(a))
    return miles
    
''' Function that loads the appropriate data from the csv file
    Input: filename
    Output:
        num_points = number of points + 1 for start (= N)
        lat = numpy array of shape (N + 1,) representing the latitudes
        long = array of longitudes (N + 1,)
        price = array of gas prices (converted to $/mi) (N,)
        true_idx = array to match the indicies inside the csv (some gas
            stations are missing) (N - 1,)
'''
def load_data(filename):
    with open(filename,'r') as f:
        data = CSVData(f,n_header = 1)
    num_points = data.number_data.shape[0] - 1 #take out finish
    lat = data.number_data[:,2]
    long = data.number_data[:,3]
    price = data.number_data[:-1,1] / MPG #in $ / mile
    true_idx = data.number_data[:,0] # somes missing indicies in csv
    return (lat, long, price, true_idx, num_points)

''' Function that discretize the range values
    As per [1], section 2.1. When we consider the optimal solution, the car
    can only at point u, and coming from v either with 0 range left or
    MAX_RANGE - dist(u,v) left. Following that, there is only a finite number
    of ranges at which the car can arrive
    Input: 
        dist: distance matrix of shape (N, N) where dist[u,v] = dist[v,u]
            is the distance from point u to point v
        price: price[u] = gas price at point u
    Output:
        range_values: dictionary from u -> np.array() containing the possible
            range values for point u
'''
def discretize_range_values(dist, price):
    # RV[u] = { MAX_RANGE - d[u,v] | for all v where price[v] < price[u]
    #                            and d[u,v] <= MAX_RANGE} U {0}
    range_values = {}
    C = MAX_RANGE * np.ones((1, num_points))
    C[0,-1] = START_RANGE
    temp = C - dist
    for u in xrange(num_points - 1):
        mask = np.logical_and(price < price[u],temp[u,:] >= 0) 
        range_values[u] = np.append(temp[u,mask],0)
        range_values[u].sort()
    return range_values

''' Function that initializes the cost table.
    Input:
        dist: distance matrix of shape (N, N) where dist[u,v] = dist[v,u]
            is the distance from point u to point v
        price: price[u] = gas price at point u
        range_values: dictionary from u -> np.array() containing the possible
            range values for point u
    Output:
        cost: dictionary from (u,rv) -> (min_cost, [path])
        where: - min_cost is the minimum cost to go from u to the finish in one
            trip with a range of rv when arriving at point u.
            If min_cost = infinity, the trip is impossible
            - [path]: list containing the optimal path to go from u to finish 
            at this minimum cost
'''
def initialize_cost(dist, range_values, price, num_points):
    # C[u,rv] = { (d[u,t] - rv)*price[u] if rv <= d[u,t] <= MAX_RANGE
    #           { inf       otherwise
    cost = {}
    for u in xrange(num_points - 1):
        for rv in range_values[u]:
            if (rv <= dist_from_target[u] and dist_from_target[u] <= MAX_RANGE):
                cost[(u, rv)] = ((dist_from_target[u] - rv) * price[u] 
                        + dist_from_target[u] * ALPHA, [num_points])
            else:
                cost[(u, rv)] = (np.inf, [])
    return cost
    
''' Function that given the cost mapping using (n-1) stops computes the cost
    mapping using n stops
    
    Input:
        dist: distance matrix of shape (N, N) where dist[u,v] = dist[v,u]
            is the distance from point u to point v
        price: price[u] = gas price at point u
        range_values: dictionary from u -> np.array() containing the possible
            range values for point u$
        cost_old: dictionary from (u,rv) -> (min_cost, [path])
            where: - min_cost is the minimum cost to go from u to the finish in 
            (n-1) stops with a range of rv when arriving at point u. (u is
            counted as stop)
            If min_cost = infinity, the trip is impossible
            - [path]: list containing the optimal path to go from u to finish 
            at this minimum cost
    Output:
        cost: dictionary from (u,rv) -> (min_cost, [path])
            where: - min_cost is the minimum cost to go from u to the finish in
            n stops with a range of rv when arriving at point u.
            If min_cost = infinity, the trip is impossible
            - [path]: list containing the optimal path to go from u to finish 
            at this minimum cost
'''
def cost_extra_step(cost_old, dist, price, range_values, num_points):
    # C[u,rv] = min { C_old[v,0] + (d[u,v] - rv) * price[u] + d[u,v] * ALPHA
    #          v s.t                  if price[v] <= price[u] and rv <= d[u,v]
    #  d[u,v] <= MAX_RANGE
    #               { C_old[v,MAX_RANGE - d[u,v]] + (MAX_RANGE - rv) * price[u]
    #                                             + d[u,v] * ALPHA
    #                                 if price[v] > price[u]
    
    cost = {}
    for u in xrange(num_points - 1):
        if (u % 300 == 0):
            sys.stdout.write('.')
            sys.stdout.flush()
        
        # compute all range_value independent terms
        # ind1 corresponds to the term C_old[v,0] + d[u,v] * (price[u] + ALPHA)
        # ind2 corresponds to the term C_old[v,MAX_RANGE - d[u,v]] 
        #                              + MAX_RANGE * price[u] + d[u,v] * ALPHA
        ind1 = []
        ind2 = []
        for v in xrange(num_points - 1):
            d = dist[u,v]
            if (d > MAX_RANGE or u == v):
                continue
            if (price[v] <= price[u]):
                c, prev = cost_old[(v,0)]
                if (prev):
                    prev = list(prev) #copy
                    prev.append(v)
                    ind1.append((c + d * (price[u] + ALPHA), prev))
            else:
                c, prev = cost_old[(v,MAX_RANGE - d)]
                if (prev):
                    prev = list(prev) #copy
                    prev.append(v)
                    ind2.append((c + d * ALPHA + MAX_RANGE * price[u], prev))
        
        # then find minimum of {ind1, ind2} over all v as long a rv <= d[u,v]
        # for the term in ind1
        ind1.sort(reverse = True) #to pop from end of list efficiently
        if ind2:
            min_ind2 = min(ind2)
        else:
            min_ind2 = (np.inf, [])
        
        # bool that becomes true once min comes from ind2
        if ind1:
            from_ind2 = min_ind2 <= ind1[-1]
        else:
            from_ind2 = True
        
        # fill in the range dependent values and deactivate terms in
        # ind1 list once rv > d[u,v]
        for rv in range_values[u]:
            if (from_ind2):
                cost[(u,rv)] = (min_ind2[0] - rv * price[u], min_ind2[1])
            else:
                #deactivate terms once d < rv
                while(ind1 and dist[ind1[-1][1][-1],u] < rv): 
                    del ind1[-1]
                if (ind1 and min_ind2 > ind1[-1]):
                    cost[(u,rv)] = (ind1[-1][0] - rv * price[u], ind1[-1][1])
                else:
                    from_ind2 = True
                    cost[(u,rv)] = (min_ind2[0] - rv * price[u], min_ind2[1])
    return cost

''' Function that given the cost mapping using at n stops checks if it is
    possible to the finish from the start. (start -> u -> finish = 1 stop)
    
    Input:
        dist: distance matrix of shape (N, N) where dist[u,v] = dist[v,u]
            is the distance from point u to point v
        cost: dictionary from (u,rv) -> (min_cost, [path])
            where: - min_cost is the minimum cost to go from u to the finish in 
            n stops with a range of rv when arriving at point u.
            If min_cost = infinity, the trip is impossible
            - [path]: list containing the optimal path to go from u to finish 
            at this minimum cost
    Output:
        cost: dictionary from (u,rv) -> (min_cost, [path])
            where: min_cost is the minimum cost to go from u to the finish in n
            stops with a range of rv when arriving at point u.
            If min_cost = infinity, the trip is impossible
        [path]: list containing the optimal path to go from u to finish at this
            minimum cost
'''
def cost_start_to_finish(cost, dist, num_points):
    min_cost = np.inf
    prev = []
    for v in xrange(num_points - 1):
        d = dist[v,-1] #dist from start
        if (d > START_RANGE):
            continue
        ccost, prevv = cost[(v,START_RANGE - d)]
        if (ccost < min_cost):
            min_cost = ccost + ALPHA * d
            prev = list(prevv)
            prev.append(v)
            prev.pop(0) # remove finish index
    return (min_cost, prev)

    
''' Function that checks if path from start to finish is possible and prints
    the optimal path contained in the list prev
'''
def print_path(min_cost, prev, cost, price, true_idx):
    if (min_cost != np.inf):
        print 'path found with cost: {}'.format(min_cost)   
        str = 'start -> '    
        for v in reversed(prev):
            str += '{} -> '.format(true_idx[v])
        str += 'finish'
        print str

'''
# sample case for debugging:
# best to use with MAX_GAS = 10 and MPG = 10
dist = np.array([[0,20,50,90],[20,0,60,1000],[50,60,0,1000],[90,1000,1000,0]])
dist_from_target = np.array([1000,1000,50,1000])
price = np.array([5,2,4,0]) / MPG
num_points = 4
true_idx = np.arange(5)
'''

# load data
lat, long, price, true_idx, num_points = load_data(FILENAME)

# calculate distance matrix
print 'Calculating distances...'
dist = distance(lat, long)
dist_from_target = dist[-1,:-1]
dist = dist[:-1,:-1]

# calculate possible range values:
print 'Discretizing range values...'
range_values = discretize_range_values(dist, price)

# cost(_old):
# dict from (u, rv) -> (cost, prev_point)
# cost[(u, rv)]: minimum cost to go from u to finish in n (or n-1 for _old)
# steps with starting range of rv
print 'Computing cost for 1 stop...'
cost_old = {}
cost = initialize_cost(dist, range_values, price, num_points)

# Check if problem worth sovling
if (dist_from_target[num_points - 1] <= START_RANGE):
    print ('Problem not well defined, can reach target from start '
           ' with initial range')
    quit()
    
# build table through recursion
for n in range(2, MAX_STOP + 1):
    # check if you can build a path from start to finish
    total_cost, prev = cost_start_to_finish(cost, dist, num_points)
    print_path(total_cost, prev, cost, price, true_idx)

    # compute cost table for next iterration
    sys.stdout.write('Computing cost for {} stop'.format(n))
    cost_old.clear()
    cost_old = cost.copy()
    cost = cost_extra_step(cost_old, dist, price, range_values, num_points)          
    sys.stdout.write('\n')
    
    



