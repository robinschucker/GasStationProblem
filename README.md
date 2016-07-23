# GasStationProblem
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
