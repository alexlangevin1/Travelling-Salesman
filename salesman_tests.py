#File Name: salesman_tests.py
#Author: Alex Langevin
#Date Created: 9 August 2017
#Last Updated: 12 August 2017
#Description: Function to run performance tests of KNN salesman algorithm a user-defined number of times.
#			  Algorithm compared to optimal route calculation and random route selection. Results output to .csv file

import pandas as pd
import numpy as np
import os
import datetime as dt
from salesman_models5 import get_clean_coords, KNN_salesman, random_route, brute_force_optimal_route
from sklearn.utils import shuffle

def salesman_test(num_cities, iterations = 1):
	col_names = ['Number of Cities','Optimal Route','KNN Route 1 Iteration', 'KNN Route 3 Iterations',
	'KNN Route 5 Iterations', 'KNN Route 10 Iterations', 'Random Route 10 Iterations', 'Random Route 100 Iterations',
	'Optimal Time sec', 'KNN Time % of Optimal (1 IT)','KNN Time % of Optimal (3 IT)', 'KNN Time % of Optimal (5 IT)',
	'KNN Time % of Optimal (10 IT)', 'Random Route Time % of Optimal (10 IT)', 'Random Route Time % of Optimal (100 IT)',
	'Optimal Route Distance', 'KNN Distance % of Optimal (1 IT)', 'KNN Distance % of Optimal (3 IT)',
	'KNN Distance % of Optimal (5 IT)', 'KNN Distance % of Optimal (10 IT)', 'Random Route Distance % of Optimal (10 IT)',
	'Random Route Distance % of Optimal (100 IT)']
	
	row_index = list(range(iterations))
	
	scores = pd.DataFrame(index = row_index, columns = col_names)
	
	scores['Number of Cities'] = num_cities
	
	#pull list of 100 tests cities and GPS coordinates from GPS_coordinates.csv
	coords, names = get_clean_coords()
	
	for i in range(iterations):
		
		#randomize test sample for each iteration of performance test
		coords, names = shuffle(coords, names)
		coords, names = shuffle(coords, names)
		test_coords = coords[:num_cities,:]
		test_names = names[:num_cities,:]
	
		start = dt.datetime.now()
		best_route, best_dist = brute_force_optimal_route(test_coords, test_names)
		stop = dt.datetime.now()
		
		#Clocked at microsecond level, elapsed time stored as seconds
		delta = (stop - start).total_seconds()
		
		scores.set_value(i,'Optimal Route',best_route)
		scores.set_value(i,'Optimal Time sec',delta)
		scores.set_value(i,'Optimal Route Distance',best_dist)
		
		start = dt.datetime.now()
		rnd_route, rnd_dist = random_route(test_coords,test_names, 10)
		stop = dt.datetime.now()
		
		delta = (stop - start).total_seconds()
		
		scores.set_value(i,'Random Route 10 Iterations',rnd_route)
		scores.set_value(i,'Random Route Time % of Optimal (10 IT)',delta / scores.ix[i]['Optimal Time sec'])
		scores.set_value(i,'Random Route Distance % of Optimal (10 IT)', rnd_dist / scores.ix[i]['Optimal Route Distance'])
		
		start = dt.datetime.now()
		rnd_route, rnd_dist = random_route(test_coords,test_names, 100)
		stop = dt.datetime.now()
		
		delta = (stop - start).total_seconds()
		
		scores.set_value(i,'Random Route 100 Iterations',rnd_route)
		scores.set_value(i,'Random Route Time % of Optimal (100 IT)', delta / scores.ix[i]['Optimal Time sec'])
		scores.set_value(i,'Random Route Distance % of Optimal (100 IT)',rnd_dist / scores.ix[i]['Optimal Route Distance'])
		
		
		KNN_times = []
		start = dt.datetime.now()
		KNN_route, KNN_dist = KNN_salesman(test_coords,test_names)
		stop = dt.datetime.now()
			
		delta = (stop - start).total_seconds()
		
		best_KNN_route = np.copy(KNN_route)
		KNN_times.append(delta)
		best_KNN_dist = KNN_dist
		
		scores.set_value(i,'KNN Route 1 Iteration',best_KNN_route)
		scores.set_value(i,'KNN Time % of Optimal (1 IT)',sum(KNN_times) / scores.ix[i]['Optimal Time sec']) 
		scores.set_value(i,'KNN Distance % of Optimal (1 IT)',best_KNN_dist / scores.ix[i]['Optimal Route Distance'])
		
		#9 more iterations run of KNN algorithm using same inputs
		for j in range(1,10):
			start = dt.datetime.now()
			KNN_route, KNN_dist = KNN_salesman(test_coords,test_names)
			stop = dt.datetime.now()
			
			delta = (stop - start).total_seconds()
			
			#Times for each iteration stored - used to calculate total run time at each milestone
			KNN_times.append(delta)
			
			if(KNN_dist < best_KNN_dist):
				best_KNN_dist = KNN_dist
				best_KNN_route = KNN_route
			
			#Performance recorded at 3, 5 and 10 iterations of (randomly initialized) algorithm
			if(j == 2):
				scores.set_value(i,'KNN Route 3 Iterations', best_KNN_route)
				scores.set_value(i,'KNN Time % of Optimal (3 IT)', sum(KNN_times) / scores.ix[i]['Optimal Time sec'])
				scores.set_value(i,'KNN Distance % of Optimal (3 IT)', best_KNN_dist / scores.ix[i]['Optimal Route Distance'])
			elif(j == 4):
				scores.set_value(i,'KNN Route 5 Iterations', best_KNN_route)
				scores.set_value(i,'KNN Time % of Optimal (5 IT)',sum(KNN_times) / scores.ix[i]['Optimal Time sec'])
				scores.set_value(i,'KNN Distance % of Optimal (5 IT)',best_KNN_dist / scores.ix[i]['Optimal Route Distance'])
			elif(j == 9):
				scores.set_value(i,'KNN Route 10 Iterations',best_KNN_route)
				scores.set_value(i,'KNN Time % of Optimal (10 IT)',sum(KNN_times) / scores.ix[i]['Optimal Time sec'])
				scores.set_value(i,'KNN Distance % of Optimal (10 IT)', best_KNN_dist / scores.ix[i]['Optimal Route Distance'])
	
	#output to .csv file - appended to file if it exists, otherwise new file created
	if(not os.path.isfile('SalesmanTestResults.csv')):
		scores.to_csv('SalesmanTestResults.csv', header = col_names)
	else:
		scores.to_csv('SalesmanTestResults.csv', mode = 'a', header = False)
	
	return