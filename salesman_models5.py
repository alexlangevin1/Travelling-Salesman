#File Name: salesman_models5.py
#Date Created: 3 July 2017
#Last Updated: 13 August 2017
#Author: Alex Langevin
#Description: Three algorithms to approach solution to travelling salesman problem along with support function
#			  First algorithm brute forces the optimal route
#			  Second algorithm returns best route after a user-specified number of random route iterations
#			  Third algorithm adapts a K-Nearest Neighbor approach to return the (approx.) optimal route
#Note: First two algorithms used mainly for performance testing purposes

#Special Thanks To:
#Ahmed Ismail for working through my ideas good and bad, and asking the right questions
#Gordon Bailey for reviewing the finished algorithm, helping clean up the code and suggesting performance improvements

import pandas as pd
import numpy as np
import math
import itertools as it
from sortedcontainers import SortedList

#Reads in .csv with GPS coordinates for 100 largest US cities, and returns numpy arrays of city names and coords
def get_clean_coords():
	data = pd.read_csv('GPS_Coordinates.csv')
	city_coords = data.ix[:,['Latitude','Longitude']].as_matrix()
	city_name = data.ix[:,['City']].as_matrix()
	
	return city_coords, city_name
	
#Reads in city route order (list of ints) and city distance matrix, returns total trip distance (km)
def trace_path(route,distance_matrix):
	distance = 0
	
	for trip in range(len(route)-1):
		distance += distance_matrix[route[trip]][route[trip+1]]

	return distance

def haversine(angle_rads):
	return (1-math.cos(angle_rads))/2

#Takes in numpy array of coordinates (lat,long) and returns a matrix of city-city km distances
def get_distance_matrix(city_coordinates):
	distance = np.zeros((city_coordinates.shape[0],city_coordinates.shape[0]))
	earth_radius_km = 6371
	
	for row in range(distance.shape[0]):
		for col in range(distance.shape[0]):
			lat1 = math.radians(city_coordinates[row,0])
			lon1 = math.radians(city_coordinates[row,1])
			lat2 = math.radians(city_coordinates[col,0])
			lon2 = math.radians(city_coordinates[col,1])
			delta_lat = lat2 - lat1
			delta_lon = lon2 - lon1
			
			#Converts latitude and longitude to km distance (straight line) using haversine formula
			hav = haversine(delta_lat) + math.cos(lat1)*math.cos(lat2)*haversine(delta_lon)
			distance[row][col] = 2 * earth_radius_km * math.asin(math.sqrt(hav))
			
	return distance
	
#Using numpy arrays of city names and coords, iterates through every possible route permutation
#Returns optimal route and total distance of optimal route
def brute_force_optimal_route(city_coordinates, city_list):
	distance = get_distance_matrix(city_coordinates)
	num_cities = city_coordinates.shape[0]
	
	#Initializes route and distance variables - first city in list assumed the starting (and ending) position
	best_route = [0] + list(range(1,num_cities)) + [0]
	shortest_dist = trace_path(best_route,distance)
	
	for perm in it.permutations(range(1,num_cities)):
		route = [0] + list(perm) + [0]
		dist = trace_path(route, distance)
		
		if (dist < shortest_dist):
			best_route = list(route)
			shortest_dist = dist
	
	city_route = []
	
	for city in best_route:
		city_route.append(city_list[city][0])
		
	return city_route, shortest_dist

#Takes city list and coordinates as inputs, returning best route found after a user-specified number of random iterations	
def random_route(city_coordinates, city_list, iterations = 1):
	iteration_count = 0
	distance = get_distance_matrix(city_coordinates)
	num_cities = city_coordinates.shape[0]
	
	perm_list = list(range(1,num_cities))
	perm = np.random.permutation(perm_list)
	
	best_route = [0] + list(perm) + [0]
	shortest_dist = trace_path(best_route,distance)
	
	iteration_count += 1
	
	while(iteration_count < iterations):
		perm = np.random.permutation(perm)
		route = [0] + list(perm) + [0]
		dist = trace_path(route,distance)
		iteration_count += 1
		
		if (dist < shortest_dist):
			best_route = list(route)
			shortest_dist = dist
	
	city_route = []
	
	for city in best_route:
		city_route.append(city_list[city][0])
	
	return city_route, shortest_dist
	
#Taking a city list and coordinates as inputs, connects vertices at random under a K-NN type rule until each vertex has degree 2
#Graph is then cleaned up to form a Hamiltonian circuit, which approximates (possibly equals) and returns optimal route
#Assumed first city in list is the starting (and ending) vertex
def KNN_salesman(city_coordinates, city_list):

	dist_matrix = get_distance_matrix(city_coordinates)
	
	#adjacency matrix - to be populated with 1s representing edges between vertices (i.e. connections between cities)
	adj_matrix = np.zeros((dist_matrix.shape[0],dist_matrix.shape[1]))
	
	num_cities = dist_matrix.shape[0]
	
	while(adj_matrix.sum() != 2 * adj_matrix.shape[0]):
	
		#Order of K-NN step initialized at random
		city_index = list(np.random.permutation(range(num_cities)))
		temp_dist_mat = np.copy(dist_matrix)
		
		#Removes zero values to facilitate nearest-neighbour loop below
		for same_city in range(temp_dist_mat.shape[0]):
			temp_dist_mat[same_city,same_city] = np.nan
		
		#TODO: Suppress warnings for all nan matrix as it is appropriately handled
		while(np.isnan(temp_dist_mat).sum() < temp_dist_mat.shape[0]**2):
			for city in city_index:
				#Return column index for first instance of min distance connection for selected city
				#i.e. nearest valid neighbour
				try:
					min_col = np.where(temp_dist_mat[city,:] == np.nanmin(temp_dist_mat[city,:]))[0][0]
				except:
					continue
				#As long as both cities have less than 2 edges, connect cities
				if(adj_matrix[city,:].sum() < 2 and adj_matrix[min_col,:].sum() < 2):
					create_point(adj_matrix,(city,min_col))
				
				#Proceed to next nearest neighbour
				temp_dist_mat[city,min_col] = np.nan
				temp_dist_mat[min_col,city] = np.nan
		
			#all vertices have degree 2
			if(adj_matrix.sum() == 2 * dist_matrix.shape[0]):
				break
		
		#if all options have been exhausted without each vertex obtaining degree 2, then start again with a new ordering
		if(np.isnan(temp_dist_mat).sum() == temp_dist_mat.shape[0]**2 and adj_matrix.sum() < 2 * adj_matrix.shape[0]):
			adj_matrix = np.zeros((dist_matrix.shape[0],dist_matrix.shape[1]))
			continue
	
		print("finished KNN step")
		
		#If the graph is disconnected after the K-NN step, the cycles are stitched together and edges pared
		if(mult_clusters(adj_matrix)):
			print("Stitching")
			adj_matrix = stitch(adj_matrix,dist_matrix,city_coordinates)
			print("done stitching and deletions")
	
	print("uncrossing")
	
	#Any crossing points in the planar graph are removed to form a Hamiltonian circuit
	adj_matrix = uncross(city_coordinates,adj_matrix)
	
	print("done uncrossing")
	print(adj_matrix)
	
	#Check that each vertex is of degree two and graph is connected
	assert(adj_matrix.sum() == 2 * adj_matrix.shape[0])
	assert(not mult_clusters(adj_matrix))
	
	next_pos = 0
	route = [0]
	
	#Adjacency matrix is traversed to record the final city route
	while(adj_matrix[0,:].sum() > 0):
		prev_pos = next_pos
		next_pos = np.where(adj_matrix[next_pos,:] == adj_matrix[next_pos,:].max())[0][0]
		route.append(next_pos)
		erase_point(adj_matrix,(prev_pos,next_pos))
		
	city_route = []
	
	for city in route:
		city_route.append(city_list[city][0])
	
	return city_route, trace_path(route,dist_matrix)

#Boolean support function returns True if graph is disconnected	
#i.e. adjacency matrix contains multiple connected subgraphs
def mult_clusters(adj_matrix, start_pos=0):
	temp_mat = np.copy(adj_matrix)
	cluster = set([start_pos])
	next_pos = np.where(temp_mat[start_pos,:] == temp_mat[start_pos,:].max())[0][0]
	cluster.add(next_pos)
	erase_point(temp_mat,(start_pos,next_pos))
	prev_pos = next_pos
	
	while(temp_mat[start_pos,:].sum() > 0):
		next_pos = np.where(temp_mat[next_pos,:] == temp_mat[next_pos,:].max())[0][0]
		cluster.add(next_pos)
		erase_point(temp_mat,(prev_pos,next_pos))
		prev_pos = next_pos

	return (len(cluster) < adj_matrix.shape[0])

#Boolean support function returns True if previously disconnected graph is now connected
#Examines whether edges exist between what were originally connected subgraphs
def is_connected_graph(cluster_mat,start_pos):
	if(cluster_mat.sum() == 0):
		connected = False
	else:
		temp_mat = np.copy(cluster_mat)
		next_pos = np.where(temp_mat[start_pos,:] == temp_mat[start_pos,:].max())[0][0]
		erase_point(temp_mat,(start_pos,next_pos))
		prev_pos = next_pos
		
		while(temp_mat[next_pos,:].sum() > 0):
			next_pos = np.where(temp_mat[next_pos,:] == temp_mat[next_pos,:].max())[0][0]
			erase_point(temp_mat,(prev_pos,next_pos))
			prev_pos = next_pos
		
		if(next_pos == start_pos):
			connected = True
		else:
			connected = False
	return connected
	
#Support function forms edges between disconnected subgraphs before swapping and paring edges as needed to form
#connected graph where all vertices are of degree 2
def stitch(adj_matrix,dist_matrix,city_coordinates):
	
	#step 1 - determine number of clusters (connected subgraphs)
	cluster_list = []
	cities_remaining = set(range(adj_matrix.shape[0]))
	temp_mat = np.copy(adj_matrix)
	
	while(len(cities_remaining) > 0):
		start_pos = cities_remaining.pop()
		cluster = set([start_pos])
		next_pos = np.where(temp_mat[start_pos,:] == temp_mat[start_pos,:].max())[0][0]
		cluster.add(next_pos)
		erase_point(temp_mat,(start_pos,next_pos))
		prev_pos = next_pos
		
		while(temp_mat[start_pos,:].sum() > 0):
			next_pos = np.where(temp_mat[next_pos,:] == temp_mat[next_pos,:].max())[0][0]
			cluster.add(next_pos)
			erase_point(temp_mat,(prev_pos,next_pos))
			prev_pos = next_pos
		
		cities_remaining = cities_remaining - cluster
		cluster_list.append(cluster)
		
	num_clusters = len(cluster_list)
	
	#step 2 - stitch clusters according to preset selection criteria
	#1. Only 1 edge between any two given clusters
	#2. new edge cannot create a closed loop before all clusters have 2 inter-cluster edges
	#3. No vertex can have a degree greater than 3
	#Note: 2 clusters is the exception, there must be 2 connections between these clusters
	
	cluster_adj_matrix = np.zeros((num_clusters,num_clusters))
	
	#boolean matrix tracks new edges formed in stitching process - these edges cannot later be deleted
	deletable_matrix = np.ones((adj_matrix.shape[0],adj_matrix.shape[0]),dtype=bool)
	cluster_count = 0
	cluster_dict = {}
	
	#dictionary mapping each vertex to its corresponding cluster
	for cluster in cluster_list:
		cluster_dict.update(dict.fromkeys(cluster,cluster_count))
		cluster_count += 1
		
	edge_info = []
	added_edges = []
	
	new_edges = 0
	
	if(num_clusters > 2):
		#create SortedLists (1 for each cluster) containing details of each possible inter-cluster edge (in ascending order by distance)
		#each SortedList element has the format (distance,(start city label, end city label),(start city cluster, end city cluster))
		for cluster in cluster_list:
			edges = SortedList()
			for vertex in cluster:
				for destination in range(vertex+1,adj_matrix.shape[1]):
					if(cluster_dict[vertex] != cluster_dict[destination]):
						edge = (dist_matrix[vertex,destination],(vertex,destination),(cluster_dict[vertex],cluster_dict[destination]))
						edges.add(edge)
			edge_info.append(edges)
	
		while(new_edges < num_clusters):
			#Current rule is choose the cluster that has the furthest min distance inter-cluster edge (i.e. maximin)
			#TODO: may need amending depending how rule functions - i.e. switch to simple min distance
			long_edge = (0,0)
			for cluster in edge_info:
				try:
					if(cluster[0][0] > long_edge[0]):
						long_edge = cluster[0]
				except:
					continue
					
			print(long_edge)
			
			if((cluster_adj_matrix[long_edge[2][0],:].sum() < 2 and cluster_adj_matrix[long_edge[2][1],:].sum() < 2) and (cluster_adj_matrix[long_edge[2][0],long_edge[2][1]] == 0)):
				if((adj_matrix[long_edge[1][0],long_edge[1][1]] == 0) and (adj_matrix[long_edge[1][0],:].sum() < 3 and adj_matrix[long_edge[1][1],:].sum() < 3)):
					create_point(adj_matrix,(long_edge[1][0],long_edge[1][1]))
					erase_point(deletable_matrix,(long_edge[1][0],long_edge[1][1]))
					create_point(cluster_adj_matrix,(long_edge[2][0],long_edge[2][1]))
					added_edges.append(long_edge)
				
					if(is_connected_graph(cluster_adj_matrix,long_edge[2][0]) and cluster_adj_matrix.sum() < 2*cluster_adj_matrix.shape[0]):
						erase_point(adj_matrix,(long_edge[1][0],long_edge[1][1]))
						create_point(deletable_matrix,(long_edge[1][0],long_edge[1][1]))
						erase_point(cluster_adj_matrix,(long_edge[2][0],long_edge[2][1]))
						added_edges.remove(long_edge)
					else:
						new_edges += 1
			edge_info[long_edge[2][0]].remove(edge_info[long_edge[2][0]][0])
	#two cluster case
	else:
		edges = SortedList()
		for row in range(adj_matrix.shape[0]):
			for col in range(row+1,adj_matrix.shape[0]):
				if(cluster_dict[row] != cluster_dict[col]):
					edge = (dist_matrix[row,col],(row,col),(cluster_dict[row],cluster_dict[col]))
					edges.add(edge)
			
		while(new_edges < num_clusters):
			min_edge = edges[0]
			print(min_edge)
			if(adj_matrix[min_edge[1][0],:].sum() < 3 and adj_matrix[min_edge[1][1],:].sum() < 3):
				create_point(adj_matrix,(min_edge[1][0],min_edge[1][1]))
				erase_point(deletable_matrix,(min_edge[1][0],min_edge[1][1]))
				create_point(cluster_adj_matrix,(min_edge[2][0],min_edge[2][1]))
				added_edges.append(min_edge)
				new_edges += 1
			edges.remove(min_edge)	
	
	print("done stitching")

	#step 3 - Pare edges as needed to ensure all vertices have degree 2
	
	row_sum_mat = np.ones((adj_matrix.shape[0],1))
	temp_cluster_list = []
	
	for cluster in cluster_list:
		temp_cluster_list.append(cluster.copy())
	
	deletable_edges = adj_matrix * deletable_matrix
	
	while(adj_matrix.dot(row_sum_mat).max() > 2):
		
		#Delete any edges where both vertices are of degree 3
		adj_matrix = delete_edges(adj_matrix,deletable_edges)
		
		#If any degree 3 vertices remain, focus on offending clusters for paring opportunities
		if(adj_matrix.dot(row_sum_mat).max() > 2):
			print("need additional swaps")
			#In each cluster, compile list of vertices of degree 3, along with their (within-cluster) edges
			for cluster in temp_cluster_list:
				three_edge_cities = []
				if(count_edges(cluster,adj_matrix) == 2 * len(cluster)):
					continue
				for vertex in cluster:
					if(adj_matrix[vertex,:].sum() > 2):
						connected_cities = set()
						for i in range(deletable_matrix.shape[0]):
							if(deletable_edges[vertex,i] == 1):
								connected_cities.add(i)
						three_edge_cities.append((vertex,tuple(connected_cities)))
						
				min_dist_swap = three_edge_cities[0]
				edge_to_add = min_dist_swap[1]
				edge_to_delete = (min_dist_swap[0],min_dist_swap[1][0])
				make_non_deletable = (min_dist_swap[0],min_dist_swap[1][1])
				
				#For a given cluster, examine each degree vertex under the following rules:
				#1. Form a hypothetical new edge between the two vertices adjacent to the degree 3 vertex (i.e. form a triangle)
				#2. Calculate the total change in distance from deleting either of the two existing edges
				#3. For the cluster as a whole select the edge swap that results in the minimum distance change for the graph
				for triangle in three_edge_cities:
					print(triangle)
					if(dist_matrix[triangle[1][0],triangle[1][1]] - dist_matrix[triangle[0],triangle[1][0]] < dist_matrix[edge_to_add[0],edge_to_add[1]] - dist_matrix[edge_to_delete[0],edge_to_delete[1]]):
						min_dist_swap = triangle
						edge_to_add = min_dist_swap[1]
						edge_to_delete = (min_dist_swap[0],min_dist_swap[1][0])
						make_non_deletable = (min_dist_swap[0],min_dist_swap[1][1])
					if(dist_matrix[triangle[1][0],triangle[1][1]] - dist_matrix[triangle[0],triangle[1][1]] < dist_matrix[edge_to_add[0],edge_to_add[1]] - dist_matrix[edge_to_delete[0],edge_to_delete[1]]):
						min_dist_swap = triangle
						edge_to_add = min_dist_swap[1]
						edge_to_delete = (min_dist_swap[0],min_dist_swap[1][1])
						make_non_deletable = (min_dist_swap[0],min_dist_swap[1][0])
						
				#Result of edge swap is that one outer(connecting) vertex has effectively been removed from the cluster
				#The degree 3 vertex has been shifted into the smaller cluster, where the first deletion rule can be re-evaluated
				#Continue the cluster shrinking process until all required edges have been pared
				create_point(adj_matrix,edge_to_add)
				erase_point(adj_matrix,edge_to_delete)
				create_point(deletable_edges,edge_to_add)
				erase_point(deletable_edges,edge_to_delete)
				erase_point(deletable_edges,make_non_deletable)
				cluster.remove(min_dist_swap[0])
					
	return adj_matrix
	
#Support function removes any crossing points from a connected graph, returning a Hamiltonian circuit
def uncross(city_coordinates,adj_matrix):
	
	#Keep swapping edges until all crossing points have been removed
	while(True):	
		pairs = list_of_points(adj_matrix)
		swap_count = 0
		
		for pair in pairs:
			#Exclude any edges that share a vertex from the swapping rule
			if((pair[0][0] in pair[1]) or (pair[0][1] in pair[1])):
				continue
			point1 = city_coordinates[pair[0][0],:]
			point2 = city_coordinates[pair[0][1],:]
			point3 = city_coordinates[pair[1][0],:]
			point4 = city_coordinates[pair[1][1],:]
			if(is_line_intersect(point1,point2,point3,point4)):
				print("Pair needs swapping:",pair)
				adj_matrix = swap(adj_matrix,pair)
				swap_count += 1
				break
		#Exit loop once every pair of edges has been evaluated without finding any crossing points
		if(swap_count == 0):
			break
			
	return adj_matrix

#Support function returns slope and intercept of a line	
def get_line_formula(x1,y1,x2,y2):
	m = (y2 - y1) / (x2 - x1)
	b = y1 - m*x1
	return m,b

#Support function determines whether two edges intersect	
def is_line_intersect(point1,point2,point3,point4):
	m1,b1 = get_line_formula(point1[0],point1[1],point2[0],point2[1])
	m2,b2 = get_line_formula(point3[0],point3[1],point4[0],point4[1])
		
	points = np.vstack((point1,point2,point3,point4))
		
	x1_max = np.max(points[:2,0])
	x1_min = np.min(points[:2,0])
	y1_max = np.max(points[:2,1])
	y1_min = np.min(points[:2,1])
		
	x2_max = np.max(points[2:,0])
	x2_min = np.min(points[2:,0])
	y2_max = np.max(points[2:,1])
	y2_min = np.min(points[2:,1])
		
	X = np.array([[m1],[m2]])
	Y = np.ones((X.shape[0],X.shape[1])) * -1
	lines = np.concatenate((X,Y),axis=1)
	b = np.array([[b1],[b2]]) * -1
		
	intersect = True
	
	try:
		solution = np.linalg.solve(lines,b)
		
		#Check whether edge intersect solution falls within the vertex coordinates
		if(not(x1_min <= solution[0,0] <= x1_max and x2_min <= solution[0,0] <= x2_max)):
			intersect = False
			
		if(not(y1_min <= solution[1,0] <= y1_max and y2_min <= solution[1,0] <= y2_max)):
			intersect = False
	except:
		intersect = False
		
	return intersect

#Support function removes crossing points by switching edges between two pairs of vertices	
def swap(adj_matrix,line_pair):
	
	x1 = line_pair[0][0]
	y1 = line_pair[0][1]
	x2 = line_pair[1][0]
	y2 = line_pair[1][1]
	
	erase_point(adj_matrix,(x1,y1))
	erase_point(adj_matrix,(x2,y2))
	
	#Swap edges in whichever order keeps the graph connected
	if(adj_matrix[x1,x2] == 0 and adj_matrix[y1,y2] == 0):
		create_point(adj_matrix,(x1,x2))
		create_point(adj_matrix,(y1,y2))
		if(mult_clusters(adj_matrix)):
			erase_point(adj_matrix,(x1,x2))
			erase_point(adj_matrix,(y1,y2))
	
	if(adj_matrix.sum() < 2 * adj_matrix.shape[0]):
		create_point(adj_matrix,(x1,y2))
		create_point(adj_matrix,(x2,y1))
		
	return adj_matrix
	
#Support function evaluates a given adjacency matrix and returns a list of all possible edge pairs
def list_of_points(adj_matrix):
	temp_mat = np.copy(adj_matrix)
	points = []
		
	for row in range(adj_matrix.shape[0]):
		while(temp_mat[row,:].max() > 0):
			col_index = np.where(temp_mat[row,:] == temp_mat[row,:].max())[0][0]
			points.append((row,col_index))
			temp_mat[row,col_index] = 0
			try:
				temp_mat[col_index,row] = 0
			except:
				continue
				
	pairs = list(it.combinations(points,2))
	return pairs
	
#Support function to remove edge from an adjacency matrix
def erase_point(adj_matrix,point):
	adj_matrix[point[0],point[1]] = 0
	adj_matrix[point[1],point[0]] = 0
	return

#Support function to add edge to an adjacency matrix	
def create_point(adj_matrix,point):
	adj_matrix[point[0],point[1]] = 1
	adj_matrix[point[1],point[0]] = 1
	return
	
def test_mat():
	t_mat = np.array([[0,1,0,0,0,0,1],[1,0,0,0,1,0,0],[0,0,0,0,1,1,0],[0,0,0,0,0,1,1],[0,1,1,0,0,0,0],[0,0,1,1,0,0,0],[1,0,0,1,0,0,0]])
	return t_mat
	
#Support function that removes any edges between two vertices of degree 3
def delete_edges(adj_matrix, deletable_edges):

	row_index = 0
	to_delete = []
	
	for row in deletable_edges:
		col_index = 0
		for cell in row:
			if((cell == 1) and (adj_matrix[row_index,:].sum() > 2) and (adj_matrix[col_index,:].sum() > 2)):
				to_delete.append((row_index,col_index))
				erase_point(deletable_edges,(row_index,col_index))
			col_index += 1
		row_index += 1
	
	print("3-edge connections to delete:",to_delete)
	for edge in to_delete:
		erase_point(adj_matrix,(edge[0],edge[1]))
		
	return adj_matrix
	
def count_edges(some_cluster, adj_matrix):
	edge_count = 0
	for vertex in some_cluster:
		edge_count += adj_matrix[vertex,:].sum()
		
	return edge_count
		