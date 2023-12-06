from queue import PriorityQueue
from collections import deque
import unittest

class Graph:
    """Graph data structures"""
    def __init__(self, vertices:list[int] = None, directed: bool = False) -> None:
        self.adjacency_list = dict()
        self.directed = directed
        if vertices is not None:
            for v in vertices:
                self.adjacency_list[v] = dict()

    def add_vertex(self, v: int):
        """Add a new vertex to the graph"""
        self.adjacency_list[v] = dict()

    def add_edge(self, u:int, v:int, w:int):
        """Add a new edge to the graph with weight w. u and v must be vertices of the graph. Self-loops are ignored. If the graph is undirected the edge (v,u) is also added."""
        if u not in self.adjacency_list or v not in self.adjacency_list:
            raise Exception("Vertices must exist!")
        if u == v: return
        self.adjacency_list[u][v] = w
        if not self.directed:
            self.adjacency_list[v][u] = w

    def set_directed(self, d: bool):
        """Change the graph to be directed if d is true, or undirected if d is false"""
        self.directed = d

    def is_directed(self):
        """Returns true if the graph is directed, false if undirected"""
        return self.directed
    
    def get_weight(self, u:int, v:int):
        """Returns the weight of an edge. u and v must be vertices of the graph. If the edge does not exist returns 'inf'. If u == v returns 0"""
        if u not in self.adjacency_list or v not in self.adjacency_list:
            raise Exception("Vertices must exist!")
        return self.adjacency_list[u].get(v, float('inf')) if u != v else 0
        
    def get_edges(self):
        """Returns a list of all edges in the graph as tuples '(u,v)' """
        return [(u, v) for v in self.adjacency_list for u in self.adjacency_list[v]]
    
    def get_edges_with_weight(self):
        """Returns a list of all edges in the graph as tupels '(u,v,w)' with w the weight of the edge"""
        return [(u, v, self.adjacency_list[v][u]) for v in self.adjacency_list for u in self.adjacency_list[v]]
    
    def get_vertices(self):
        """Returns the vertices of the graph as list"""
        return list(self.adjacency_list.keys())
    
    def get_successors(self, v:int):
        """Returns a list of vertices succeeding v. That is, all u such that (v, u) is an edge in the graph"""
        return [u for u in self.adjacency_list[v]]
    
    def get_out_degree(self, v:int):
        """Returns the out degree of vertex v. That is, the number of successors of v"""
        return len(self.adjacency_list[v])
    
    def get_in_degree(self, v:int):
        """Returns the in degree of vertex v. That is, the number of predecessors of v"""
        return len([1 for u in self.adjacency_list if v in self.adjacency_list[u]])
        
    def remove_edge(self, u:int, v:int):
        """Removes the edge (u,v) from the graph. If the graph is undirected (v,u) is also removed. Returns the weight of the removed edge. If the edge does not exist returns None."""
        if not self.directed:
            if v not in self.adjacency_list or u not in self.adjacency_list[v]:
                return None
            del self.adjacency_list[v][u]
        if u not in self.adjacency_list:
            return None
        return self.adjacency_list[u].pop(v)

    def remove_vertex(self, v:int):
        """Removes the vertex v from the graph"""
        del self.adjacency_list[v]
        for u in self.adjacency_list:
            self.adjacency_list[u].pop(v, None)
    
    def copy(self):
        """Creates a copy of the graph"""
        c = Graph()
        c.directed = self.directed
        c.adjacency_list = self.adjacency_list.copy()
        for v in c.adjacency_list:
            c.adjacency_list[v] = self.adjacency_list[v].copy()
        return c

    def __len__(self):
        """Returns the length of the graph. That is, the number of vertices"""
        return len(self.adjacency_list)
    
    def __str__(self) -> str:
        """Returns a string representation of the graphs adjacency_list"""
        return str(self.adjacency_list)


def djp(graph: Graph) -> Graph:
    """Implementation of the DJP-Algorithm. The input is an undirected, connected graph with weights. The algorithm returns a minimum spanning tree in form of a graph."""
    # TODO: Aufgabe 1b


def euler(graph: Graph) -> list[int]:
    """This algorithm gets a directed, connected graph as input and returns an Eulerian path in form of a list of vertices. If no Eulerian path exists, the algorithm returns None"""
    # TODO: Aufgabe 2b

def tsp(graph: Graph) -> list[int]:
    """This algorithm gets a metric graph as input and returns an approximation for the Travelling salesman problem in form of a list of vertices, by using minimum spanning trees and Eulerian paths."""
    # TODO: Aufgabe 2c

def merge(list0:list[int], list1:list[int]) -> list[int]:
    """Merge operation used in merge sort. Takes the two sorted lists list0 and list1 and returns a sorted list containing the elements of both input lists."""
    result = []
    while list0 and list1: # repeat until list0 or list1 is empty
        if list0[0] <= list1[0]: 
            result.append(list0.pop(0))
        else: 
            result.append(list1.pop(0))
            
    # put everything left in list0 OR list1 at the end
    result.extend(list0)
    result.extend(list1)
    return result

def run_decomposition(input_list: list[int]) -> list[list[int]]:
    """Separates a list of numbers into runs such that each run has length >= minrun and is sorted."""
    minrun = len(input_list)
    while minrun >= 64: 
        minrun = minrun // 2 + minrun % 2 # round up

    runs = []  # List to store runs

    #initialize values i and n as the beginning and the end of the list
    i = 0
    n = len(input_list)

    while i < n: #while there are still elements to add to a run
        
        start = i #starting index of sorted sublist
        end = start+1 #ending index of sorted sublist -> set to the 2nd position 
        
        #Find a run (sorted sublist)
        while end < n and input_list[end - 1] <= input_list[end]: # while were not at the end of the array and the sublist is sorted
            end += 1

        #if the (sublist)-run is smaller than the minrun it needs to be extended using insertionsort until it meets the minlength requirement
        while end < n and end - start < minrun:
            insertion_sort(input_list, start, end)
            end = end + 1
        
        # if the run is still smaller than minrun, merge it with the previous run
        if end - start < minrun and runs:
            runs[-1] = runs[-1] + input_list[start:end] #merge the run with the previous run
            insertion_sort(runs[-1], 0, len(runs[-1])-1) #sort the merged run
        else:
            runs.append(input_list[start:end]) #add the run to the runs list
        
        i = end #set the beginning of the next run
        
    return runs


def insertion_sort(input_list: list[int], start: int, end: int):
    for i in range(start+1, end+1): # starts at 2nd element and ends at the last element. i is the index of the element that should be inserted
        j = i #tmp variable that keeps track of the 
        #insert element into sorted list
        while j > start and input_list[j] < input_list[j-1] : #while the number is smaller than its predecessor...
            input_list[j], input_list[j-1] = input_list[j-1], input_list[j]#swap the elements 
            j = j-1 # decrement j to check the next lower element for a potential swap


def timsort(input_list: list[int]) -> list[int]:
    """Sorts the input list using Timsort."""
    stack = deque() #stack to store the input_list
    runs = run_decomposition(input_list)
    
    while runs or len(stack) >= 2: #while there are still runs to merge or more than 1 run on the stack (that need to be merged)
        if runs: 
            stack.append(runs.pop()) #add a new run to the stack
        
        #merges the last two runs on the stack
        while len(stack) >= 2:
            run2 = stack.pop()
            run1 = stack.pop()
            merge_run = merge(run1, run2)
            stack.append(merge_run)
    
    return stack[0] if stack else []

class TestExercise1(unittest.TestCase):
    def test_djp_tree(self):
        g = Graph([0, 1, 2, 3])
        g.add_edge(0, 1, 5)
        g.add_edge(1, 3, 3)
        g.add_edge(0, 2, 10)
        gs = g.copy()

        self.assertEqual(djp(g).adjacency_list, gs.adjacency_list)

    def test_djp_4(self):
        gs = Graph([0, 1, 2, 3])
        gs.add_edge(0, 2, 3)
        gs.add_edge(1, 2, 1)
        gs.add_edge(2, 3, 3)
        g = gs.copy()
        g.add_edge(0, 1, 5)
        g.add_edge(0, 3, 10)
        g.add_edge(1, 3, 5)

        self.assertEqual(djp(g).adjacency_list, gs.adjacency_list)

    def test_djp_10(self):
        gs = Graph(list(range(10)))
        gs.add_edge(0, 4, 2)
        gs.add_edge(1, 7, 3)
        gs.add_edge(2, 9, 1)
        gs.add_edge(3, 5, 3)
        gs.add_edge(7, 9, 5)
        gs.add_edge(7, 8, 12)
        gs.add_edge(0, 2, 6)
        gs.add_edge(0, 6, 9)
        gs.add_edge(5, 7, 3)
        g = gs.copy()
        g.add_edge(4, 5, 8)
        g.add_edge(3, 4, 6)
        g.add_edge(0, 3, 14)
        g.add_edge(3, 6, 15)
        g.add_edge(2, 3, 7)
        g.add_edge(2, 4, 9)
        g.add_edge(1, 3, 8)
        g.add_edge(1, 5, 7)
        g.add_edge(0, 8, 14)
        g.add_edge(8, 9, 13)

        self.assertEqual(djp(g).adjacency_list, gs.adjacency_list)

class TestExercise2(unittest.TestCase):
    def is_euler(self, g: Graph, p: list[int]):
        g = g.copy()
        for x, y in zip(p[:-1], p[1:]):
            if not g.remove_edge(x, y):
                return False
        return g.get_edges() == []

    def test_euler_isolated(self):
        g = Graph([1, 2], directed=True)

        self.assertIs(euler(g), None)

    def test_euler_in_unequal_out(self):
        g = Graph([1,2,3], directed=True)
        g.add_edge(1,2,1)
        g.add_edge(2,3,1)
        g.add_edge(3,1,1)
        g.add_edge(1,3,1)

        self.assertIs(euler(g), None)

    def test_euler5(self):
        g = Graph([1,2,3,4,5], directed=True)
        g.add_edge(1, 2, 1)
        g.add_edge(2, 3, 1)
        g.add_edge(3, 4, 1)
        g.add_edge(4, 5, 1)
        g.add_edge(2, 4, 1)
        g.add_edge(4, 1, 1)
        g.add_edge(5, 2, 1)

        self.assertTrue(self.is_euler(g, euler(g)))

    def test_euler10(self):
        g = Graph(list(range(10)))
        g.add_edge(0, 3, 14)
        g.add_edge(0, 4, 2)
        g.add_edge(0, 6, 9)
        g.add_edge(0, 8, 14)
        g.add_edge(1, 5, 7)
        g.add_edge(1, 7, 3)
        g.add_edge(2, 3, 7)
        g.add_edge(2, 4, 9)
        g.add_edge(3, 4, 6)
        g.add_edge(3, 6, 15)
        g.add_edge(4, 8, 7)
        g.add_edge(5, 7, 3)
        g.add_edge(7, 8, 12)
        g.add_edge(7, 9, 5)
        g.add_edge(8, 9, 13)
        g.directed = True

        self.assertTrue(self.is_euler(g, euler(g)))

    def test_tsp5(self):
        g = Graph([1,2,3,4,5])
        g.add_edge(1, 2, 5)
        g.add_edge(1, 3, 10)
        g.add_edge(1, 4, 12)
        g.add_edge(1, 5, 8)
        g.add_edge(2, 3, 15)
        g.add_edge(2, 4, 14)
        g.add_edge(2, 5, 12)
        g.add_edge(3, 4, 9)
        g.add_edge(3, 5, 12)
        g.add_edge(4, 5, 7)

        trip = tsp(g)
        self.assertEqual(trip[0], trip[-1]) # start = finish
        self.assertCountEqual(trip[:-1], [1,2,3,4,5]) # all vertices are visited


    def test_tsp10(self):
        g = Graph(list(range(10)))
        g.add_edge(0, 1, 8)
        g.add_edge(0, 2, 7)
        g.add_edge(0, 3, 6)
        g.add_edge(0, 4, 10)
        g.add_edge(0, 5, 10)
        g.add_edge(0, 6, 4)
        g.add_edge(0, 7, 8)
        g.add_edge(0, 8, 3)
        g.add_edge(0, 9, 3)
        g.add_edge(1, 2, 12)
        g.add_edge(1, 3, 12)
        g.add_edge(1, 4, 3)
        g.add_edge(1, 5, 9)
        g.add_edge(1, 6, 5)
        g.add_edge(1, 7, 13)
        g.add_edge(1, 8, 6)
        g.add_edge(1, 9, 6)
        g.add_edge(2, 3, 13)
        g.add_edge(2, 4, 11)
        g.add_edge(2, 5, 17)
        g.add_edge(2, 6, 7)
        g.add_edge(2, 7, 1)
        g.add_edge(2, 8, 6)
        g.add_edge(2, 9, 10)
        g.add_edge(3, 4, 14)
        g.add_edge(3, 5, 7)
        g.add_edge(3, 6, 9)
        g.add_edge(3, 7, 13)
        g.add_edge(3, 8, 9)
        g.add_edge(3, 9, 6)
        g.add_edge(4, 5, 12)
        g.add_edge(4, 6, 6)
        g.add_edge(4, 7, 12)
        g.add_edge(4, 8, 7)
        g.add_edge(4, 9, 8)
        g.add_edge(5, 6, 10)
        g.add_edge(5, 7, 18)
        g.add_edge(5, 8, 11)
        g.add_edge(5, 9, 7)
        g.add_edge(6, 7, 8)
        g.add_edge(6, 8, 1)
        g.add_edge(6, 9, 3)
        g.add_edge(7, 8, 7)
        g.add_edge(7, 9, 11)
        g.add_edge(8, 9, 4)

        trip = tsp(g)
        self.assertEqual(trip[0], trip[-1]) # start = finish
        self.assertCountEqual(trip[:-1], list(range(10))) # all vertices are visited

class TestExercise3(unittest.TestCase):
    def test_run_decom_reverse_100(self):
        l = list(range(100, 0, -1))
        l1 = list(range(51, 101))
        l2 = list(range(1, 51))
        self.assertIn(run_decomposition(l), [[l1, l2], [l2, l1]])

    def test_run_decom_streak(self):
        l = list(range(0, 80)) + list(range(70, 150))
        l1 = list(range(0, 80))
        l2 = list(range(70, 150))
        self.assertIn(run_decomposition(l), [[l1, l2], [l2, l1]])

    def test_run_decom_last_wrong(self):
        l = list(range(1, 100)) + [0]
        l1 = list(range(0, 100))
        self.assertEqual(run_decomposition(l), [l1])

    def test_run_decom_small(self):
        l = list(range(1, 10, 2)) + list(range(0, 10, 2))
        l1 = list(range(0, 10))
        self.assertEqual(run_decomposition(l), [l1])

    def test_run_decom_127(self):
        l = list(range(0, 40)) + list(range(100, 39, -2)) + list(range(41, 101, 2)) + list(range(101, 127))
        l1 = list(range(0, 40)) + [100]
        l2 = list(range(40, 45)) + list(range(46, 99, 2))
        l3 = list(range(45, 102, 2)) + list(range(102, 127))
        self.assertIn(run_decomposition(l), [[l1, l2, l3], [l1, l3, l2], [l2, l1, l2], [l2, l3, l1], [l3, l1, l2], [l3, l2, l1]])

    def test_timsort_127(self):
        l = list(range(0, 40)) + list(range(100, 39, -2)) + list(range(41, 101, 2)) + list(range(101, 127))
        ls = list(range(0,127))
        self.assertEqual(timsort(l), ls)

    def test_timsort_rev_1000(self):
        l = list(range(1000, 0, -1)) 
        ls = list(range(1, 1001))
        self.assertEqual(timsort(l), ls)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    print("---------------- Test Exercise 1  ----------------")
    runner.run(unittest.TestLoader().loadTestsFromTestCase(TestExercise1))
    print("---------------- Test Exercise 2  ----------------")
    runner.run(unittest.TestLoader().loadTestsFromTestCase(TestExercise2))
    print("---------------- Test Exercise 3  ----------------")
    runner.run(unittest.TestLoader().loadTestsFromTestCase(TestExercise3))