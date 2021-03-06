#import random
#import matplotlib.pyplot as plt
from typing import Dict, Tuple
from itertools import cycle
from collections import defaultdict
import pstats
#import cProfile
#import random as rd
import queue
import time
#from config import filename
import bisect
from copy import deepcopy
import json
#import numpy as np
import math


filename = "./smallRandom.json"

for x in [6,7,8,9] :
    n_cores = x

    def n2letter(n):
        '''0 to 'a', 1 to 'b', ... '''
        return chr(96+n)


    def string2duration(string):
        ''' "01:50:19.3177493" to duration in seconds'''
        return 3600*int(string[:2]) + 60*int(string[3:5]) + int(string[6:8])  # Duration is int


    def read_data(path):
        global task_count
        global tasks
        file = open(path)
        data = json.load(file)
        nodes = data['nodes']
        tasks = dict()
        for task_str, info in nodes.items():
            task = int(task_str)
            tasks[task] = {'Data': string2duration(
                info['Data']), 'Dependencies': info['Dependencies']}
        task_count = len(tasks)
        print("Data loaded successfully. Number of tasks: " + str(task_count))



    read_data(filename)
    tasks

    # Tasks to child tasks / Tasks to parents / Task is terminal / Task is inital
    task2childs = {task: list() for task in tasks}
    task2parents = {task: list() for task in tasks}
    for task, info in tasks.items():
        # Add childs
        list_task_parents = info['Dependencies']
        for task_parent in list_task_parents:
            task2childs[task_parent].append(task)
        # Add parents
        task2parents[task] = tasks[task]['Dependencies']


    def task_is_terminal(task: int):
        return len(task2childs[task]) == 0


    def task_is_inital(task: int):
        return len(task2parents[task]) == 0


    print(task2childs)
    print(task2parents)

    task2sbl = {}


    def save_static_bottom_level(task: int):
        task_duration = tasks[task]["Data"]
        if task_is_terminal(task):
            sbl = task_duration
        else:
            list_sbl_child = list()
            for task_child in task2childs[task]:
                if task_child in task2sbl:
                    sbl_child = task2sbl[task_child]
                else:
                    sbl_child = save_static_bottom_level(task_child)
                list_sbl_child.append(sbl_child)
            sbl = max(list_sbl_child) + task_duration

        task2sbl[task] = sbl
        return sbl


    for task in tasks:
        if task_is_inital(task):
            save_static_bottom_level(task)

    task2sbl


    class Graph:

        def __init__(self):
            self.tasks = tasks
            self.tasks_to_sbl = task2sbl
            self.tasks_to_parent = task2parents
            self.tasks_to_child = task2childs
            self.n_cores = n_cores
            self.nodes = list()


    graph = Graph()


    class Node():
        graph = graph

        def __init__(self, parent=None, task_to_add=None, core_where_to_add=None, time_task_start=None):
            '''Create a Node object ie a partial scheduling
            parent = parent Node, None if root
            task_to_add : task added to the partial schedule
            core_where_to_add : core where to do task
            time_task_start : instant where the core will start computing the task
            '''
            if parent is None:
                self.parent = None
                self.tasks_done_time = dict()
                self.cores = {core_n: {"task": -1, "task_end_time": 0}
                              for core_n in range(n_cores)}

                self.g = 0
                self.f = self.h()

                self.hist = ''
                self.schedule = dict()

            else:
                task_end_time = time_task_start + \
                    self.graph.tasks[task_to_add]['Data']

                self.parent = parent
                self.tasks_done_time = parent.tasks_done_time.copy()
                self.tasks_done_time[task_to_add] = task_end_time

                self.cores = parent.cores.copy()
                self.cores[core_where_to_add] = {
                    "task": task_to_add, "task_end_time": task_end_time}

                self.g = self.compute_g()
                self.f = max(self.g + self.h(), parent.f)

                self.schedule = parent.schedule.copy()
                self.schedule[task_to_add] = (
                    time_task_start, task_end_time, core_where_to_add)
                self.hist = parent.hist + \
                    f"|Task {task_to_add} start at time {time_task_start} on core {core_where_to_add} "

        def __repr__(self):
            string = '[' + ','.join([n2letter(task)
                                     for task in self.tasks_done_time]) + ']'
            string += ''.join(
                [f"({core['task']} end at {core['task_end_time']})" for core in self.cores.values()])
            return string

        def is_goal(self):
            '''Return whether a node is a full schedule'''
            return len(self.tasks_done_time) == task_count

        def successors(self):
            '''Create and return list of child node of self'''
            childs = list()

            # On regarde toutes les t??ches qu'on va tenter de rajouter
            for task, info in self.graph.tasks.items():

                # On passe les taches d??j?? ajout??es
                if task in self.tasks_done_time:
                    continue

                # On ne garde que les taches dont toutes les d??pendances ont ??t?? r??alis??es
                if not all([task_required in self.tasks_done_time for task_required in info['Dependencies']]):
                    continue

                # On calcul le temps ou toutes les d??pendances de task seront termin??s par les coeurs
                time_all_tasks_done = max(
                    [0] + [self.tasks_done_time[task_required] for task_required in info['Dependencies']])

                for core_n, core in self.cores.items():
                    # On ne commence ?? faire la task que lorsque toutes les d??pendances sont calcul??es et que le core est disponible.
                    time_core_available = core["task_end_time"]
                    time_task_start = max(time_all_tasks_done, time_core_available)

                    child = Node(parent=self, task_to_add=task,
                                 core_where_to_add=core_n, time_task_start=time_task_start)
                    childs.append(child)

            return sorted(childs, key=lambda node: node.f)

        def cost(self, child_node):
            '''Return the cost of going from self to child_node, a child node of self
            '''
            res = child_node.g - self.g
            if res < 0:
                raise Exception("Cost difference is negative")
            return res

        def h(self):
            '''Estimated remaining time of the node-schedule for reaching a terminal node. Must understimate true value.
            '''
            successor_tasks = list()
            for task, info in self.graph.tasks.items():
                if task in self.tasks_done_time:  # On passe les taches d??j?? ajout??es
                    continue
                # On ne garde que les taches dont toutes les d??pendances ont ??t?? r??alis??es
                if not all([task_required in self.tasks_done_time for task_required in info['Dependencies']]):
                    continue
                successor_tasks.append(task)
            if len(successor_tasks) == 0:
                return 0
            return max([self.graph.tasks_to_sbl[task] for task in successor_tasks])

        # Node-schedule method

        def __lt__(self, node):
            return self.f < node.f

        def __hash__(self):
            return int(self.f)

        def __eq__(self, node):
            '''Return whether a node is equal to another. Two nodes are considered equal if they have completed the same tasks and if all their cores stop working at same time.
            '''
            if self.g != node.g:
                return False
            if self.tasks_done_time != node.tasks_done_time:
                return False
            # if set(self.cores.values()) != set(node.cores.values()):
            #     return False
            return self.set_of_core() == node.set_of_core()

        def set_of_core(self):
            return set([core["task_end_time"] for core in self.cores.values()])

        def compute_g(self):
            return max([core["task_end_time"] for core in self.cores.values()])

    # r = Node()
    # x,y = r.successors()
    # a,b,c,d,e,f, *_ = x.successors()[0].successors()
    # g, h, i, j, k, l, *_ = y.successors()[0].successors()

    # #Test successors
    # print(a, b, c, d, e, f, sep = '\n')
    # print()
    # print(g, h, i, j, k, l, sep = '\n')
    # print()

    # #Test __eq__
    # for i in (g, h, i, j, k, l):
    #     for j in (a, b, c, d, e, f):
    #         if i == j:
    #             print(i, j)

    # #Test cost
    # x.cost(a)


    # #Test h
    # print(r.h())
    # print(x.h())


    class A_star():
        def __init__(self, root, graph):
            self.root = root
            self.graph = graph

        def find_best_path(self, max_time=float('inf')):
            t0 = time.time()

            OPEN_QUEUE = queue.PriorityQueue()
            # Open queue, pile of most urgent node to be evaluated
            OPEN_QUEUE.put(self.root)
            # Open set, more efficient way to compute if a node is in the open list
            OPEN_DICT = {self.root: None}
            CLOSED_DICT = dict()  # Closed list, list of already explored node

            while not OPEN_QUEUE.empty():
                if time.time() - t0 > max_time:
                    print('Time out')
                    return

                current = OPEN_QUEUE.get()
                del OPEN_DICT[current]
                CLOSED_DICT[current] = None
                if current.is_goal():  # If we reach a final node, it is the optimal solution and we return it
                    return current

                for child in current.successors():

                    if child in CLOSED_DICT:  # We pass the node already visited
                        continue

                    # We pass the node already waiting to be visited (note: in A* general, we need to recompute child.g and child.parent here, but for us child.g only depends on child (not on parent))
                    if child in OPEN_DICT:
                        continue

                    else:
                        # child.g = current.g + current.cost(child)     #General method. In our case, node.g does not depend on the path and the parent node
                        OPEN_QUEUE.put(child)
                        OPEN_DICT[child] = None

            raise Exception("No path from root to a terminal node")


    # A_star(root = Node(), graph=graph).find_best_path()

    root = Node()

    h=1
    tps_moyen = 0

    for i in range(h):

        start = time.time()
        a_star = A_star(root=root, graph=graph)

        final_node = a_star.find_best_path(max_time=3600)
        end = time.time()
        tps_moyen += (end-start)

    tps_moyen = tps_moyen/h
    print(tps_moyen)

    print(final_node)


    schedule = final_node.schedule


    def cycle(lst) -> str:
        x = lst.pop(0)
        lst.append(x)
        return x




    print(schedule)





