import heapq
import timeit
from collections import defaultdict
from utility.exceptions import *
class PriorityQueue:
    heap:list
    map:dict
    count=int
    nelements=int
    def __init__(self):
        self.heap=list()
        self.map=dict()
        self.count=0
        self.nelements=0
    def insert(self,key,priority):
        if key in self.map:
            existing=self.map[key]
            if not existing[2] and existing[0]<=priority:
                return existing
            else:
                existing[2]=True
                self.nelements-=1
        element=[priority,self.count,False,key]
        heapq.heappush(self.heap,element)
        self.map[key]=element
        self.nelements+=1
        self.count+=1
        return element
    def pop(self):
        done=len(self)==0
        while not done:
            element=heapq.heappop(self.heap)
            if not element[2]:
                self.nelements-=1
                element[2]=True
                return element[3],element[0]
        return None,None
    def peek(self):
        return self.heap[0]
    def __len__(self):
        return self.nelements
            

class AStar():
    def __init__(self, env):
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_edges = env.get_edges

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            if current is not None:
                total_path.append(current)
        return total_path[::-1]

    def search(self, initial_state,agent_name,start_time,time_limit):
        """
        low level search 
        """
        
        closed_set = set()
        open_list=PriorityQueue()
        hstart=self.admissible_heuristic(initial_state,agent_name)
        open_list.insert((initial_state,None),(hstart,hstart))

        came_from = {}

        g_score = defaultdict(lambda:(float("inf"),float("inf")))
        g_score[initial_state] = (initial_state.time,0)

        h_score = {initial_state:hstart}

        f_score = {} 

        f_score[initial_state] = hstart

        while len(open_list)>0:
            if timeit.default_timer()-start_time>=time_limit:
                raise TimeoutError(f"AStar.search({agent_name}) timed out")
            current,priority=open_list.pop()
            state,edge_taken=current
            closed_set.add(state)

            edge_list = self.get_edges(state,agent_name)

            for edge in edge_list:
                neighbor=edge.exit_state()
                if neighbor in closed_set:
                    continue
                if self.is_at_goal(edge, agent_name):
                    came_from[edge] = edge_taken
                    return self.reconstruct_path(came_from, edge),neighbor
                parent_time,parent_cost=g_score[state]
                tentative_g_score = (parent_time+1,parent_cost + edge.cost)
                if tentative_g_score>=g_score[neighbor]:
                    #don't reinsert if we found a longer path to neighbor than we already have processed
                    continue

                came_from[edge] = edge_taken

                g_score[neighbor] = tentative_g_score
                if neighbor not in h_score:
                    h_score[neighbor] = self.admissible_heuristic(neighbor, agent_name)
                f_score[neighbor] =tuple(gi+hi for gi,hi in zip(g_score[neighbor],h_score[neighbor]))
                open_list.insert((neighbor,edge),(f_score[neighbor],h_score[neighbor]))
        return False,None