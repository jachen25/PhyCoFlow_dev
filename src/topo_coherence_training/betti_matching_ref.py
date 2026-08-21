import numpy as np
import torch
import heapq
import copy
import random as rd
from operator import itemgetter
# Vendored from the Stucki et al. Betti-matching reference implementation.
# Plotting, representative-cycle, and Wasserstein code have been removed;
# restore from upstream if needed.


class UnionFind:
    def __init__(self, n, dual=False):
        self.n = n
        self.dual = dual
        self.parent = list(range(n))
        self.rank = n*[0]
        self.birth = list(range(n))


    def set_birth(self, x, val):
        self.birth[x] = val
        return

    
    def get_birth(self, x):
        y = self.find(x)
        return self.birth[y]


    def find(self, x):
        y = x
        z = self.parent[y]
        while z != y:
            y = z
            z = self.parent[y]
        y = self.parent[x]
        while z != y:
            self.parent[x] = z
            x = y
            y = self.parent[x]
        return z

    
    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
            if self.dual == False:
                self.birth[x] = min(self.birth[x],self.birth[y])
            else:
                self.birth[x] = max(self.birth[x],self.birth[y])
        else:
            self.parent[x] = y
            if self.dual == False:
                self.birth[y] = min(self.birth[x],self.birth[y])
            else:
                self.birth[y] = max(self.birth[x],self.birth[y])
            if self.rank[x] == self.rank[y]:
                self.rank[y] += 1

    
    def get_component(self, x):
        component = []
        x = self.find(x)
        for y in range(self.n):
            z = self.find(y)
            if z == x:
                component.append(y) 
        return component



class BoundaryMatrix:
    def __init__(self):
        self.columns = {}
        self.pivots = {}
        self.columns_to_reduce = None
        self.reduced = False


    def set_one(self, i, j):
        if j not in self.columns.keys():
            self.columns[j] = [-i]
            return
        
        if -i not in self.columns[j]:
            heapq.heappush(self.columns[j], -i)
            return

    
    def add_column(self, i, j):
        #self.columns[j] = list(heapq.merge(self.columns[j],self.columns[i]))
        #self.columns[j].extend(self.columns[i])
        #heapq.heapify(self.columns[j])
        heapq.heappop(self.columns[j])
        for item in self.columns[i][1:]:
            if item == self.columns[j][0]:
                heapq.heappop(self.columns[j])
            else:
                heapq.heappush(self.columns[j], item)
        return
        

    def get_pivot(self, j):
        if j not in self.columns.keys():
            return -1

        count = 0
        while (len(self.columns[j]) != 0) and (count%2 == 0):
            pivot = -heapq.heappop(self.columns[j])
            count = 1
            while (len(self.columns[j]) != 0) and (-self.columns[j][0] == pivot):
                heapq.heappop(self.columns[j])
                count += 1
        if count%2 == 0:
            return -1

        else:
            heapq.heappush(self.columns[j], -pivot)
            return pivot


    def reduce(self, clearing=False):
        if self.reduced == True:
            return

        if self.columns_to_reduce == None:
            clearing = False
            self.columns_to_reduce = [sorted(self.columns.keys()),[],[]]
        for dim in [2,1]:
            for column in self.columns_to_reduce[dim]:
                pivot = self.get_pivot(column)
                while pivot in self.pivots.keys():
                    previous = self.pivots[pivot]
                    self.add_column(previous, column)
                    pivot = self.get_pivot(column)
                if pivot != -1:
                    self.pivots[pivot] = column
                    if clearing == True and dim != 1:
                        self.columns.pop(pivot)
                        self.columns_to_reduce[dim-1].remove(pivot)    
                else:
                    self.columns.pop(column)
        self.reduced = True
        return

    
    def get_Pairings(self):      
        return list(zip(self.pivots.keys(), self.pivots.values())) 


    def get_column(self, j):
        column = []
        if j not in self.columns.keys():
            return column
        
        column_heap = copy.deepcopy(self.columns[j])
        while(len(column_heap) != 0):
            pivot = -heapq.heappop(column_heap)
            count = 1
            while (len(column_heap) != 0) and (-column_heap[0] == pivot):
                heapq.heappop(column_heap)
                count += 1
            if count%2 == 1:
                column.append(pivot)
        return column

        

class CubicalPersistence:
    def __init__(self, Picture, relative=False, reduced=False, filtration='sublevel', construction='V', valid='positive', get_image_columns_to_reduce=False, get_critical_edges=False, training=False):
        self.reduced = reduced
        assert filtration in ['sublevel','superlevel']
        self.filtration = filtration
        assert construction in ['V','T']       
        self.construction = construction
        assert valid in ['all','nonnegative','positive']
        self.valid = valid
        assert not (get_image_columns_to_reduce and get_critical_edges)
        self.get_image_columns_to_reduce = get_image_columns_to_reduce
        self.get_critical_edges = get_critical_edges
        if self.get_critical_edges:
            self.critical_edges = []
        if type(Picture) == torch.Tensor:
            Picture = torch.squeeze(Picture)
        self.m, self.n = Picture.shape
        if relative == False:
                self.PixelMap = Picture
        else:
            self.m += 2
            self.n += 2
            if type(Picture) == torch.Tensor:
                if self.filtration == 'sublevel':
                    min = torch.min(Picture)
                    if training:
                        self.PixelMap = min*torch.ones((self.m,self.n)).cuda()
                    else:
                        self.PixelMap = min*torch.ones((self.m,self.n))
                    self.PixelMap[1:self.m-1,1:self.n-1] = Picture
                else:
                    max = torch.max(Picture)
                    if training:
                        self.PixelMap = max*torch.ones((self.m,self.n)).cuda()
                    else:
                        self.PixelMap = max*torch.ones((self.m,self.n))
                    self.PixelMap[1:self.m-1,1:self.n-1] = Picture
            else:
                if self.filtration == 'sublevel':
                    min = np.min(Picture)
                    self.PixelMap = min*np.ones((self.m,self.n))
                    self.PixelMap[1:self.m-1,1:self.n-1] = Picture
                else:
                    max = np.max(Picture)
                    self.PixelMap = max*np.ones((self.m,self.n))
                    self.PixelMap[1:self.m-1,1:self.n-1] = Picture
        if self.construction == 'V':
            self.M = 2*self.m-1
            self.N = 2*self.n-1
        else:
            self.M = 2*self.m+1
            self.N = 2*self.n+1
        if type(self.PixelMap) == torch.Tensor:
            self.ValueMap = torch.zeros((self.M,self.N))
        else:
            self.ValueMap = np.zeros((self.M,self.N))
        self.IndexMap = -np.ones((self.M,self.N), dtype=int)
        self.num_cubes = self.M*self.N
        self.num_edges = int((self.M*self.N-1)/2)
        self.edges = self.num_edges*[0]
        self.coordinates = self.num_cubes*[0]
        self.intervals = [[],[]]
        self.columns_to_reduce = [[],[],[]]
        self.set_CubeMap()
        self.compute_persistence(valid=valid)
 
        
    def set_CubeMap(self):
        if type(self.PixelMap) == torch.Tensor:
            if self.filtration == 'sublevel':
                PixelMap = np.array(self.PixelMap.cpu().detach().numpy(), dtype=float)
            else:
                PixelMap = -np.array(self.PixelMap.cpu().detach().numpy(), dtype=float)
        else:
            if self.filtration == 'sublevel':
                PixelMap = np.array(self.PixelMap, dtype=float)
            else:
                PixelMap = -np.array(self.PixelMap, dtype=float)
        if self.construction == 'V':
            counter = int(self.num_cubes-1)
            counter_edges = int(self.num_edges-1)
            max = np.max(PixelMap)
            while max != -np.inf:
                argmax = np.where(PixelMap == max)
                for i,j in zip(argmax[0],argmax[1]):
                    for k in [-1,1]:
                        for l in [-1,1]:
                            if 2*i+k >=0 and 2*i+k <= self.M-1 and 2*j+l >= 0 and 2*j+l <= self.N-1:
                                if self.IndexMap[2*i+k,2*j+l] == -1:
                                    self.ValueMap[2*i+k,2*j+l] = self.PixelMap[i,j]
                                    self.IndexMap[2*i+k,2*j+l] = counter
                                    self.coordinates[counter] = (2*i+k,2*j+l)
                                    counter = int(counter-1)
                for i,j in zip(argmax[0],argmax[1]):  
                    for k in [-1,1]:
                        if 2*i+k >=0 and 2*i+k <= self.M-1:
                            if self.IndexMap[2*i+k,2*j] == -1:
                                self.ValueMap[2*i+k,2*j] = self.PixelMap[i,j]
                                self.IndexMap[2*i+k,2*j] = counter
                                self.coordinates[counter] = (2*i+k,2*j)
                                self.edges[counter_edges] = counter
                                counter = int(counter-1)
                                counter_edges = int(counter_edges-1)
                        if 2*j+k >=0 and 2*j+k <= self.N-1:
                            if self.IndexMap[2*i,2*j+k] == -1:
                                self.ValueMap[2*i,2*j+k] = self.PixelMap[i,j]
                                self.IndexMap[2*i,2*j+k] = counter
                                self.coordinates[counter] = (2*i,2*j+k)
                                self.edges[counter_edges] = counter
                                counter = int(counter-1)
                                counter_edges = int(counter_edges-1)
                for i,j in zip(argmax[0],argmax[1]):
                    self.ValueMap[2*i,2*j] = self.PixelMap[i,j]
                    self.IndexMap[2*i,2*j] = counter
                    self.coordinates[counter] = (2*i,2*j)
                    counter = int(counter-1)                
                    PixelMap[i,j] = -np.inf  
                max = np.max(PixelMap)
        else:
            counter = int(0)
            counter_edges = int(0)
            min = np.min(PixelMap)
            while min != np.inf:
                argmin = np.where(PixelMap == min)
                for i,j in zip(argmin[0],argmin[1]):
                    for k in [-1,1]:
                        for l in [-1,1]:
                            if self.IndexMap[2*i+1+k,2*j+1+l] == -1:
                                self.ValueMap[2*i+1+k,2*j+1+l] = self.PixelMap[i,j]
                                self.IndexMap[2*i+1+k,2*j+1+l] = counter
                                self.coordinates[counter] = (2*i+1+k,2*j+1+l)
                                counter = int(counter+1)
                for i,j in zip(argmin[0],argmin[1]):   
                    for k in [-1,1]:
                        if self.IndexMap[2*i+1+k,2*j+1] == -1:
                            self.ValueMap[2*i+1+k,2*j+1] = self.PixelMap[i,j]
                            self.IndexMap[2*i+1+k,2*j+1] = counter
                            self.coordinates[counter] = (2*i+1+k,2*j+1)
                            self.edges[counter_edges] = counter
                            counter = int(counter+1)
                            counter_edges = int(counter_edges+1)
                        if self.IndexMap[2*i+1,2*j+1+k] == -1:
                            self.ValueMap[2*i+1,2*j+1+k] = self.PixelMap[i,j]
                            self.IndexMap[2*i+1,2*j+1+k] = counter
                            self.coordinates[counter] = (2*i+1,2*j+1+k)
                            self.edges[counter_edges] = counter
                            counter = int(counter+1)
                            counter_edges = int(counter_edges+1)    
                for i,j in zip(argmin[0],argmin[1]):
                    self.ValueMap[2*i+1,2*j+1] = self.PixelMap[i,j]
                    self.IndexMap[2*i+1,2*j+1] = counter
                    self.coordinates[counter] = (2*i+1,2*j+1)
                    counter = int(counter+1)                                    
                    PixelMap[i,j] = np.inf  
                min = np.min(PixelMap)


    def index_to_coordinates(self, idx):
        return self.coordinates[idx]


    def index_to_dim(self, idx):
        i,j = self.index_to_coordinates(idx)
        if i%2 == 0 and j%2 == 0:
            dim = 0 
        elif i%2+j%2 == 1:
            dim = 1  
        else:
            dim = 2      
        return dim


    def index_to_value(self, idx):
        if idx == np.inf:
            if self.filtration == 'sublevel':
                return np.inf
                
            else:
                return -np.inf

        x,y = self.index_to_coordinates(idx)
        return self.ValueMap[x,y]


    def fine_to_coarse(self, interval):
        return (self.index_to_value(interval[0]),self.index_to_value(interval[1]))


    def valid_interval(self, interval, valid='positive'):
        if valid in ['all','nonnegative']:
            return True
        
        else:
            if self.filtration == 'sublevel':
                return self.index_to_value(interval[0]) < self.index_to_value(interval[1])

            else:
                return self.index_to_value(interval[0]) > self.index_to_value(interval[1])


    def get_boundary(self, idx):
        boundary = []
        x,y = self.index_to_coordinates(idx)
        if x%2 != 0:
            boundary.extend([self.IndexMap[x-1,y],self.IndexMap[x+1,y]])
        if y%2 != 0:
            boundary.extend([self.IndexMap[x,y-1],self.IndexMap[x,y+1]])
        return boundary

    
    def get_dual_boundary(self, idx):
        boundary = []
        x,y = self.index_to_coordinates(idx)
        if x%2 == 0:
            if x == 0:
                boundary.extend([self.num_cubes,self.IndexMap[x+1,y]])
            elif x == self.M-1:
                boundary.extend([self.num_cubes,self.IndexMap[x-1,y]])
            else:
                boundary.extend([self.IndexMap[x-1,y],self.IndexMap[x+1,y]])
        if y%2 == 0:
            if y == 0:
                boundary.extend([self.num_cubes,self.IndexMap[x,y+1]])
            elif y == self.N-1:
                boundary.extend([self.num_cubes,self.IndexMap[x,y-1]])
            else:
                boundary.extend([self.IndexMap[x,y-1],self.IndexMap[x,y+1]])
        return boundary
            

    def compute_dim0(self, valid='positive'):
        if self.reduced == False:
            self.intervals[0] = [(0,np.inf)]
        else:
            self.intervals[0] = []
        UF = UnionFind(self.num_cubes, dual=False)
        for edge in self.columns_to_reduce[1]:
            boundary = self.get_boundary(edge)
            x = UF.find(boundary[0])
            y = UF.find(boundary[1])
            if x == y:
                continue
            birth = max(UF.get_birth(x), UF.get_birth(y))
            if self.valid_interval((birth,edge), valid=valid):
                self.intervals[0].append((birth,edge))
            UF.union(x,y)
        return

    
    def compute_dim1(self, valid='positive'):
        if self.get_image_columns_to_reduce:
            UF = UnionFind(self.num_cubes+1, dual=True)    
            for edge in self.edges[::-1]:
                boundary = self.get_dual_boundary(edge)
                x = UF.find(boundary[0])
                y = UF.find(boundary[1])
                if x == y:
                    self.columns_to_reduce[1].append(edge)
                    continue
                birth = min(UF.get_birth(x), UF.get_birth(y))
                self.columns_to_reduce[2].append(birth)
                if self.valid_interval((edge,birth), valid=valid):
                    self.intervals[1].append((edge,birth))       
                UF.union(x,y)
            self.columns_to_reduce[1].reverse()
            self.columns_to_reduce[2].sort()
        elif self.get_critical_edges:
            UF = UnionFind(self.num_cubes+1, dual=True)    
            for edge in self.edges[::-1]:
                boundary = self.get_dual_boundary(edge)
                x = UF.find(boundary[0])
                y = UF.find(boundary[1])
                if x == y:
                    self.columns_to_reduce[1].append(edge)
                    continue
                self.critical_edges.append(edge)
                birth = min(UF.get_birth(x), UF.get_birth(y))
                if self.valid_interval((edge,birth), valid=valid):
                    self.intervals[1].append((edge,birth))       
                UF.union(x,y)
            self.columns_to_reduce[1].reverse()
        else:
            UF = UnionFind(self.num_cubes+1, dual=True)    
            for edge in self.edges[::-1]:
                boundary = self.get_dual_boundary(edge)
                x = UF.find(boundary[0])
                y = UF.find(boundary[1])
                if x == y:
                    self.columns_to_reduce[1].append(edge)
                    continue
                birth = min(UF.get_birth(x), UF.get_birth(y))
                if self.valid_interval((edge,birth), valid=valid):
                    self.intervals[1].append((edge,birth))       
                UF.union(x,y)
            self.columns_to_reduce[1].reverse()
        return


    def compute_persistence(self, valid='positive'):
        self.compute_dim1(valid=valid)
        self.compute_dim0(valid=valid)
        return


    def get_intervals(self, refined=False):
        if refined:
            return copy.deepcopy(self.intervals)

        intervals = [[self.fine_to_coarse(interval) for interval in self.intervals[dim]] for dim in range(2)]
        return intervals

    
    def get_Betti_numbers(self, threshold=0.5):
        betti = [0,0]
        for dim in [0,1]:
            for (i,j) in self.intervals[dim]:
                if self.valid_interval((i,j), valid='positive'):
                    a = self.index_to_value(i)
                    b = self.index_to_value(j)
                    if self.filtration == 'sublevel':
                        if a <= threshold and threshold < b:
                            betti[dim] += 1
                    else:
                        if a >= threshold and threshold > b:
                            betti[dim] += 1
        return betti

    
    def get_generating_vertex(self, cube):
        boundary = [cube]
        while boundary != []:
            generating_boundary = max(boundary)
            boundary = self.get_boundary(generating_boundary)
        return generating_boundary



class ImagePersistence:
    def __init__(self, CubicalPersistence_0, CubicalPersistence_1, valid='all', use_UnionFind=True):
        self.CP_0 = CubicalPersistence_0
        self.CP_1 = CubicalPersistence_1
        assert self.CP_0.m == self.CP_1.m and self.CP_0.n == self.CP_1.n
        assert self.CP_0.reduced == self.CP_1.reduced
        self.reduced = self.CP_0.reduced
        assert self.CP_0.filtration == self.CP_1.filtration
        self.filtration = self.CP_0.filtration
        assert self.CP_0.construction == self.CP_1.construction
        assert valid in ['all','nonnegative','positive']

        self.intervals = [[],[]]
        if use_UnionFind:
            assert self.CP_0.get_critical_edges
            self.compute_persistence_UF(valid=valid)
        else:
            assert self.CP_1.get_image_columns_to_reduce
            self.B = BoundaryMatrix()
            self.set_BoundaryMatrix()
            self.compute_persistence(valid=valid)


    def set_BoundaryMatrix(self):
        self.B.columns_to_reduce = self.CP_1.columns_to_reduce
        for idx_col in self.B.columns_to_reduce[1]:
            i,j = self.CP_1.index_to_coordinates(idx_col)
            if i%2 == 0:
                idx_row = self.CP_0.IndexMap[i,j+1]
                self.B.set_one(idx_row,idx_col)
                idx_row = self.CP_0.IndexMap[i,j-1]
                self.B.set_one(idx_row,idx_col)
            else:
                idx_row = self.CP_0.IndexMap[i+1,j]
                self.B.set_one(idx_row,idx_col)
                idx_row = self.CP_0.IndexMap[i-1,j]
                self.B.set_one(idx_row,idx_col)
        for idx_col in self.B.columns_to_reduce[2]:
            i,j = self.CP_1.index_to_coordinates(idx_col)               
            idx_row = self.CP_0.IndexMap[i,j+1]
            self.B.set_one(idx_row,idx_col)
            idx_row = self.CP_0.IndexMap[i,j-1]
            self.B.set_one(idx_row,idx_col)               
            idx_row = self.CP_0.IndexMap[i+1,j]
            self.B.set_one(idx_row,idx_col)
            idx_row = self.CP_0.IndexMap[i-1,j]
            self.B.set_one(idx_row,idx_col)            
        return


    def fine_to_coarse(self, interval):
        return (self.CP_0.index_to_value(interval[0]),self.CP_1.index_to_value(interval[1]))

    
    def valid_interval(self, interval, valid='all'):
        if valid == 'all':
            return True

        elif valid == 'nonnegative':
            if self.CP_0.filtration == 'sublevel':
                return self.CP_0.index_to_value(interval[0]) <= self.CP_1.index_to_value(interval[1])

            else:
                return self.CP_0.index_to_value(interval[0]) >= self.CP_1.index_to_value(interval[1])

        else:
            if self.CP_0.filtration == 'sublevel':
                return self.CP_0.index_to_value(interval[0]) < self.CP_1.index_to_value(interval[1])

            else:
                return self.CP_0.index_to_value(interval[0]) > self.CP_1.index_to_value(interval[1])
            

    def compute_persistence(self, valid='all'):
        self.B.reduce(clearing=False)
        pairings = self.B.get_Pairings()
        if self.reduced == False:
            self.intervals[0] = [(0,np.inf)]
        else:
            self.intervals[0] = []
        for (i,j) in pairings:
            if self.valid_interval((i,j), valid=valid):
                self.intervals[self.CP_0.index_to_dim(i)].append((i,j))                                             
        return


    def get_boundary(self, idx):
        boundary = []
        x,y = self.CP_1.index_to_coordinates(idx)
        if x%2 != 0:
            boundary.extend([self.CP_0.IndexMap[x-1,y],self.CP_0.IndexMap[x+1,y]])
        if y%2 != 0:
            boundary.extend([self.CP_0.IndexMap[x,y-1],self.CP_0.IndexMap[x,y+1]])
        return boundary


    def get_dual_boundary(self, idx):
        boundary = []
        x,y = self.CP_0.index_to_coordinates(idx)
        if x%2 == 0:
            if x == 0:
                boundary.extend([self.CP_1.num_cubes,self.CP_1.IndexMap[x+1,y]])
            elif x == self.CP_0.M-1:
                boundary.extend([self.CP_1.num_cubes,self.CP_1.IndexMap[x-1,y]])
            else:
                boundary.extend([self.CP_1.IndexMap[x-1,y],self.CP_1.IndexMap[x+1,y]])
        if y%2 == 0:
            if y == 0:
                boundary.extend([self.CP_1.num_cubes,self.CP_1.IndexMap[x,y+1]])
            elif y == self.CP_0.N-1:
                boundary.extend([self.CP_1.num_cubes,self.CP_1.IndexMap[x,y-1]])
            else:
                boundary.extend([self.CP_1.IndexMap[x,y-1],self.CP_1.IndexMap[x,y+1]])
        return boundary
    

    def compute_dim0(self, valid='all'):
        self.intervals[0] = [(0,np.inf)]
        UF = UnionFind(self.CP_0.num_cubes, dual=False)
        for edge in self.CP_1.columns_to_reduce[1]:
            boundary = self.get_boundary(edge)
            x = UF.find(boundary[0])
            y = UF.find(boundary[1])
            if x == y:
                continue
            birth = max(UF.get_birth(x),UF.get_birth(y))
            if self.valid_interval((birth,edge), valid=valid):
                self.intervals[0].append((birth,edge))
            UF.union(x,y)
        return


    def compute_dim1(self, valid='all'):
        UF = UnionFind(self.CP_1.num_cubes+1, dual=True)
        for edge in self.CP_0.critical_edges:
            boundary = self.get_dual_boundary(edge)
            x = UF.find(boundary[0])
            y = UF.find(boundary[1])
            if x == y:
                continue
            birth = min(UF.get_birth(x),UF.get_birth(y))
            if self.valid_interval((edge,birth), valid=valid):
                self.intervals[1].append((edge,birth))
            UF.union(x,y)
        return


    def compute_persistence_UF(self, valid='all'):
        self.compute_dim0(valid=valid)
        self.compute_dim1(valid=valid)
        return



class InducedMatching:
    def __init__(self, ImagePersistence):
        self.IP = ImagePersistence
        self.matched = [[],[]]
        self.unmatched_0 = copy.deepcopy(self.IP.CP_0.intervals)
        self.unmatched_1 = copy.deepcopy(self.IP.CP_1.intervals)
        self.match()


    def find_match(self, interval, dim):
        match_0 = None
        match_1 = None
        for (a,b) in self.unmatched_0[dim]:
            if a == interval[0]:
                match_0 = (a,b)
                break
        if match_0 == None:
            return None

        for (a,b) in self.unmatched_1[dim]:
            if b == interval[1]:
                match_1 = (a,b)
                break
        if match_1 == None:
            return None

        else:
            return (match_0,interval,match_1)


    def match(self):
        for dim in range(2):
            for (a,b) in self.IP.intervals[dim]:
                match = self.find_match((a,b), dim)
                if match == None:
                    continue
                else:
                    self.matched[dim].append(match)
                    self.unmatched_0[dim].remove(match[0])
                    self.unmatched_1[dim].remove(match[2])

    
    def get_matching(self):
        matched = [[(self.IP.CP_0.fine_to_coarse(match[0]), self.IP.CP_1.fine_to_coarse(match[2]))for match in self.matched[dim]]for dim in range(2)]
        unmatched_0 = [[self.IP.CP_0.fine_to_coarse(interval)for interval in self.unmatched_0[dim]]for dim in range(2)]
        unmatched_1 = [[self.IP.CP_1.fine_to_coarse(interval)for interval in self.unmatched_1[dim]]for dim in range(2)]
        return matched, unmatched_0, unmatched_1



class BettiMatching:
    def __init__(self, Picture_0, Picture_1, relative=False, reduced=False, filtration='sublevel', construction='V', comparison='union', valid='positive', valid_image='all',  use_UnionFind_for_image=True, training=False):
        assert valid in ['all','nonnegative','positive']
        assert valid_image in ['all','nonnegative','positive']
        assert filtration in ['sublevel','superlevel']
        self.filtration = filtration
        assert construction in ['V','T']
        self.construction = construction
        assert comparison in ['union','intersection']
        self.comparison = comparison

        if comparison == 'union':
            if filtration == 'sublevel':
                if type(Picture_0) == torch.Tensor:
                    Picture_comp = torch.minimum(Picture_0, Picture_1)
                else:
                    Picture_comp = np.minimum(Picture_0, Picture_1)
            else:
                if type(Picture_0) == torch.Tensor:
                    Picture_comp = torch.maximum(Picture_0, Picture_1)
                else:
                    Picture_comp = np.maximum(Picture_0, Picture_1)
            self.CP_0 = CubicalPersistence(Picture_0, relative=relative, reduced=reduced, valid=valid, filtration=filtration, construction=construction, get_critical_edges=use_UnionFind_for_image, training=training)
            self.CP_1 = CubicalPersistence(Picture_1, relative=relative, reduced=reduced, valid=valid, filtration=filtration, construction=construction, get_critical_edges=use_UnionFind_for_image, training=training)          
            self.CP_comp = CubicalPersistence(Picture_comp, relative=relative, reduced=reduced, valid=valid, filtration=filtration, construction=construction, get_image_columns_to_reduce=not use_UnionFind_for_image, training=training)
            self.IP_0 = ImagePersistence(self.CP_0, self.CP_comp, valid=valid_image, use_UnionFind=use_UnionFind_for_image)
            self.IP_1 = ImagePersistence(self.CP_1, self.CP_comp, valid=valid_image, use_UnionFind=use_UnionFind_for_image)
        else:
            if filtration == 'sublevel':
                if type(Picture_0) == torch.Tensor:
                    Picture_comp = torch.maximum(Picture_0, Picture_1)
                else:
                    Picture_comp = np.maximum(Picture_0, Picture_1)
            else:
                if type(Picture_0) == torch.Tensor:
                    Picture_comp = torch.minimum(Picture_0, Picture_1)
                else:
                    Picture_comp = np.minimum(Picture_0, Picture_1)
            self.CP_comp = CubicalPersistence(Picture_comp, relative=relative, reduced=reduced, valid=valid, filtration=filtration, construction=construction, get_critical_edges=use_UnionFind_for_image, training=training)
            self.CP_0 = CubicalPersistence(Picture_0, relative=relative, reduced=reduced, valid=valid, filtration=filtration, construction=construction, get_image_columns_to_reduce=not use_UnionFind_for_image, training=training)
            self.CP_1 = CubicalPersistence(Picture_1, relative=relative, reduced=reduced, valid=valid, filtration=filtration, construction=construction, get_image_columns_to_reduce=not use_UnionFind_for_image, training=training) 
            self.IP_0 = ImagePersistence(self.CP_comp, self.CP_0, valid=valid_image, use_UnionFind=use_UnionFind_for_image)
            self.IP_1 = ImagePersistence(self.CP_comp, self.CP_1, valid=valid_image, use_UnionFind=use_UnionFind_for_image)
        self.IM_0 = InducedMatching(self.IP_0)
        self.IM_1 = InducedMatching(self.IP_1)
        self.matched = [[],[]]
        self.unmatched_0 = copy.deepcopy(self.CP_0.intervals)
        self.unmatched_comp = copy.deepcopy(self.CP_comp.intervals)
        self.unmatched_1 = copy.deepcopy(self.CP_1.intervals)
        self.match()


    def match(self):
        matched_1 = copy.deepcopy(self.IM_1.matched)    
        if self.comparison == 'union':
            for dim in range(2):
                for match_0 in self.IM_0.matched[dim]:
                    for match_1 in matched_1[dim]:
                        if match_0[2] == match_1[2]:
                            self.matched[dim].append((match_0[0],match_0[2],match_1[0]))
                            self.unmatched_0[dim].remove(match_0[0])
                            self.unmatched_comp[dim].remove(match_0[2])
                            self.unmatched_1[dim].remove(match_1[0])
                            matched_1[dim].remove(match_1)
                            break
        else:
            for dim in range(2):
                for match_0 in self.IM_0.matched[dim]:
                    for match_1 in matched_1[dim]:
                        if match_0[0] == match_1[0]:
                            self.matched[dim].append((match_0[2],match_0[0],match_1[2]))
                            self.unmatched_0[dim].remove(match_0[2])
                            self.unmatched_comp[dim].remove(match_0[0])
                            self.unmatched_1[dim].remove(match_1[2])
                            matched_1[dim].remove(match_1)
                            break              
        return


    def get_matching(self, refined=False):
        if refined:
            return copy.deepcopy(self.matched), copy.deepcopy(self.unmatched_0), copy.deepcopy(self.unmatched_1)
            
        matched = [[(self.CP_0.fine_to_coarse(match[0]), self.CP_1.fine_to_coarse(match[2]))for match in self.matched[dim]]for dim in range(2)]
        unmatched_0 = [[self.CP_0.fine_to_coarse(interval)for interval in self.unmatched_0[dim]]for dim in range(2)]
        unmatched_1 = [[self.CP_1.fine_to_coarse(interval)for interval in self.unmatched_1[dim]]for dim in range(2)]
        return matched, unmatched_0, unmatched_1
