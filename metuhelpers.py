from commons import progressBar
import numpy as np

def calculate_distances(origin, array, distance_norm):
    map_f = distance_norm(origin)
    return map_f(array)

def euclidean_sq(origin):
    def euclidean_sq_map(point):
        return np.sum(np.power(point-origin,2), axis=-1)
    return euclidean_sq_map

def cosine(origin):
    origin_norm = np.linalg.norm(origin)
    def cosine_map(point):
        return 1.0 - np.dot(point, origin) / np.linalg.norm(point, axis=-1) / origin_norm
    return cosine_map

def arg_binsearch(array, x, cond):
    b = -1
    e = len(array)
    while e - b > 1:
        middle = (e+b)//2
        if cond(array[middle], x):
            b = middle
        else:
            e = middle
    return b

class ProxyArray:
    def __init__(self, origin, proxy):
        self.origin = origin
        self.proxy = proxy
    
    def __getitem__(self, idx):
        return self.origin[self.proxy[idx]]
    
    def __len__(self):
        return len(self.origin)


def get_ranks(trainArray, evalArray, x, distance_norm):
    '''
    Returns rankings table of evalDistX distances in trainDistX sorted order
    '''
    trainDistX = calculate_distances(x, trainArray, distance_norm)
    evalDistX = calculate_distances(x, evalArray, distance_norm)
    argSortTrain = np.argsort(trainDistX)
    proxy = ProxyArray(trainDistX, argSortTrain)
    
    return np.array(
        [(arg_binsearch(proxy, dist, np.less) + arg_binsearch(proxy, dist, np.less_equal))/2 for dist in evalDistX]
    )


def calculate_rank(trainArray, evalArray, evalGroups, distance_norm):
    '''
    For each group of similar images in evalGroups (dictionary groupID: [indices of images in group]) calculates ranks
    of each image in the group ranking in trainArray
    '''
    progbarsteps = (i for i in range(len(evalArray)))
    pbar = progressBar(progbarsteps, iterable_size = len(evalArray))
    results = {}
    for group, groupIdxs in evalGroups.items():
        for idx in groupIdxs:
            ranks = get_ranks(trainArray, evalArray, evalArray[idx], distance_norm)
            relevant_ranks = np.array([ranks[i] for i in groupIdxs if i != idx])
            results[idx] = relevant_ranks
            next(pbar)
    return results


def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of given element in
    the list- listOfElements 
    '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list