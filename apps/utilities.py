import numpy as np
import pickle

def str2vec(s: str):
    remove = set((' ', '[' , ']' , '(' , ')' , '\n'))
    allowed = set((',', '-', '.'))
    new_s = []
    for c in s:
        if c not in remove:
            new_s.append(c)
        
    new_s = "".join(new_s)
    for c in new_s:
        if(c.isnumeric() or c in allowed): 
            continue
        else:
            return None, False
    
    l = new_s.split(",")
    for i in range(len(l)):
        l[i] = float(l[i])
    return np.array(l), True
    
def str2matrix(s: str):
    remove = set((' ', '[' , ']' , '(' , ')' , '\n'))
    allowed = set((',', '-', ';', '.'))
    new_s = []
    for c in s:
        if c not in remove:
            new_s.append(c)
    new_s = "".join(new_s)
    for c in new_s:
        if(c.isnumeric() or c in allowed): 
            continue
        else:
            return None, False
    
    l = new_s.split(";")
    for i in range(len(l)):
        l[i] = l[i].split(",")
        for j in range(len(l)):
            l[i][j] = float(l[i][j])
    return np.array(l), True

def k_decimal(n,k=3):
    return int(n * 10**k)/10**k

def k_decimal_matrix(m,k=3):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            m[i][j] = k_decimal(m[i][j],k)
        
def vec2str(vector):
    s = ["["]
    for v in vector:
        s.append(str(v))
        s.append(", ")
    s.pop()
    s.append("]")
    return "".join(s)

def save(vectors, name):
    with open(name,'wb') as f:
        pickle.dump(vectors, f)
        
def load(name):
    with open(name,'rb') as f:
        vectors  = pickle.load(f)
    return vectors