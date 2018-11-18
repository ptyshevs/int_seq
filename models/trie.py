import math
import pickle
from functools import reduce
import tqdm


class _prefixTree:
    def __init__(self):
        self.data = {}
        self.puts = 0
        self.nodes = 0
        self.params = {'puts': self.puts, 'nodes': self.nodes}
    
    def put(self, seq, value):
        node=self.data
        nodeCreated=False
        for i in range(0,len(seq)):
            item=seq[i]
            if not item in node:
                node[item]={}
                if 'value' in node:
                    del node['value']
                self.nodes+=1
                nodeCreated=True
            node=node[item]
        if nodeCreated:
            node['value']=value
            self.puts+=1
        elif 'value' in node:
            node['value']=max(node['value'], value)
    
    def prefix(self, seq):
        list=[]
        node=self.data
        for i in range(0,len(seq)):
            item=seq[i]
            if item in node:
                node=node[item]
            else:
                return list
        self.addAll(seq, node, list)
        return list
    
    def hasPrefix(self, seq):
        node=self.data
        for i in range(0,len(seq)):
            item=seq[i]
            if item in node:
                node=node[item]
            else:
                return False
        return True
    
    def addAll(self, seq, node, list):
        if 'value' in node:
            list.append( ( seq, node['value'] ) )
        for key in node:
            if key != 'value':
                self.addAll(seq + [key], node[key], list)
    
    def __repr__(self):
        params = ', '.join([f"{par}={val}" for par, val in self.params.items()])
        return f"{self.__class__.__name__}({params})"


class Trie:
    def __init__(self, shift=4, load=True):
        """
        Prefixes lookup
        
        Courtesy of: https://www.kaggle.com/balzac/prefixes-lookup-0-22
        """
        self.shift = shift
        self.trie = _prefixTree()
        self.params = {'shift': self.shift, 'trie': self.trie}
        if load:
            self.load()
    
    def fit(self, data):
        for seq, ind in tqdm.tqdm(zip(data, data.index), desc='Fitting to dataset'):
            der_train = [int(x) for x in seq]
            for derAttempts in range(self.shift):
                train_seq = der_train
                firstInTrie = False
                for subseqAttempts in range(self.shift - derAttempts):
                    while len(train_seq)>0 and (-1 <= train_seq[0] <= 0):
                        train_seq = train_seq[1:]
                    signature = self._find_signature(train_seq)
                    if self.trie.hasPrefix(signature):
                        if subseqAttempts == 0:
                            firstInTrie = True
                        break
                    #задаем веса для каждого узла len(train_seq)*100//len(der_train)
                    self.trie.put(signature, len(train_seq) * 100 // len(der_train))
                    if len(train_seq) <= (self.shift - 1):
                        break
                    train_seq = train_seq[1:]
                if firstInTrie:
                    break
                der_train = self._find_derivative(der_train)
        return self
    
    def _find_derivative(self, seq):
        return [0] if len(seq)<=1 else [seq[i]-seq[i-1] for i in range(1,len(seq))]
    
    def _find_signature(self, seq):
        nonzero_seq = [d for d in seq if d!=0]
        if len(nonzero_seq)==0:
            return seq
        sign = 1 if nonzero_seq[0]>0 else -1
        gcd = self._find_gcd(seq)
        return [sign*x//gcd for x in seq]
    
    def _find_gcd(self, seq):
        gcd = seq[0]
        for i in range(1,len(seq)):
            gcd=math.gcd(gcd, seq[i])
        return gcd

    def _findNext(self, seq):
        while True:
            nonZeroIndex =-1
            for i in range(0,len(seq)):
                if seq[i] != 0: 
                    nonZeroIndex = i
                    break
            if nonZeroIndex < 0:
                return 0
            signature = self._find_signature(seq)
            list = self.trie.prefix( signature )
            list = filter(lambda x: len(x[0])>len(signature), list)
            item = next(list, None)
            if item != None:
                best = reduce(lambda a, b: a if a[1]>b[1] else b if b[1]>a[1] else a if len(b[0])<=len(a[0]) else b, list, item)
                nextElement = best[0][len(seq)]
                nextElement *= seq[nonZeroIndex]//signature[nonZeroIndex]
                return nextElement
            if len(seq) <= 3: 
                break
            seq = seq[1:]
        return None

    def _findNextAndDerive(self, seq):
        nextElement=self._findNext(seq)
        if nextElement==None:
            der=self._find_derivative(seq)
            if len(der)<=3:
                return None
            nextElement=self._findNextAndDerive(der)
            if nextElement==None:
                return None
            return seq[len(seq)-1]+nextElement
        return nextElement
    
    def predict(self, data):
        sequences = []
        idx = []
        prediction = []
        for seq, ind in tqdm.tqdm(zip(data, data.index), desc="Predicting"):
            der = [int(x) for x in seq]
            next_elm = self._findNextAndDerive(der)
            if next_elm:
                sequences.append(seq)
                idx.append(ind)
                prediction.append(next_elm)
        return sequences, idx, prediction
    
    def __repr__(self):
        params = ', '.join([f"{par}={val}" for par, val in self.params.items()])
        return f"{self.__class__.__name__}({params})"
    
    def save(self, filename='pre_train/trie_ds.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.trie, f)
    
    def load(self, filename='pre_train/trie_ds.pkl'):
        with open(filename, 'rb') as f:
            self.trie = pickle.load(f)
