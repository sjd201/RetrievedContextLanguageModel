#!/data/EM/venv3.9/bin/python
from numpy import exp, array, zeros, ones, dot, set_printoptions, log, argmax, clip, sqrt, asarray, multiply, abs, fill_diagonal, quantile, copy, int32, reshape, fromfile, uint32, ndarray, arange, median, int64, log1p, where, float64, argsort, argpartition
from collections import Counter
from scipy.sparse import load_npz, save_npz, lil_matrix, csr_matrix, csc_matrix, coo_matrix, diags
import shelve
from numpy.random import multinomial, choice, binomial
import warnings
import sentencepiece as spm
from itertools import repeat, chain
from functools import reduce
from sys import float_info
from numba import njit
import re
from time import time
import pickle
from os.path import exists
from banks import banks 
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from multiprocessing import Pool
from colored import fg
from scipy.optimize import minimize

warnings.filterwarnings("ignore", message="Changing the sparsity structure of a csr_matrix is expensive.")
set_printoptions(precision=4, edgeitems = 10, linewidth=200, suppress = True)

@njit
def find_first(net, startAt, netCutoff):
    """return the index of the first occurence of an item in vec"""
    for i in range(startAt, net.shape[0]):
        if net[i] > netCutoff: # don't update if prob is too low
            return i
    return -1

def softmax(v):
    expv = exp(v)
    return (exp(v)/expv.sum())

def sparsemax(v):
    if type(v) is lil_matrix:
        v = v.toarray()
    z = v.copy()
    z.sort()
    z = z[::-1]
    k = 1
    total = z[0]
    try:
        while k < z.shape[0] and 1 + k * z[k] > total:
            total += z[k]
            k += 1
    except Exception as e:
        print(z.shape, total.shape)
        print (e)
        exit()
    tau = (total-1)/k
    p = v-tau
    p[p<0] = 0.0
    p = p / (sum(p)+0.0000001) # can we ensure p sums to 1 more efficiently??
    return p

def sigmoid (net):
    return (1./(1+exp(-net)))

def f(c, net = None, target = 1.0):
    p = sigmoid(net+c)
    return ((target - p.sum())**2)

def calcP(net, target = 1):
    res = minimize(f, x0 = 0., args = (net, target), method="Nelder-Mead") # default optimizer (BFGS) often gives poor results - Nelder-Mead seems to do better
    return (sigmoid(net + res.x[0]))

class RC():

    def __init__(self, corpus, verbose = False, numSamples = 50):
        self.corpus = corpus
        self.banks = banks()
        self.verbose = verbose
        self.numSamples = numSamples
        self.temperature = 1.0
        self.buffersize = 3
        self.epsilon = 0.001
        self.colors = [fg("royal_blue_1"), fg('dodger_blue_1'), fg("green"), fg("green_yellow"), fg("light_yellow_3"), fg("dark_orange"), fg("light_red"), fg("red")]
        self.lam = 0.01

        # load tokenizer

        if self.verbose: print ("Loading tokenizer ... ", end="", flush=True)
        self.tokenizer = spm.SentencePieceProcessor(model_file=f"{self.corpus}/token.model")
        if self.verbose: print ("Tokenizer loaded")
        self.underscore = self.tokenizer.encode("_")[1]
        self.V = self.tokenizer.get_piece_size()
        self.excludeTokens = [self.underscore, 0, 1, 2]

        # initialize matrices

        self.N = 0
        self.Cij = {}
        N = self.N
        V = self.V
        e = self.epsilon
        self.Ci = {}
        self.Cj = {}
        self.logPriorOdds = {}
        for b in self.banks.inputBanks:
            self.Ci[b] = zeros(self.V)
        for b in self.banks.hiddenBanks:
            self.Ci[b] = zeros(self.V)
            self.Cj[b] = zeros(self.V)
            self.logPriorOdds[b] = ones(self.V) * - log(V-1)
        for b in self.banks.outputBanks:
            self.Cj[b] = zeros(self.V)
            self.logPriorOdds[b] = ones(self.V) * - log(V-1)
           
        for c in self.banks.connections:
            self.Cij[c] = lil_matrix((self.V, self.V))
        self.Wij = {}
        for c in self.banks.connections:
            self.Wij[c] = lil_matrix((self.V, self.V))

    def decode(self, x):
        # sentencepiece does not work with int16 or int32. Tokens need to be int.
        x = [int(t) for t in x]
        try:
            s = self.tokenizer.decode(x)
        except:
            print (x)
            exit()
        return (s)

    def encode(self, s):
        return self.tokenizer.encode(s)

    def saveParameters(self):
        self.negWeights.tofile(f"{self.corpus}/negWeights")
        self.Gij.tofile(f"{self.corpus}/Gij")
        self.Gj.tofile(f"{self.corpus}/Gj")

    def loadParameters(self):
        if self.verbose: print ("Loading parameters ... ", end="", flush=True)
        self.negWeights = reshape(fromfile(f"{self.corpus}/negWeights"), (self.banks.NumberOfBanks, self.banks.NumberOfBanks))
        self.Gij = reshape(fromfile(f"{self.corpus}/Gij"), (self.banks.NumberOfBanks, self.banks.NumberOfBanks))
        self.Gj = fromfile(f"{self.corpus}/Gj")

        if self.verbose: 
           #self.showParameters()
           print ("Parameters loaded", flush=True)

    def showParameters(self):
        print ("Gij")
        print (self.Gij)
        print ("negWeights")
        print (self.negWeights)
        print ("Gj")
        print (self.Gj)

    def loadWeights(self):

        if self.verbose: print ("Loading weights ... ", flush=True)
        self.C = fromfile(f"{self.corpus}/C", dtype = int64)
        self.C[self.C == 0] = max(self.C)
        self.N = sum(self.C)
        V = self.V
        e = self.epsilon
        self.logPriorOdds = log(self.C + V * e) - log(self.N - self.C + V * (V-1) * e)
        self.logPriorOdds = self.logPriorOdds.reshape((1, self.V))
        self.Wij = {}
        for c in self.banks.connections:
            self.Cij[c] = load_npz(f"{self.corpus}/{c[0]}->{c[1]}.npz")
            Cij = self.Cij[c].tocoo()
            Ci = self.C[Cij.row]
            Cj = self.C[Cij.col]
            Cij.data = log(Cij.data + e) - log(e) - log (Ci - Cij.data + (V-1) * e) + log(Ci + (V-1) * e)
            self.Wij[c] = Cij.tocsr()
        
        if self.verbose: print ("Weights loaded", flush = True)

    def strProbs(self, v):
        counter = Counter(dict(enumerate(v)))
        res  = ""
        for item, value in counter.most_common(10):
            res += f'{self.decode([item])} {value:1.2f} '
        res += "\n"
        return res[0:-1]

    def strVec(self, v, perps = None):
        counters = [Counter() for _ in range(self.banks.NumberOfBanks)]
        res  = ""
        cv = coo_matrix(v)
        for i,j,v in zip(cv.row, cv.col, cv.data):
            if v > 0.001:
                counters[i][j] = v
        for i, counter in enumerate(counters):
            if len(counter) > 0:
                if perps is not None:
                    res += f"{i}[{perps[i]:1.1f}]:"
                else:
                    res += f"{i}: "
                for item, value in counter.most_common(15):
                    #res += f'{self.decode([item])} {item} ({value}) '
                    res += f'{self.decode([item])} {value:1.3f} '
                res += "\n"
        return res[0:-1]

    def displayResults(self, prefix, title = None, showtop = 20):
        toks = self.tokenizer.encode(prefix)
        results = self.iterate(toks + [None] * self.buffersize)
        flatresults = [res.reshape((self.V * self.banks.NumberOfBanks, 1)) for res in results]
        sumresults = reduce(lambda x, y: x + y, flatresults).tocoo()
        displayrows = sorted(zip(sumresults.data, sumresults.row))[-showtop:]
        displayrows = [r for d, r in displayrows]
        displayrows.sort()
        labels = []
        for row in displayrows:
            labels.append(f"{row//self.V}:{self.decode([row%self.V])}")
        resultsarray = zeros((len(displayrows), len(results)))
        m = dict((old, new) for new, old in enumerate(displayrows))
        for j, flatresult in enumerate(flatresults):
            for oi, d in zip(flatresult.row, flatresult.data):
                if oi in m:
                    resultsarray[m[oi], j] = d

        fig = plt.figure()
        ax = fig.add_subplot(111, axes_class=axisartist.Axes)
        ax.imshow( resultsarray, cmap = 'Blues' , interpolation = 'nearest' )
        ax.set_yticks(arange(len(labels)), labels=labels)
        ax.axis["left"].major_ticklabels.set_ha("left")
        if title:
            ax.set_title(title)
        plt.show()

    def initializeInput(self, prefixToks):
        prefixToks = array([t for t in prefixToks])
        activations = {}
        for bank in self.banks.inputBanks:
            indexes = [prefixToks[len(prefixToks) + lag] for lag in self.banks.lags[bank] if len(prefixToks) + lag >= 0 and prefixToks[len(prefixToks)+lag] not in self.excludeTokens]
            activations[bank] = coo_matrix((ones(len(indexes)), (zeros(len(indexes)), indexes)), shape = (1, self.V))
        return activations
        
    def iterate(self, toks):
        prefixToks = toks[0:-1]
        V = self.V
        e = self.epsilon
        activations = self.initializeInput(prefixToks)
        if self.verbose: tmpTime = time()
        net = zeros(self.V)
        for outBank in self.banks.hiddenBanks + self.banks.outputBanks:
            net[:] = self.logPriorOdds[outBank]
            for inBank in self.banks.graph[outBank]:
                for k in range(activations[inBank].nnz):
                    i = activations[inBank].col[k]
                    a = activations[inBank].data[k]
                    net += a * (self.Wij[(inBank, outBank)][i,:].todense().A1 + log(e) - log(self.Ci[inBank][i] + (V - 1) * e)) - self.logPriorOdds[outBank]
            #p = sparsemax(net.flatten())
            p = softmax(net.flatten())
            activations[outBank] = coo_matrix(p, dtype = float64)
        if self.verbose: print (f"Time per iteration = {(time()-tmpTime):1.3f} seconds.")
        return activations

    def update(self, activations, toks):
        target = toks[-1]
        if target == self.underscore:
            return
        print (f"target = {self.decode([int(target)])}")
        self.N += 1
        V = self.V
        e = self.epsilon
        for b in self.banks.inputBanks + self.banks.hiddenBanks:
            inp = activations[b].T
            self.Ci[b] += inp.todense().A1
        for b in self.banks.outputBanks + self.banks.hiddenBanks:
            self.Cj[b][target] += 1
            self.logPriorOdds[b] = log(self.Cj[b] + V * e) - log(self.N - self.Cj[b] + V * (V-1) * e)

        for c in self.banks.connections:
            inp = activations[c[0]].T
            targetVec = lil_matrix((1, self.V))
            targetVec[0, target] = 1.
            newCij = inp * targetVec
            self.Cij[c] += newCij

            updatedCij = lil_matrix((V, V))

            updatedCij[newCij.nonzero()] = self.Cij[c][newCij.nonzero()]
            updatedCij = updatedCij.tocoo()
            Ci = self.Ci[c[0]][updatedCij.row]
            updatedCij.data = log(updatedCij.data + e) - log(e) - log (Ci - updatedCij.data + (V-1) * e) + log(Ci + (V-1) * e)
            updatedCij = updatedCij.tolil()
            self.Wij[c][updatedCij.nonzero()] = updatedCij[updatedCij.nonzero()]

    def generate(self, s, maxNumberOfToks=20):
        toks = self.tokenizer.encode(s)
        toks = [t for t in toks if len(self.decode([t])) > 0] 
        while len(toks) < maxNumberOfToks:  
            results = self.iterate(toks + [self.underscore] * self.buffersize)
            counts = self.getCounts(results)
            j = self.banks.NumberOfBanks - self.buffersize
            toks.append(argmax(counts[j*self.V:(j+1)*self.V]))
            print (f"{self.decode(toks)}", flush=True)
        print (f"Result: {self.decode(toks)}", flush=True)

    def test(self, prefix, topN = 30):
        toks = self.tokenizer.encode(prefix)
        toks = [t for t in toks if len(self.decode([t])) > 0] 
        results = self.iterate(toks)
        print (results)
        self.showSpikeCounts(results)
        print()
        counts = self.getCounts(results)
        most_active = argpartition(-counts, topN)[:topN]
        most_active.sort()
        watchSet = [t for t in most_active if t // self.V > self.banks.NumberOfBanks - 4][0:12]
        #watchSet =  array([self.getIndex(w, b) for w, b in [("LF1", 0), ("LF2", 0), ("Lysander", 4), ("Bellamira", 4), ("John", 4), ("loves", 5), (".", 5), ("Bellamira", 6)]])
        self.showWeights(most_active, watchSet)

    def strTok(self, t, columnWidth = 10):
        s = f"{self.tokenizer.decode([int(t%self.V)])}{t//self.V}"
        if len(s) >= columnWidth-1:
            s = "*" + s[(-columnWidth+2):]
        return (s.rjust(columnWidth))

    def getIndex(self, word, bank):
        return (self.tokenizer.encode(word)[1] + self.V * bank)

    def findMostActive(self, results, topN = 30):
        counts = self.getCounts(results)
        most_active = argpartition(-counts, topN)[:topN]
        most_active.sort()
        return (most_active)

    def showWeights(self, toks, watchSet, columnWidth=15):
        print (" "*columnWidth, end="")
        for t1 in watchSet:
            s = self.strTok(t1, columnWidth)
            print (s, end="")
        print ()
        for line, t1 in enumerate(toks):
            print (self.colors[line%len(self.colors)], end = "")
            print (self.strTok(t1, columnWidth), end = "")
            for j, t2 in enumerate(watchSet):
                if t1 % self.V == t2 % self.V:
                    w = self.negWeights[t1 // self.V, t2 // self.V]
                else:
                    w = self.logLR[t1, t2]
                if w != 0.:
                    print (f"{w:1.3f}".rjust(columnWidth), end = "")
                else:
                    print (" " * columnWidth, end = "")
            print()

    def showSpikeTrains(self, results):
        allActiveIndices = set()
        counts = zeros(self.V*self.banks.NumberOfBanks)
        for result in results:
            allActiveIndices = allActiveIndices | set(list(result.col))
        allActiveIndices = list(allActiveIndices)
        allActiveIndices.sort()
        lastBank = 0
        for index in allActiveIndices:
            newBank = index // self.V
            if newBank != lastBank:
                for i in range(lastBank, newBank):
                    print ("-" * (15 + len(results)))
                lastBank = newBank
            print (f"{self.colors[index//V]}{self.decode([index%V]).rjust(15)}", end = " ")
            for result in results:
                 result = result.tocsr()
                 if result[0, index] > 0.:
                     print ("|", end="")
                     counts[index] += 1
                 else:
                     print (" ", end="")
            print(f" {counts[index]:1.0f}")
        print (fg("white"))

    def showSpikeCounts(self, results, topN = 10):
        counts = self.getCounts(results)
        for b in range(self.banks.NumberOfBanks):
            bankcounts = counts[b*self.V:(b+1)*self.V]
            print (f"{b}: ", end = "")
            items = argsort(bankcounts)[::-1]
            i = 0
            while bankcounts[items[i]] > 0 and i < topN:
                print (fg("white"), end = "")
                print (f"{self.decode([items[i]])} ", end = " ")
                print (fg("purple_4a"), end = "")
                print (f"{bankcounts[items[i]]:1.0f}", end = " ")
                print (fg("white"), end = "")
                i += 1
            print ()

    def getCounts(self, results):
        counts = zeros(self.V*self.banks.NumberOfBanks)
        for result in results:
            counts[result.col] += 1
        return counts

    def showActivations(self, results):
        for bank in results.keys():
            print (bank + ": ", end = "")
            inds = argsort(results[bank].data)[::-1][0:10]
            for k in inds:
                col = results[bank].col[k]
                val = results[bank].data[k]
                print (f"{self.decode([results[bank].col[k]])} {val:1.2f}", end = " ")
            print ()

    def show(self, s):
        toks = self.tokenizer.encode(s)
        toks = [t for t in toks if len(self.decode([t])) > 0]
        for i in range(2, len(toks)+1):
            print ("========================================================")
            print (i, self.decode(toks[0:i]))
            results = self.iterate(toks[0:i]) # don't include last token
            self.showActivations(results)
            self.update(results, toks[0:i])

        for c in self.banks.connections:
            print (c)
            C = self.Cij[c].tocoo()
            for k in range(C.nnz):
                r = C.row[k]
                c = C.col[k]
                d = C.data[k]
                print (f"{self.decode([r])}{r}-{self.decode([c])}{c} {d:1.2f}", end = " ")
            print()

    def learnPattern(self, toks):

        inp = self.initializeInput(toks)
        inpcoo = inp.tocoo()
        net = zeros(self.V)
        initialC = zeros(self.banks.NumberOfBanks)

        net = zeros(self.banks.NumberOfBanks* self.V)
        for bj in range(self.banks.NumberOfBanks):
            net[bj*self.V:(bj+1)*self.V] += self.Gj[bj] * self.logPriorOdds[bj*self.V:(bj+1)*self.V]
        inpcoo = inp.tocoo()
        for i in inpcoo.col:
            bi = i // self.V
            for bj in range(self.banks.NumberOfBanks):
                net[bj*self.V:(bj+1)*self.V] += (self.Gij[bi, bj] * self.logLR[i,bj*self.V:(bj+1)*self.V]).todense().A1
                if bi != bj:
                    net[i%self.V+bj*self.V] += self.negWeights[bi, bj] 
        p = zeros(self.V*self.banks.NumberOfBanks)
        for b in range(self.banks.NumberOfBanks):
            target = self.banks.bankLengths[b]
            p[(b*self.V):((b+1)*self.V)] = calcP(net[(b*self.V):((b+1)*self.V)], initialC, target, bank = b)
        delta = (inp-p).A1
        for bj in range(self.banks.NumberOfBanks):
            self.Gj[bj] += self.lam * (delta[bj*self.V:(bj+1)*self.V] * self.logPriorOdds[bj*self.V:(bj+1)*self.V]).sum()
        inpcoo = inp.tocoo()
        for i in inpcoo.col:
            bi = i // self.V
            for bj in range(self.banks.NumberOfBanks):
                self.Gij[bi, bj] += self.lam * (delta[bj*self.V:(bj+1)*self.V] * self.logLR[i,bj*self.V:(bj+1)*self.V].todense().A1).sum()
                if bi != bj:
                    self.negWeights[bi, bj] += self.lam * delta[i%self.V+bj*self.V] 

        # calc error

        totalLog = zeros(self.banks.NumberOfBanks)
        counts = zeros(self.banks.NumberOfBanks)
        for j in inpcoo.col:
            bj = j // self.V
            totalLog[bj] += log(p[j])
            counts[bj] += 1
        return (totalLog, counts)

    def learn(self, numSamples=None):
        toks = fromfile(self.corpus+"/train.tok", dtype=uint32)
        if numSamples is not None:
            toks = toks[0:numSamples]

        totalLog = zeros(self.banks.NumberOfBanks)
        counts = zeros(self.banks.NumberOfBanks)
        for i in range(len(toks)):
            currentLog, currentCounts = self.learnPattern(toks[i:(i+self.banks.MaxLag)])
            totalLog += currentLog
            counts += currentCounts
            if (i+1) % 100 == 0 or i == len(toks) - 1:
                self.saveParameters()
                print (f"{i}/{len(toks)} ", end = "")
                print (totalLog/counts, flush=True)
                totalLog = zeros(self.banks.NumberOfBanks)
                counts = zeros(self.banks.NumberOfBanks)
                print ("Gij")
                print (self.Gij)
                print ("negWeights")
                print (self.negWeights)
                print ("Gj")
                print (self.Gj)

        self.saveParameters()

    def learnParallel(self, targetBanks):
        print (f"Training from token file {self.corpus}/corpus.tok")
        print (f"Initial learning rate = {self.lam}")
        print (f"Vocab size = {self.V}")
        print (f"Number of CPUs = {self.NumberOfCPUs}")
        with Pool(min(len(targetBanks), self.NumberOfCPUs)) as p:
            p.map(self.learnBank, targetBanks)

if __name__ == "__main__":
    
    s = RC("tiny", verbose=True)
    s.loadWeights()
    s.learn()
    s.showParameters()
