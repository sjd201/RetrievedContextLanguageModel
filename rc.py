#!/data/EM/venv3.9/bin/python
from numpy import exp, array, zeros, ones, dot, set_printoptions, log, argmax, clip, sqrt, asarray, multiply, abs, fill_diagonal, quantile, copy, int32, reshape, fromfile, uint32, ndarray, arange, median, int64, log1p, where, float64, argsort, argpartition, mean, std
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
from colored import fg, bg
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

def sigmoid (net):
    return (1./(1+exp(-net)))

def f(c, net, target = 1.0):
    p = sigmoid(net+c)
    return ((target - p.sum())**2)

def calcPnew(net, mask, target = 1):
    net[mask] = min(net)
    meannet = mean(net)
    stdnet = std(net)
    p = sigmoid((net-meannet)/stdnet-2.)
    return (p)

def calcP(net, mask, target = 1):
    net[mask] = min(net)
    meannet = mean(net)
    res = minimize(f, x0 = -meannet, args = (net, target), method="Nelder-Mead") # default optimizer (BFGS) often gives poor results - Nelder-Mead seems to do better
    p = sigmoid(net + res.x[0])
    return (p)

class RC():

    def __init__(self, corpus, verbose = False, numSamples = 50):
        self.corpus = corpus
        self.banks = banks([6, 1, 1, 1, 1, 1, 1])
        self.verbose = verbose
        self.numSamples = numSamples
        self.temperature = 1.0
        self.buffersize = 3
        self.colors = [fg("royal_blue_1"), fg('dodger_blue_1'), fg("green"), fg("green_yellow"), fg("light_yellow_3"), fg("dark_orange"), fg("light_red"), fg("red")]
        self.setParams()
 
        self.lam = 0.01

        # load tokenizer

        if self.verbose: print ("Loading tokenizer ... ", end="", flush=True)
        self.tokenizer = spm.SentencePieceProcessor(model_file=f"{self.corpus}/token.model")
        self.underscore = self.tokenizer.encode("_")[1]
        self.V = self.tokenizer.get_piece_size()
        #self.banks = zeros(self.V * self.banks.NumberOfBanks, dtype=int)
        #for b in range(self.banks.NumberOfBanks):
        #    self.banks[b*self.V:(b+1)*self.V] = b
        if self.verbose: print ("Tokenizer loaded", flush = True)
        if self.verbose: print ("Loading weights ... ", end="", flush=True)
        self.loadWeights()
        if self.verbose: print ("Weights loaded", flush = True)

    def setParams(self, Gij = 1.0, negWeight = -4.):

        self.Gij = ones((self.banks.NumberOfBanks, self.banks.NumberOfBanks))
        for b1 in range(self.banks.NumberOfBanks):
            for b2 in range(self.banks.NumberOfBanks):
                self.Gij[b1, b2] = Gij/self.banks.bankLengths[b1]
        self.Gj = ones(self.banks.NumberOfBanks)
        self.negWeights = ones((self.banks.NumberOfBanks, self.banks.NumberOfBanks)) * negWeight
        for b in range(self.banks.NumberOfBanks):
            self.negWeights[b, b] = 0.
            self.negWeights[0, b] = 0.
            self.negWeights[b, 0] = 0.
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
        #data = fromfile(f"{self.corpus}/logLR.data")
        #indptr = fromfile(f"{self.corpus}/logLR.indptr", dtype=int32)
        #indices = fromfile(f"{self.corpus}/logLR.indices", dtype=int32)
        #self.logLR = csr_matrix((data, indices, indptr), shape = (self.V * self.banks.NumberOfBanks, self.V * self.banks.NumberOfBanks))
        self.W = load_npz(f"{self.corpus}/W.npz")
        #LF1 = self.encode("LF1")[1]
        #LF2 = self.encode("LF2")[1]
        #print (LF1)
        #print (f"{LF1} {LF2} {self.decode([LF1])} {self.decode([LF2])} W = {self.W[LF1, LF2]}")
        self.Cij = load_npz(f"{self.corpus}/Cij.npz")

        self.logElogCi = fromfile(f"{self.corpus}/logElogCi")
        self.logPriorOdds = fromfile(f"{self.corpus}/logPriorOdds")
        self.C = fromfile(f"{self.corpus}/C")
        self.mask = self.C == 0
        
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
        current = []
        for bank in range(self.banks.NumberOfBanks):
            indexes = [len(prefixToks) + lag for lag in self.banks.bankLags[bank] if len(prefixToks) + lag >= 0 and prefixToks[len(prefixToks)+lag] != self.underscore]
            current.extend(set(t + self.V*bank for t in prefixToks[indexes]))
        inp = coo_matrix((ones(len(current)), (zeros(len(current)), current)), shape = (1, self.banks.NumberOfBanks*self.V))
        return inp
        
    def iterate(self, prefixToks):
        inp = self.initializeInput(prefixToks).tocoo()
        current = self.initializeInput(prefixToks).col
        initialC = zeros(self.banks.NumberOfBanks)
        results = []
        tmpTime = time()
        for iteration in range(self.numSamples):
            net = zeros(self.banks.NumberOfBanks* self.V)
            for bj in range(self.banks.NumberOfBanks):
                net[bj*self.V:(bj+1)*self.V] += self.Gj[bj] * self.logPriorOdds[bj*self.V:(bj+1)*self.V]
            for i in inp.col:
                bi = i // self.V
                for bj in range(self.banks.NumberOfBanks):
                    net[bj*self.V:(bj+1)*self.V] += self.Gij[bi, bj] * ((self.W[i,bj*self.V:(bj+1)*self.V]).todense().A1 + self.logElogCi[i] - self.logPriorOdds[bj*self.V:(bj+1)*self.V])
                    if bi != bj:
                        net[i%self.V+bj*self.V] += self.negWeights[bi, bj] 
            p = zeros(self.V*self.banks.NumberOfBanks)
            for b in range(self.banks.NumberOfBanks):
                target = self.banks.bankLengths[b]
                if b > 3:
                    target = 4
                p[(b*self.V):((b+1)*self.V)] = calcP(net[(b*self.V):((b+1)*self.V)], self.mask[(b*self.V):((b+1)*self.V)], target)
            m = binomial(n=1, p = p, size=len(p))
            for t in current: # make sure input remains active
                m[t] = 1
            inp = coo_matrix(m, dtype = float64)
            inp.eliminate_zeros()
            results.append(inp)
        if self.verbose:
            print (f"Time per iteration = {(time()-tmpTime)/(self.numSamples+0.00001):1.3f} seconds.")
        return results

    def look(self, b2, i2):
        tok = self.decode([i2])
        contrib = Counter()
        negcontrib = Counter()
        for b1 in range(self.banks.NumberOfBanks):
            if b1 == b2:
                p = sparsemax(self.net[b2])[i2]
                print (f"{b2}{tok} {p:1.3f}: ", end="")
            else:
                vec = self.slots[b1, :] + self.fixed[b1,:]
                vec = vec.tocoo()
                for i1, d in zip(vec.col, vec.data):
                    if d > 0:
                        net = self.slots[b1, i1] * self.Wij[b1][b2][i1, i2]
                        net += self.fixed[b1, i1] * self.Wij[b1][b2][i1, i2]
                        contrib[f"{b1}{self.decode([i1])}"] = net
                        #negcontrib[f"{b1}{self.decode([i1])}"] = -net
            
        for x, n in contrib.most_common(6):
            print (f"{x} ({n:1.3f})", end = " ", flush=True)
            #print (f"{x} {n:1.2f}", end = " ", flush=True)
        print (" ... ", end="")
        #for x, n in negcontrib.most_common(3)[::-1]:
            #print (f"{x}{-n:1.2f}", end = " ", flush=True)
        #    print (f"{x} ({-n:1.3f})", end = " ", flush=True)
        print ()

    def lookBank(self, bank):
        vec = self.slots[bank, :]
        vec = vec.tocoo()
        for i1, d in zip(vec.col, vec.data):
           if d > 0:
               self.look(bank, i1)

    def samplesFromStr(self, s):
        toks = self.tokenizer.encode(s)
        f = self.iterate(toks, self.numIterations)
        for i in range(len(f)):
            print (self.strVec(f[i]))
        return f

    def bestSampleFromStr(self, s):
        toks = self.tokenizer.encode(s)
        f = self.iterate(toks)
        c = Counter(tuple(f[i,:]) for i in range(f.shape[0]))
        best = c.most_common(1)[0][0]
        print (self.strVec(best))

    def EndOfSentence(self, toks):
        return 764 in toks or 198 in toks

    def getLastFromResult(self, result):
        res = []
        for j in range(self.banks.NumberOfBanks-self.buffersize, self.banks.NumberOfBanks):
            res.append(result[j*self.V:(j+1)*self.V].argmax())
        return tuple(res)
        
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

    def test(self, prefix = "who is Bellamira loved by? _ _ _"):
        toks = self.tokenizer.encode(prefix)
        toks = [t for t in toks if len(self.decode([t])) > 0] 
        self.results = self.iterate(toks)

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

    def watch(self, st="LF10 LF20 Lysander4 Bellamira4 loves5"):
        st = st.split()
        st = [(s[0:-1], int(s[-1])) for s in st]
        watchSet =  array([self.getIndex(w, b) for w, b in st])
        return (watchSet)

    def showWeights(self, watchSet = None, topN = 30, columnWidth=12):
        counts = self.getCounts(self.results)
        most_active = argpartition(-counts, topN)[:topN]
        most_active.sort()
        if watchSet is None:
            watchSet = [t for t in most_active if t // self.V > self.banks.NumberOfBanks - 4][0:8]

        print (" "*columnWidth, end="")
        for t1 in watchSet:
            print (self.colors[t1//self.V], end = "")
            s = self.strTok(t1, columnWidth)
            print (s, end="")
        print ()
        for line, t1 in enumerate(most_active):
            print (self.colors[t1//self.V], end = "")
            print (self.strTok(t1, columnWidth), end = "")
            for j, t2 in enumerate(watchSet):
                if t1 % self.V == t2 % self.V:
                    w = self.W[t1, t2] + self.logElogCi[t1] - self.logPriorOdds[t2]
                    w += self.negWeights[t1 // self.V, t2 // self.V]
                else:
                    w = self.W[t1, t2] + self.logElogCi[t1] - self.logPriorOdds[t2]
                    if False:
                        Cij = self.Cij[t1, t2]
                        Ci = self.C[t1]
                        Cj = self.C[t2]
                        N = 417
                        V = 39
                        e = 0.001
                        w2 = log(Cij + e) - log(Cj + V * e) + log(N - Cj + V * (V-1) * e) - log(Ci - Cij + (V-1) * e)
                        print (f"{self.decode([t1%self.V])}{t1//V} {t1} {self.decode([t2%self.V])}{t2//V} {t2} N = {N} V = {V} e = {e} Cij = {Cij} Ci = {Ci} Cj = {Cj}")
                        print ("W", self.W[t1, t2], log(Cij + e) - log(e) - log(Ci - Cij + (V-1) * e) + log(Ci+(V-1) * e))
                        print ("logElogCi", t1, Ci, self.logElogCi[t1], log(e) - log(Ci+(V-1) * e))
                        print ("logPriorOdds", self.logPriorOdds[t2], log(Cj + V * e) - log(N - Cj + V * (V-1) * e))
                        print("logLR", w, w2)
                        print ()
                        #print (w, w2)

                if w != 0.:
                    print (f"{w:1.3f}".rjust(columnWidth), end = "")
                else:
                    print (" " * columnWidth, end = "")
            print()
        print (fg("white"))

    def showSpikeTrains(self):
        allActiveIndices = set()
        counts = zeros(self.V*self.banks.NumberOfBanks)
        for result in self.results:
            allActiveIndices = allActiveIndices | set(list(result.col))
        allActiveIndices = list(allActiveIndices)
        allActiveIndices.sort()
        print()
        lastBank = 0
        for index in allActiveIndices:
            newBank = index // self.V
            if newBank != lastBank:
                for i in range(lastBank, newBank):
                    print (" " * (15 + len(self.results)))
                lastBank = newBank
            print (f" {self.colors[index//self.V]}{self.decode([index%self.V]).rjust(15)}", end = " ")
            for result in self.results:
                 result = result.tocsr()
                 if result[0, index] > 0.:
                     print ("|", end="")
                     counts[index] += 1
                 else:
                     print (" ", end="")
            print(f" {counts[index]:1.0f}")
        print (fg("white"))
        print()

    def showSpikeCounts(self, topN = 10):
        counts = self.getCounts(self.results)
        print (bg("black"))
        for b in range(self.banks.NumberOfBanks):
            bankcounts = counts[b*self.V:(b+1)*self.V]
            print (f" {self.colors[b]}{b}: ", end = "")
            items = argsort(bankcounts)[::-1]
            i = 0
            while bankcounts[items[i]] > 0 and i < topN:
                #print (fg(255-int(bankcounts[items[i]]/self.numSamples*22)), end = "")
                print (f"{self.decode([items[i]])} ", end = " ")
                #print (fg("purple_4a"), end = "")
                #print (f"{bankcounts[items[i]]:1.0f}", end = " ")
                #print (255-int(bankcounts[items[i]]/self.numSamples*20), bankcounts[items[i]], end = "")
                #print (fg("white"), end = "")
                i += 1
            print ()
        print (fg("white"))

    def getCounts(self, results):
        counts = zeros(self.V*self.banks.NumberOfBanks)
        for result in results:
            counts[result.col] += 1
        return counts

    def show(self, prefix):
        toks = self.tokenizer.encode(prefix)
        toks = [t for t in toks if len(self.decode([t])) > 0]
        results = self.iterate(toks)
        counts = results.sum(axis=0).A1
        allActiveIndices = where(counts > 0)[0].tolist()
        lastBank = 0
        for index in allActiveIndices:
            newBank = index // self.V
            if newBank != lastBank:
                for i in range(lastBank, newBank):
                    print ("-" * (15 + results.shape[1]))
                lastBank = newBank
            print (f"{self.colors[index//self.V]}{self.decode([int(index%self.V)]).rjust(15)}", end = " ")
            for result in range(results.shape[0]):
                 if results[result, index] == 1:
                     print ("|", end="")
                     counts[index] += 1
                 else:
                     print (" ", end="")
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
            p[(b*self.V):((b+1)*self.V)] = calcP(net[(b*self.V):((b+1)*self.V)], mask, target)
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
