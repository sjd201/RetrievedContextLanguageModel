#!/data/EM/venv3.9/bin/python
from numpy import exp, array, zeros, ones, dot, set_printoptions, log, argmax, clip, sqrt, asarray, multiply, abs, max, fill_diagonal, quantile, copy, int32, reshape, fromfile, uint32, ndarray, arange, median
from collections import Counter
from scipy.sparse import load_npz, save_npz, lil_matrix, csr_matrix, csc_matrix, coo_matrix, diags
import shelve
from numpy.random import multinomial
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
from banks import bankLags, relativeLags, strRelativeLag, relativeLag, bankLengths, NumberOfBanks, MaxLag, MaxBankLength 
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from multiprocessing import Pool

warnings.filterwarnings("ignore", message="Changing the sparsity structure of a csr_matrix is expensive.")
set_printoptions(precision=4, edgeitems = 10, linewidth=200, suppress = True)


def softmax(v):
    if type(v) is lil_matrix:
        v = v.toarray()
    v = v-v.mean() # to avoid overflow errors
    #print ("v = ")
    #print (v)
    es = exp(v)
    output = es/(es.sum()+0.00000001)
    return output

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
    p = p /sum(p) # can we ensure p sums to 1 more efficiently??
    return p

class RC():

    def __init__(self, corpus, verbose = False, numIterations = 50):
        self.corpus = corpus
        self.verbose = verbose
        self.numIterations = numIterations
        self.lam = 0.00001
        self.temperature = 0.5
        self.buffersize = 1
        self.Wij = [[None for b in range(NumberOfBanks)] for b in range(NumberOfBanks)]

        # load tokenizer

        if self.verbose: print ("Loading tokenizer ... ", end="", flush=True)
        params = shelve.open(f"{self.corpus}/params")
        self.tokenizer = spm.SentencePieceProcessor(model_file=params["sentencepiece model"])
        self.V = self.tokenizer.get_piece_size()
        if self.verbose: print ("Tokenizer loaded")

        # initialize parameters

        self.GPost = zeros(NumberOfBanks)
        self.G = zeros((NumberOfBanks, NumberOfBanks, self.V))

    def showParameters(self):
        print ("G")
        print (self.G)
        print ("GPost")
        print (self.GPost)

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

    def saveParameters(self, bank, G, GPost):
        G.tofile(f"{self.corpus}/G{bank}")
        GPost.tofile(f"{self.corpus}/GPost{bank}")

    def loadParameters(self, bank = None):
        if self.verbose: print ("Loading parameters ... ", end="", flush=True)
        if bank:
            banks = [bank]
        else:
            banks = range(NumberOfBanks)
        for b in banks:
            #print (f"loading bank {b}")
            #print (reshape(fromfile(f"{self.corpus}/G{b}"), (NumberOfBanks, self.V)))
            self.G[:, b, :] = reshape(fromfile(f"{self.corpus}/G{b}"), (NumberOfBanks, self.V))
            self.GPost[b] = fromfile(f"{self.corpus}/GPost{b}")

        if self.verbose: 
           #self.showParameters()
           print ("Parameters loaded", flush=True)

    def maxCountIndexes(self):
        for i in range(NumberOfBanks):
            for j in range(NumberOfBanks):
                c = self.Cij[(i*self.V):((i+1)*self.V), (j*self.V):((j+1)*self.V)].tocoo()
                print (f"max row = {c.row.max()} max col = {c.col.max()}")

    def buildLogPFromLags(self):
        # only slots done here context is done separately
        for b1 in range(NumberOfBanks):
            for b2 in range(NumberOfBanks): 
                lag = abs(b1-b2)
                if b2 > b1:
                    self.logP[b1][b2] = self.logPLags[lag]
                elif b1 > b2:
                    self.logP[b1][b2] = self.logPLagsTranspose[lag]
                else:
                    pass 

    def loadWeights(self):
        if self.verbose: print ("Loading weights ... ", flush=True)
        WijRelative = {}
        WijRelativeTranspose = {}
        for rl in relativeLags:
            s = strRelativeLag(rl)
            if self.verbose: print (f"lag {s}", flush=True, end=" ")
            WijRelative[s] = load_npz(self.corpus+f"/WijRelative{s}.npz")
            #print (WijRelative[s].min(), WijRelative[s].max())
            WijRelativeTranspose[s] = load_npz(self.corpus+f"/WijRelative{s}T.npz") 
            # need to convert to csr to make extraction of rows efficient - note this does increase memory usage
            # cheaper to load rather than transpose on the fly when the corpus is large
        if self.verbose: print ()

        for b1 in range(len(bankLags)):
            for b2 in range(len(bankLags)):
                if b2 > b1:
                    s = strRelativeLag(relativeLag(bankLags[b1], bankLags[b2]))
                    self.Wij[b1][b2] = WijRelative[s]
                if b2 < b1:
                    s = strRelativeLag(relativeLag(bankLags[b2], bankLags[b1]))
                    self.Wij[b1][b2] = WijRelativeTranspose[s]

        self.logCj = fromfile(self.corpus+f"/logCi")
        if self.verbose: print ("Weights loaded", flush = True)

    def showWeights(self, toks):
        for b1, tok in enumerate(toks):
            for b2, tok2 in enumerate(toks):
                if b1 != b2:
                    self.showWeight(tok, b1, tok2, b2)

    def showWeight(self, i1, b1, i2, b2, verbose = True):
        if b2 > b1:
            tCij = self.Cij[b2-b1][i1, i2]
        else:
            tCij = self.Cij[b1-b2][i2, i1]
        tCi = self.Ci[i1]
        tCj = self.Ci[i2]
        V = self.V
        N = self.N
        #G = self.G[b2-b1+NumberOfBanks-1]
        G = self.G[b1, b2]
        GPost = self.GPost
        word1 = self.decode([i1])
        word2 = self.decode([i2])
        print (f"{word1}{b1} {word2}{b2}: ", end="")
        if verbose:
            print (f"Indices {i1} {i2}")
            print (f"Cij = {tCij}")
            print (f"Ci = {tCi}")
            print (f"Cj = {tCj}")
            print (f"logP[{word1}{b1}, {word2}{b2}] = {self.logP[b1][b2][i1, i2]:1.3f} == {G*(log(tCij+e)-log(e)):1.3f}")
            print (f"logPriors[{word2}{b2}] = {GPost * self.logPriors[i2]:1.3f} == {GPost*log(tCj+V*e)-log(N+V*V*e):1.3f}")
            print (f"logCj[{word2}{b2}] = {G*self.logCj[i2]:1.3f} == {G*-log(tCj+V*e):1.3f}")
        print (f"{self.logP[b1][b2][i1, i2] + G*self.logCj[i2] + G * log(e):1.3f}")

    def strProbs(self, v):
        counter = Counter(dict(enumerate(v)))
        res  = ""
        for item, value in counter.most_common(10):
            res += f'{self.decode([item])} {value:1.2f} '
        res += "\n"
        return res[0:-1]

    def strVec(self, v, perps = None):
        counters = [Counter() for _ in range(NumberOfBanks)]
        res  = ""
        cv = coo_matrix(v)
        for i,j,v in zip(cv.row, cv.col, cv.data):
            if v > 0.001:
                counters[i][j] = v
        for i, counter in enumerate(counters):
            if perps:
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
        flatresults = [res.reshape((self.V * NumberOfBanks, 1)) for res in results]
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

    def fillSlotsFromPrefix(self, prefixToks):
        result = csr_matrix((NumberOfBanks, self.V), dtype=float)
        for b in range(NumberOfBanks):
            for l in bankLags[b]:
                if len(prefixToks) + l >= 0 and prefixToks[l]:
                    result[b, prefixToks[l]] += 1.0 / bankLengths[b]
        return result
        
    def iterate(self, prefixToks, showDots = False):
        result = []
        self.fixed = self.fillSlotsFromPrefix(prefixToks)  
        self.slots = csr_matrix((NumberOfBanks, self.V), dtype=float)
        result.append(self.fixed.copy())
        if self.verbose: 
            print ("fixed before initialization")
            print (self.strVec(self.fixed))
        self.net = zeros((NumberOfBanks, self.V))
        fixedBanks = range(NumberOfBanks-self.buffersize)
        allBanks = range(NumberOfBanks)

        # initialise self.net

        for b2 in allBanks:
            self.net[b2] = self.logCj 
            for b1 in allBanks:
                if b1 != b2:
                    self.net[b2] += self.fixed[b1,:] * self.Wij[b1][b2] 
                    self.net[b2] -= self.fixed[b1,:].sum() * self.logCj 

            p = softmax(self.temperature * self.net[b2])
            m = multinomial(self.numslotsamples, p) * 1.0
            self.slots[b2,:] = csr_matrix(m) / m.sum()

        if self.verbose: 
            print ("fixed + slots after initialization")
            print (self.strVec(self.fixed + self.slots))
        
        # update nets based on generated samples

        for b2 in allBanks:
            for b1 in allBanks:
                if b1 != b2:
                    self.net[b2] += self.slots[b1,:] * self.Wij[b1][b2] 
                    self.net[b2] -= self.slots[b1,:].sum() * self.logCj 

        # collect Gibbs samples

        tmpTime = time()
        for iteration in range(self.numIterations):
            if self.verbose: print (f"Iteration {iteration}")
            perps = []
            for b1 in allBanks:
                p = softmax(self.temperature * self.net[b1])
                m = multinomial(self.numslotsamples, p)
                newslot = csr_matrix(m)/ m.sum()
                perps.append(exp(-(newslot* log(p)).sum()/self.numslotsamples))
                diff = newslot - self.slots[b1,:]
                diff.eliminate_zeros()
                for b2 in allBanks:
                    G = self.G[b1, b2]
                    if b1 != b2:
                        self.net[b2] += diff * self.Wij[b1][b2] 
                self.slots[b1] = newslot

            self.slots.eliminate_zeros()
            result.append(self.slots.copy())
            #if self.verbose: print ("Perps: " + " ".join(f"{x:1.3f}" for x in perp))
            if self.verbose: 
                print (self.strVec(self.fixed + self.slots, perps = perps))
                print()
            if showDots: print (".", end="", flush=True)
        if showDots: print()
        if self.verbose:
            print (f"Time per iteration = {(time()-tmpTime)/(self.numIterations+0.00001):1.3f} seconds.")
        return result

    def look(self, b2, i2):
        tok = self.decode([i2])
        contrib = Counter()
        negcontrib = Counter()
        for b1 in range(NumberOfBanks):
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
        #print (f"logPriors = {self.GPost[b2] * self.logCj[i2]:1.3f}")

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
        for j in range(NumberOfBanks-self.buffersize, NumberOfBanks):
            res.append(result[j].argmax())
        return tuple(res)
        
    def generate(self, s, maxNumberOfToks=20):
        toks = self.tokenizer.encode(s)
        while len(toks) < maxNumberOfToks:  
            f = self.iterate(toks + [None] * self.buffersize)
            c = Counter(self.getLastFromResult(f[i]) for i in range(len(f)))
            best = c.most_common(1)[0][0]
            toks += best
            print (self.decode(toks), flush=True)
        print ("Result: ", self.decode(toks), flush=True)

    def test(self, prefix):
        toks = self.tokenizer.encode(prefix)
        f = self.iterate(toks + [None] * self.buffersize)
        if len(f) > 0:
            c = Counter(self.getLastFromResult(f[i]) for i in range(len(f)))
            best = c.most_common(1)[0]
            toks += best[0]
            print (f"Result: {self.decode(toks)} ({best[1]}/{len(f)})", flush=True)

    def learnPattern(self, toks, targetBank, G, GPost):

        self.slots = self.fillSlotsFromPrefix(toks)
        net = zeros(self.V)
        inp = zeros((NumberOfBanks, self.V))
        fixedBanks = range(NumberOfBanks-self.buffersize)
        allBanks = range(NumberOfBanks)
        totalLog = 0.0
        deltaG = zeros((NumberOfBanks, self.V))
        deltaGPost = 0.0

        net = GPost * self.logCj
        for b1 in allBanks:
            if b1 != targetBank:
                net += self.slots[b1,:] * diags(G[b1, :]) * self.logCij[b1][targetBank] 
        p = softmax(net)
        delta = -p
        for outUnit in toks[bankLags[targetBank]]:
            delta[0, outUnit] += 1.
            totalLog += log(p[0, outUnit]) / bankLengths[targetBank]
            for b1 in allBanks:
                if b1 != targetBank:
                    for inUnit in toks[bankLags[b1]]:
                        #print ("deltaG ", deltaG[b1, inUnit])
                        #print ("slots ", self.slots[b1, inUnit])
                        #print ("delta ", delta.shape)
                        #print ("logCij ", self.logCij[b1][targetBank][inUnit, :].shape)
                        #print ("dot ", self.logCij[b1][targetBank][inUnit, :].dot(delta.T))
                        deltaG[b1, inUnit] += self.slots[b1, inUnit] * self.logCij[b1][targetBank][inUnit, :].dot(delta.T)
            delta[0, outUnit] -= 1.
        deltaGPost += dot(delta, self.logCj)
        
        return totalLog, deltaG, deltaGPost

    def learnBank(self, targetBank):
        print (f"Learning target bank {targetBank}")
        toks = fromfile(self.corpus+"/corpus.tok", dtype=uint32)
        print ("Loaded training corpus")
        # make parameters

        GPost = 0.0
        G = zeros((NumberOfBanks, self.V))
        lam = self.lam
        totalLoss = 0.0
        currentLoss = 0.0
        currentCounter = 0
        for i in range(len(toks)):
            tmpTotalLoss, deltaG, deltaGPost = self.learnPattern(toks[i:(i+MaxLag)], targetBank, G, GPost)
            G += lam * deltaG 
            GPost += lam * deltaGPost 
            totalLoss += tmpTotalLoss 
            currentLoss += tmpTotalLoss 
            currentCounter += 1
            if (i+1) % 1000 == 0:
                self.saveParameters(targetBank, G, GPost)
                print (f"{targetBank} {i+1} / {len(toks)}: logLik = {currentLoss/currentCounter}", flush=True)
                currentCounter = 0
                currentLoss = 0.0
        self.saveParameters(targetBank, G, GPost)

    def learnParallel(self, targetBanks):
        print (f"Training from token file {self.corpus}/corpus.tok")
        print (f"Initial learning rate = {self.lam}")
        print (f"Vocab size = {self.V}")
        print (f"Number of CPUs = {self.NumberOfCPUs}")
        with Pool(min(len(targetBanks), self.NumberOfCPUs)) as p:
            p.map(self.learnBank, targetBanks)

if __name__ == "__main__":
    
    s = RC("small", verbose=True)
    s.loadWeights()
    s.NumberOfCPUs = 40
    s.lam = 0.0001
    s.learnParallel([7])
    s.loadParameters(7)
    print ("G after load")
    print (s.G[:, 7, :])
