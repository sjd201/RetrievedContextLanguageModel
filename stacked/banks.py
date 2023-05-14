from numpy import zeros, arange, uint32
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from scipy.sparse import csr_matrix


class banks:

    def __init__(self):

        self.lags = {}
        self.lags["Syn"] = [-5, -4, -3, -2, -1]
        self.lags["T-1"] = [-1]
        self.lags["T-2"] = [-2]
        self.lags["T-3"] = [-3]
        self.lags["T-4"] = [-4]
        self.lags["Para"] = [0]
        self.lags["Output"] = [0]
        self.MaxLag = -min(min(b) for b in self.lags.values())
        self.graph = {"Output": ["Para", "Syn"], "Para": ["T-1", "T-2", "T-3", "T-4"]} 
        self.connections = [("T-1", "Para"), ("T-2", "Para"),("T-3", "Para"),("T-4", "Para"),("Para", "Output"),("Syn", "Output")]
        self.inputBanks = ["Syn", "T-4", "T-3", "T-2", "T-1"]
        self.hiddenBanks = ["Para"] # list these in order they need to be evaluated
        self.outputBanks = ["Output"] 

    def checkConnections(self):
        if "T" not in self.lags:
            print ("No target bank")
        
    def plotBanks(self):
        resultsarray = zeros((self.NumberOfBanks, self.MaxLag))
        for bank, lags in enumerate(self.bankLags):
            for i in lags:
                resultsarray[self.NumberOfBanks - bank-1, -i-1] = self.MaxBankLength/self.bankLengths[bank]
        fig = plt.figure()
        ax = fig.add_subplot(111, axes_class=axisartist.Axes)
        ax.imshow( resultsarray, cmap = 'Blues' , interpolation = 'nearest' )
        #ax.set_yticks(arange(len(labels)), labels=labels)
        #ax.axis["left"].major_ticklabels.set_ha("left")
        #if title:
        #    ax.set_title(title)
        plt.show()

if __name__ == "__main__":
    bs = banks()
    #bs.plotBanks()
