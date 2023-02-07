from numpy import zeros, arange, uint32
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
#import sentencepiece as spm
from scipy.sparse import csr_matrix

#tokenizer = spm.SentencePieceProcessor(model_file="large.model")
#V = tokenizer.get_piece_size()

def strRelativeLag(rl):
    if min(rl[0]) == max(rl[0]):
        origin = f"{min(rl[0])}"
    else:
        origin = f"{min(rl[0])}-{max(rl[0])}"
    if min(rl[1]) == max(rl[1]):
        destination = f"{min(rl[1])}"
    else:
        destination = f"{min(rl[1])}-{max(rl[1])}"
    return f"{origin}_{destination}"

def relativeLag(al1, al2):
# convert two absolute banks to a relative lag
    m = min(min(al1), min(al2))
    return (frozenset(b - m for b in al1), frozenset(b - m for b in al2))

def l(start, end):
    return(list(range(start, end+1)))

# define the banks and calculate the set of relativeLags and the MaxLag

# indexes in bankLags are the indexes required to get the tokens from the end of the token list
bankLags = [l(-67, -44), l(-43, -32), l(-31, -24), l(-23, -18), l(-17, -14), l(-13, -11), l(-10, -9), [-8], [-7], [-6], [-5], [-4], [-3], [-2], [-1]]
bankLags = [l(-36, -21), l(-20, -13), l(-12, -9), l(-8, -7), [-6], [-5], [-4], [-3], [-2], [-1]]
bankLengths = [len(b) for b in bankLags]
NumberOfBanks = len(bankLags)
relativeLags = set()
MaxLag = -1
for i1 in range(len(bankLags)-1):
    for i2 in range(i1+1, len(bankLags)):
        tMaxLag = max(bankLags[i2]) - min(bankLags[i1]) + 1
        if tMaxLag > MaxLag:
            MaxLag = tMaxLag
        rel = relativeLag(bankLags[i1], bankLags[i2])
        relativeLags.add(rel)
relativeLags = list(relativeLags)
MaxBankLength = max(bankLengths)

def plotBanks():
    resultsarray = zeros((NumberOfBanks, MaxLag))
    for bank, lags in enumerate(bankLags):
        for i in lags:
            resultsarray[NumberOfBanks - bank-1, -i-1] = MaxBankLength/bankLengths[bank]
    fig = plt.figure()
    ax = fig.add_subplot(111, axes_class=axisartist.Axes)
    ax.imshow( resultsarray, cmap = 'Blues' , interpolation = 'nearest' )
    #ax.set_yticks(arange(len(labels)), labels=labels)
    #ax.axis["left"].major_ticklabels.set_ha("left")
    #if title:
    #    ax.set_title(title)
    plt.show()

if __name__ == "__main__":
    rl = relativeLag(bankLags[0], bankLags[1])
    print (strRelativeLag(rl))
    rl = relativeLag(bankLags[0], bankLags[8])
    print (strRelativeLag(rl))
    plotBanks()
