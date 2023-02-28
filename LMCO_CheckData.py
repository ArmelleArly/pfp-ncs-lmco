import argparse, logging
import numpy as np
from scipy import signal
import pandas as pd
from SigMFUtils import SigMFUtil
import types
import matplotlib.pyplot as plt




LOG_LEVEL = {
     'debug': logging.DEBUG,
     'info': logging.INFO,
     'warning': logging.WARNING,
     'error': logging.ERROR,
     'critical': logging.CRITICAL,
 }

logging.getLogger('matplotlib.font_manager').disabled = True

args = types.SimpleNamespace()
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--log-level', dest='log_level', help='Log level', default='info',nargs='?', choices=LOG_LEVEL)
parser.add_argument('-p', '--infile', dest='DataPath',help='Path to Data')
args = parser.parse_args()


config = {
    'format': '%(asctime)s:%(levelname)s %(message)s',
    'datefmt': '%m/%d/%Y %I:%M:%S %p'
    }

if args.log_level:
    config['level'] = LOG_LEVEL[args.log_level]
    logging.basicConfig(**config)
    pfp_log = logging.getLogger('pfp')
    pfp_log.info('PFP Started')

if args.DataPath:
    DataPath = args.DataPath
else:
    DataPath = []

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def moving_average(self, data=[], win=5):
    b = np.ones(win) / float(win)
    a = 1.0
    Sxx = signal.filtfilt(b, a, data, axis=0, method='gust')
    return Sxx  # np.convolve(x, np.ones(w), 'valid') / w

def DisplaySync(data, FigNum, TitleStr=[]):
    # Calculate simple envelope
    Env = signal.decimate(np.abs(data),10,ftype='fir')
    # Scale data bewteen 0-1 to fit on screen
    EnvNorm = NormalizeData(Env)
    if FigNum ==[]:
        FigNum = 0
    plt.figure(FigNum)
    for idx in range(0, len(data), 1):
        if idx % 10 == 0 and FigNum != 0:
            FigNum = FigNum+1
            plt.figure(FigNum)
        plt.plot(EnvNorm[idx, :] + idx * 1.1, 'r')
        plt.title('{}'.format(TitleStr))
        plt.tight_layout()
        plt.minorticks_on()
        plt.grid(True, which='minor')
    plt.show(block=True)

    debug =1
    return FigNum


def DisplayConfidBound(df, FigNum,TitleStr=[]):

    plt.figure(FigNum)
    plt.plot(df.Env, 'b', label='Env')
    plt.fill_between(df.index,df.Env+df.Std, df.Env-df.Std, color='r', alpha=.2, label='std')
    plt.legend()
    plt.title('{}'.format(TitleStr))
    plt.tight_layout()
    plt.minorticks_on()
    plt.show(block=True)
    debug = 1

def Overlay(data, FigNum, TitleStr=[]):
    plt.figure(FigNum)
    for idx in range(0,10,1):
        plt.plot(data[idx,:], 'r')
    plt.title('{}'.format(TitleStr))
    plt.tight_layout()
    plt.minorticks_on()
    plt.grid(True, which='minor')
    plt.show(block = True)
    debug = 1


def run(DataPath=[]):


    DH = SigMFUtil()
    NumOfSigMFFiles = DH.NumOfFiles(DataPath)
    print("Number of Files ",NumOfSigMFFiles)
    print("Directory Exist ", DH.DirExists(DataPath))
    print("Directory Empty ", DH.DirEmpty(DataPath))

    PercentTrain = 1
    PercentDev = 0
    TrainList, DevList, TestList = DH.LoadPFPData(DataPath,PercentTrain,PercentDev)

    # sort into labels
    parameters,SortedSigMFList = DH.SigMFSort(TrainList)


    TitleStr = 'LMCO'
    # Now get the data
    Chan0,Label0 = DH.ReadDataFromList(SortedSigMFList[0], parameters,ChanNum=1)
    Chan1,Label1 = DH.ReadDataFromList(SortedSigMFList[0],parameters,ChanNum=2)

    Chan2,Label2 = DH.ReadDataFromList(SortedSigMFList[0],parameters,ChanNum=3)
    Chan3,Label3 = DH.ReadDataFromList(SortedSigMFList[0],parameters,ChanNum=4)

    Overlay(Chan0, 0, 'Overlay, Chan 0')
    Overlay(Chan1, 1, 'Overlay, Chan 1')
    Overlay(Chan2, 2, 'Overlay, Chan 2')
    Overlay(Chan3, 3, 'Overlay, Chan 3')


    Chan0Stats = {'Std': pd.Series(signal.decimate(np.std(Chan0,axis=0),10,ftype='fir'),name='STD'),
                   'Env': pd.Series(np.mean(signal.decimate(np.abs(Chan0),10, ftype='fir'),axis=0),name = 'Env')}

    Chan1Stats = {'Std': pd.Series(signal.decimate(np.std(Chan1, axis=0), 10, ftype='fir'), name='STD'),
                  'Env': pd.Series(np.mean(signal.decimate(np.abs(Chan1), 10, ftype='fir'), axis=0), name='Env')}

    Chan2Stats = {'Std': pd.Series(signal.decimate(np.std(Chan2, axis=0), 10, ftype='fir'), name='STD'),
                  'Env': pd.Series(np.mean(signal.decimate(np.abs(Chan2), 10, ftype='fir'), axis=0), name='Env')}

    Chan3Stats = {'Std': pd.Series(signal.decimate(np.std(Chan3, axis=0), 10, ftype='fir'), name='STD'),
                  'Env': pd.Series(np.mean(signal.decimate(np.abs(Chan3), 10, ftype='fir'), axis=0), name='Env')}

    Chan0DF = pd.DataFrame(Chan0Stats)
    Chan1DF = pd.DataFrame(Chan1Stats)
    Chan2DF = pd.DataFrame(Chan2Stats)
    Chan3DF = pd.DataFrame(Chan3Stats)

    DisplayConfidBound(Chan0DF, FigNum=4, TitleStr='Variance Check, Chan 0')
    DisplayConfidBound(Chan1DF, FigNum=5, TitleStr='Variance Check, Chan 1')
    DisplayConfidBound(Chan2DF, FigNum=6, TitleStr='Variance Check, Chan 2')
    DisplayConfidBound(Chan3DF, FigNum=7, TitleStr='Variance Check, Chan 3')

    FigNum = DisplaySync(Chan0, FigNum=8,TitleStr='Synch Check, Chan 0')
    FigNum = DisplaySync(Chan1, FigNum=FigNum, TitleStr='Synch Check, Chan 1')
    FigNum = DisplaySync(Chan2, FigNum=FigNum, TitleStr='Synch Check, Chan 2')
    FigNum = DisplaySync(Chan3, FigNum=FigNum, TitleStr='Synch Check, Chan 3')
    debug = 1




if __name__ == '__main__':

    run(DataPath)