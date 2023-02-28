import argparse, logging, json, gc, random, types
import numpy as np
from scipy import signal
import pandas as pd
import seaborn as sns
from SigMFUtils import SigMFUtil
import matplotlib.pyplot as plt
import EnvIndicatorsLMCO as env
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler

# python3 LMCO_DutStats.py -l debug -c 'json DUT Config" -p "path to pickle file containing Reference and Error Stats"


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
parser.add_argument('-c', '--config', dest="config",help='json Envelop Configuration file')
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



def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global max of dmax-chunks of locals max
    lmin = lmin[[i + np.argmin(s[lmin[i:i + dmin]]) for i in range(0, len(lmin), dmin)]]
    # global min of dmin-chunks of locals min
    lmax = lmax[[i + np.argmax(s[lmax[i:i + dmax]]) for i in range(0, len(lmax), dmax)]]

    return lmin, lmax




def DisplayBOIsErrorDist2(df, title='', FigNum= 1, PltVar = ''):
    penguins = sns.load_dataset("penguins")

    fig, ax = plt.subplots(1, 2,num= FigNum)
    p = sns.histplot(data=df, bins=20, x=PltVar, kde=True, ax=ax[0], hue='Labels', fill=True, stat='density')
    sns.kdeplot(data=df, x= PltVar, ax=ax[0],hue='Labels', fill=True)
    x = p.lines[0].get_xdata()
    y = p.lines[0].get_ydata()
  #  ax[0].axvline(Threshold, color='red', )
    p.lines[0].remove()
    p.set(xlabel='Error', ylabel='Count',
          title='{}'.format(title))
    p1 = sns.histplot(data=df, cumulative=True, x=PltVar, kde=True, log_scale=False, stat='density', fill=True,
                      common_norm=False,hue='Labels', ax=ax[1])

    p1.set(xlabel='Cumulative Error', ylabel='Count', title='CDF {}'.format(title))
    fig.tight_layout()
    plt.show(block=True)

    debug = 1



def DisplayBOIsErrorDist(error1=[], error2=[], title='', FigNum= 1):
    penguins = sns.load_dataset("penguins")
    errordict1 = {'error1': pd.Series(error1, name='error1'),
                     'error2': pd.Series(error2, name='error1')}

    df1 = pd.DataFrame(errordict1)


    fig, ax = plt.subplots(1, 2,num= FigNum)
    p = sns.histplot(data=df1, bins=10, kde=True, stat='density',hue='error',ax=ax[0])
    x = p.lines[0].get_xdata()
    y = p.lines[0].get_ydata()
  #  ax[0].axvline(Threshold, color='red', )
    p.lines[0].remove()
    p.set(xlabel='Error', ylabel='Count',
          title='{} errors ()'.format(title))
    p1 = sns.histplot(data=df1, cumulative=True, kde=True, log_scale=False, stat='density', fill=False,
                      common_norm=False, ax=ax[1])

    p1.set(xlabel='Cumulative Error', ylabel='Count', title='CDF {} errors ()'.format(title))
    fig.tight_layout()
    plt.show(block=True)
    debug = 1

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


def run(RefConfig=[],config=[]):


    DH = SigMFUtil()
    NumOfSigMFFiles = DH.NumOfFiles(config['Dut']['InputParams']['DataPath'])
    print("Number of Files ",NumOfSigMFFiles)
    print("Directory Exist ", DH.DirExists(config['Dut']['InputParams']['DataPath']))
    print("Directory Empty ", DH.DirEmpty(config['Dut']['InputParams']['DataPath']))


    parameters,DataDict = DH.ReadMetaData(DirToProcess=config['Dut']['InputParams']['DataPath'],
                                                   Label=config['Dut']['InputParams']['Label'])


    # Now get the DUT data
    pfp_log.info('Loading Channel {}, Label: {}'.format(config['Dut']['InputParams']['ChanNum'], config['Dut']['InputParams']['Label']))
    # Now get the data
    LabelStr = 'Label-{}'.format(config['Dut']['InputParams']['Label'])
    RawData, Label = DH.ReadDataFromList(FileList=DataDict[LabelStr], params=parameters,
                                         ChanNum=config['Dut']['InputParams']['ChanNum'])
    # Decimate/Filter
    pfp_log.info('Create Envelop, Decimation Factor: {}'.format(config['CommonParams']['Decimation']))
    if config['CommonParams']['Decimation'] != 1:
        DataDec = signal.decimate(np.abs(RawData), config['CommonParams']['Decimation'], ftype='fir')
    else:
        DataDec = np.abs(RawData)

    if config['Dut']['OutputParams']['Visualization'] == True:
        randomIdx = random.randint(0,DataDec.shape[0]-1)

        plt.figure(2)
        plt.plot(RawData[randomIdx, :], 'k')
        plt.title('Raw Data')
        plt.show(block=False)

        plt.figure(3)
        plt.plot(DataDec[randomIdx, :], 'k')
        plt.title('abs(Raw) Data')
        plt.show(block=False)
        low_idx, high_idx = hl_envelopes_idx(DataDec[randomIdx, :], dmin=config['CommonParams']['dmin'], dmax=config['CommonParams']['dmax'])

        plt.figure(4)
        plt.plot(DataDec[randomIdx, low_idx], 'k')
        plt.plot(DataDec[randomIdx, high_idx], 'b')
        plt.title('Envelope')
        plt.show(block=True)


    # Remove RawData from memory
    del RawData
    gc.collect()

    # Use local Max/Min to create a Smoother Envelope
    IdxLen = np.zeros((DataDec.shape[0]))
    DataDecEnv = np.zeros(DataDec.shape)
    for idx in range(0, DataDec.shape[0], 1):
        low_idx, high_idx = hl_envelopes_idx(DataDec[idx, :], dmin=config['CommonParams']['dmin'], dmax=config['CommonParams']['dmax'])
        DataDecEnv[idx, 0:len(high_idx)] = DataDec[idx, high_idx]
        IdxLen[idx] = len(high_idx)
    DataDecEnv = DataDecEnv[:, 0:int(min(IdxLen))]

    # need to make the Ref and Dut the same length
    ref = RefConfig['Ref'].flatten()
    if len(ref) > DataDecEnv.shape[1]:
        ref = ref[0:DataDecEnv.shape[1]]
    elif len(ref) < DataDecEnv.shape[1]:
        DataDecEnv = np.delete(DataDecEnv,np.s_[len(ref):DataDecEnv.shape[1]],axis=1)

    # Remove DataDec from memory
    del DataDec
    gc.collect()

    # Normalize Data 0-    pfp_log.info('Normalize (0-1)')
    DataDecEnvNorm = np.zeros(DataDecEnv.shape)
    for idx in range(0, DataDecEnv.shape[0], 1):
        DataDecEnvNorm[idx, :] = NormalizeData(DataDecEnv[idx, :])

    # Remove DataDecEnv from memory
    del DataDecEnv
    gc.collect()

        # Calculate the requested average
    R = DataDecEnvNorm.shape[0]
    Rdiv = int(np.floor(R / config['CommonParams']['Average']))
    RRem = R % config['CommonParams']['Average']

    pfp_log.info('Create Average Reference Waveform'.format(config['CommonParams']['Average']))
    DataDecEnvNormAvg = np.average(DataDecEnvNorm[0:Rdiv * config['CommonParams']['Average'], :].reshape(Rdiv,
                config['CommonParams']['Average'],DataDecEnvNorm.shape[1]),axis=1)

    # Remove DataDecEnvNorm from memory
    del DataDecEnvNorm
    gc.collect()

    # Segment the DUT data
    N = DataDecEnvNormAvg.shape[1]
    Ndiv = int(np.floor(N / config['CommonParams']['Segments']))
    NRem = N % config['CommonParams']['Segments']


    MACD = env.MACD()
    BB = env.BB()
    Cu = env.CuSu()
    # setup Error Arrays for the results
    # (Trace, Segments, Num of results)
    ResultsBB = np.zeros((Rdiv-1, config['CommonParams']['Segments']))
    ResultsCu = np.zeros((Rdiv-1, config['CommonParams']['Segments'], 2))
    ResultsMACD = np.zeros((Rdiv-1, config['CommonParams']['Segments']))


    # Reshape the Reference Data from the Results file
    if NRem > 0:
        ref = ref[0:-NRem].reshape((config['CommonParams']['Segments'], Ndiv))
    else:
        ref = ref[:].reshape((config['CommonParams']['Segments'], Ndiv))


    pfp_log.info('Processing Data')
    for TraceIdx in range(0,DataDecEnvNormAvg.shape[0]-1,1):
        if NRem > 0:
            data = DataDecEnvNormAvg[TraceIdx,0:-NRem].reshape((config['CommonParams']['Segments'],Ndiv))
        else:
            data = DataDecEnvNormAvg[TraceIdx, :].reshape((config['CommonParams']['Segments'], Ndiv))

        for SegIdx, sut in enumerate(data):
            # Bollinger Bands
            ResultsBB[TraceIdx,SegIdx], (PercentAbove, PercentBelow) = BB.BollingerBands(ref=ref[SegIdx,:], sut=sut,
                        std=config['CommonParams']['BB']['OPs']['NumStd'], period=config['CommonParams']['BB']['OPs']['Period'], Fs=1)
            if config['CommonParams']['BB']['OPs']['Visualization'] == True:
                TitleStr = 'BB:Trace {}, Segment {}'.format(TraceIdx, SegIdx)
                BB.DisplayChart(title=TitleStr)

            # Cumluative Sum CDF
            refErrorCuSuStats, sutErrorCuSuStats, SlowErrorCuSuStats, FastErrorCuSuStats = Cu.CuSu(ref=ref[SegIdx,:], sut=sut,
                        FastEMA = config['CommonParams']['Cu']['OPs']['FastEMA'], SlowEMA = config['CommonParams']['Cu']['OPs']['SlowEMA'],Fs=1)
            if config['CommonParams']['Cu']['OPs']['Visualization'] == True:
                TitleStr = 'Cu:Trace {}, Segment {}'.format(TraceIdx, SegIdx)
                Cu.DisplayChart(title=TitleStr)
            ResultsCu[TraceIdx,SegIdx, 0] = sutErrorCuSuStats[0]
            ResultsCu[TraceIdx, SegIdx, 1] = sutErrorCuSuStats[1]

            _, ResultsMACD[TraceIdx, SegIdx] = MACD.MACD(ref=ref[SegIdx,:],sut=sut,FastEMA=config['CommonParams']['MACD']['OPs']['FastEMA'],
                            SlowEMA=config['CommonParams']['MACD']['OPs']['SlowEMA'], SigEMA=config['CommonParams']['MACD']['OPs']['SigEMA'],  Fs=1)
            if config['CommonParams']['MACD']['OPs']['Visualization'] == True:
                TitleStr = 'MACD:Trace {}, Segment {}'.format(TraceIdx, SegIdx)
                MACD.DisplayChart(title=TitleStr)
            debug = 1

    Centroid = (np.mean(ResultsBB, axis=0), np.mean(ResultsCu[:, :, 0], axis=0), np.mean(ResultsCu[:, :, 1], axis=0),
                np.mean(ResultsMACD, axis=0))

    if config['Dut']['OutputParams']['SaveResults'] == True:
        pfp_log.info('Saving results to {}{}'.format(config['Dut']['OutputParams']['Path'],config['Dut']['OutputParams']['FileName']))

        # Write Results to File
        EnvStats = { 'Params':{
                                'Segments' :config['CommonParams']['Segments'],
                                'ChanNum': config['Dut']['InputParams']['ChanNum'],
                                'Average': config['CommonParams']['Average']
                                },
                    'ResultsBB': ResultsBB,
                    'ResultsCu': ResultsCu,
                    'ResultsMACD': ResultsMACD,
                    'SegmentCentroids': {
                        'BB': Centroid[0],
                        'CuMean': Centroid[1],
                        'CuSigma': Centroid[2],
                        'MACD': Centroid[3]
                    }}
        # Save Pickle checks if the directory exist and creates it fi not
        env.save_pickle(dir=config['Dut']['OutputParams']['Path'],filename=config['Dut']['OutputParams']['FileName'],obj=EnvStats)


    # Normalize (0-1) the data for Visualization
    scaler = MinMaxScaler()
    BBFeatures = np.concatenate((RefConfig['ResultsBB'], ResultsBB), axis=0)
    BBFeatures = scaler.fit_transform(BBFeatures)
    x1 = np.split(BBFeatures, [len(RefConfig['ResultsBB']),len(BBFeatures)], axis=0)
    RefConfig['ResultsBB'] = x1[0]
    ResultsBB = x1[1]


    Cu0Features = np.concatenate((RefConfig['ResultsCu'][:,:,0], ResultsCu[:,:,0]), axis=0)
    Cu0Features = scaler.fit_transform(Cu0Features)
    x1 = np.split(Cu0Features, [len(RefConfig['ResultsCu'][:,:,0]),len(Cu0Features)], axis=0)
    RefConfig['ResultsCu'][:,:,0] = x1[0]
    ResultsCu[:,:,0] = x1[1]

    Cu1Features = np.concatenate((RefConfig['ResultsCu'][:, :, 1], ResultsCu[:, :, 1]), axis=0)
    Cu1Features = scaler.fit_transform(Cu1Features)
    x1 = np.split(Cu1Features, [len(RefConfig['ResultsCu'][:,:,1]),len(Cu1Features)], axis=0)
    RefConfig['ResultsCu'][:, :, 1] = x1[0]
    ResultsCu[:, :, 1] = x1[1]

    MACDFeatures = np.concatenate((RefConfig['ResultsMACD'], ResultsMACD), axis=0)
    MACDFeatures = scaler.fit_transform(MACDFeatures)
    x1 = np.split(MACDFeatures, [len(RefConfig['ResultsMACD']),len(MACDFeatures)], axis=0)
    RefConfig['ResultsMACD'] = x1[0]
    ResultsMACD = x1[1]

    Centroid = (np.mean(ResultsBB, axis=0), np.mean(ResultsCu[:, :, 0], axis=0),np.mean(ResultsCu[:, :, 1], axis=0), np.mean(ResultsMACD, axis=0))

    RefCentroid = (np.mean(RefConfig['ResultsBB'], axis=0), np.mean(RefConfig['ResultsCu'][:, :, 0], axis=0),np.mean(RefConfig['ResultsCu'][:, :, 1], axis=0), np.mean(RefConfig['ResultsMACD'], axis=0))

    if config['Dut']['OutputParams']['Visualization'] == True:
        MarkerSize = 20 * 4 ** 2
        pfp_log.info('Plotting Results')
        for SegIdx in range(config['CommonParams']['Segments']):
            Colors = ['r', 'b', 'k', 'g']
            Markers = ['o', '*', 'x', '.']
            fig = plt.figure(SegIdx)
            ax = plt.axes(projection='3d')
            # plt.title('Means: ' + V1TestList[ii])
            plt.title(config['Dut']['OutputParams']['TitleStr'] + ' Segment {}'.format(SegIdx))
            ax.scatter(RefConfig['ResultsBB'][:,SegIdx], RefConfig['ResultsCu'][:,SegIdx, 1], RefConfig['ResultsMACD'][:, SegIdx], c=Colors[0], marker=Markers[0])
            ax.scatter(ResultsBB[:, SegIdx], ResultsCu[:,SegIdx,1], ResultsMACD[:,SegIdx], c=Colors[1], marker=Markers[1])
            # Centroids
            ax.scatter(Centroid[0][SegIdx], Centroid[2][SegIdx], Centroid[3][SegIdx], s=MarkerSize, c='black',
                       marker='x')
            ax.scatter(RefCentroid[0][SegIdx], RefCentroid[2][SegIdx], RefCentroid[3][SegIdx], s=MarkerSize, c='black',
                       marker='x')

            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
            Refline = Line2D([0], [0], color='red', linewidth=3, linestyle='None',marker='o')
            Dutline = Line2D([0], [0], color='blue', linewidth=3, linestyle='None', marker='o')

            ax.legend([Refline,Dutline],['Ref', 'Dut'], numpoints=1, loc='upper right')
            ax.set_xlim3d(0, 1)
            ax.set_ylim3d(0, 1)
            ax.set_zlim3d(0, 1)
            plt.show(block=False)
        plt.show(block= True)
    debug =1




if __name__ == '__main__':

    if args.config != []:
        with open(args.config) as json_file:
            config = json.load(json_file)
            json_file.close()

    RefConfig = env.load_pickle(config['Ref']['OutputParams']['Path'], config['Ref']['OutputParams']['FileName'])

    run(RefConfig,config)