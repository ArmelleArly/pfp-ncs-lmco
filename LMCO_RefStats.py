import argparse, logging, json
import numpy as np
from scipy import signal
from SigMFUtils import SigMFUtil
import types,gc,random,sys
import EnvIndicatorsLMCO as env
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler


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


if args.config:
    configFile = args.config
else:
    configFile = []


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


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def moving_average(self, data=[], win=5):
    b = np.ones(win) / float(win)
    a = 1.0
    Sxx = signal.filtfilt(b, a, data, axis=0, method='gust')
    return Sxx  # np.convolve(x, np.ones(w), 'valid') / w

def Visualize(F0=[],F1=[],F2=[],Centroids=[],labels=[],title=''):

    cmap = matplotlib.cm.get_cmap('tab20')
    cmap_values = range(0,len(cmap.colors))
    colors = np.array(
        ["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "beige", "brown", "gray", "cyan",
         "magenta"])
    MarkerSize = 20*4**2
    for SegIdx in range(F0.shape[1]):
        fig = plt.figure(SegIdx)
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(F0[:,SegIdx], F1[:,SegIdx,1], F2[:,SegIdx], c=colors[SegIdx],marker='o', alpha=0.5)
        ax.scatter(Centroids[0][SegIdx], Centroids[2][SegIdx], Centroids[3][SegIdx], s=MarkerSize,c='black', marker='x')

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.title(title + ', Seg:{}'.format(SegIdx))
       #ax.legend(AllOthersStr, numpoints=1, loc='upper right')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=len(labels))
        # vertical colorbar
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm,ticks = range(0,len(labels)))
        cbar.ax.set_title("scale")
     #   ax.set_xlim3d(0, 1)
     #   ax.set_ylim3d(0, 1)
     #   ax.set_zlim3d(0, 1)
        plt.show(block=False)
    plt.show(block= True)
    debug = 1



def run(config=[]):

    DH = SigMFUtil()
    NumOfSigMFFiles = DH.NumOfFiles(config['Ref']['InputParams']['DataPath'])
    print("Number of Files ",NumOfSigMFFiles)
    print("Directory Exist ", DH.DirExists(config['Ref']['InputParams']['DataPath']))
    print("Directory Empty ", DH.DirEmpty(config['Ref']['InputParams']['DataPath']))

    parameters, DataDict = DH.ReadMetaData(DirToProcess=config['Ref']['InputParams']['DataPath'],
                            Label=config['Ref']['InputParams']['Label'])


    # Now get the data
    pfp_log.info('Loading Channel {}, Label: {}'.format(config['Ref']['InputParams']['ChanNum'],config['Ref']['InputParams']['Label']))

    LabelStr = 'Label-{}'.format(config['Ref']['InputParams']['Label'])
    RawData, Label = DH.ReadDataFromList(FileList=DataDict[LabelStr], params=parameters, ChanNum=config['Ref']['InputParams']['ChanNum'])

    # Create the envelope
    pfp_log.info('Create Envelop, Decimation Factor: {}'.format(config['CommonParams']['Decimation']))
    if config['CommonParams']['Decimation'] != 1:
        DataDec = signal.decimate(np.abs(RawData), config['CommonParams']['Decimation'], ftype='fir')
    else:
        DataDec = np.abs(RawData)

    # Visually Check a random trace IF Visualization is TRUE
    if config['Ref']['OutputParams']['Visualization'] == True:

        randomIdx = random.randint(0, DataDec.shape[0]-1)

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
        debug = 1
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

    # Remove DataDec from memory
    del DataDec
    gc.collect()

    # Normalize Data 0-1
    DataDecEnvNorm = np.zeros(DataDecEnv.shape)
    pfp_log.info('Normalize (0-1)')
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
    DataDecEnvNormAvg = np.average(DataDecEnvNorm[0:Rdiv*config['CommonParams']['Average'],:].reshape(Rdiv,
                    config['CommonParams']['Average'], DataDecEnvNorm.shape[1]), axis=1)

    # Remove DataDecEnvNorm from memory
    del DataDecEnvNorm
    gc.collect()

    # Segment the data
    N = DataDecEnvNormAvg.shape[1]
    Ndiv = int(np.floor(N/config['CommonParams']['Segments']))
    NRem = N%config['CommonParams']['Segments']

    MACD = env.MACD()
    BB = env.BB()
    Cu = env.CuSu()
    # setup Error Arrays for the results
    # (Trace, Segments, Num of results)
    ResultsBB = np.zeros((Rdiv-1, config['CommonParams']['Segments']))
    ResultsCu = np.zeros((Rdiv-1, config['CommonParams']['Segments'], 2))
    ResultsMACD = np.zeros((Rdiv-1, config['CommonParams']['Segments']))

    # Randomize the Rows
    np.random.shuffle(DataDecEnvNormAvg)
    # Pick the first Row for the  Ref, data is already randomized
    if NRem > 0:
        ref = DataDecEnvNormAvg[0, 0:-NRem].reshape((config['CommonParams']['Segments'], Ndiv))
    else:
        ref = DataDecEnvNormAvg[0, :].reshape((config['CommonParams']['Segments'], Ndiv))


    # Delete the Row that was used for the Reference
    DataDecEnvNormAvg = np.delete(DataDecEnvNormAvg,0,0)

    pfp_log.info('Processing Data')
    for TraceIdx in range(0,DataDecEnvNormAvg.shape[0],1):

        if NRem > 0:
            data = DataDecEnvNormAvg[TraceIdx,0:-NRem].reshape((config['CommonParams']['Segments'],Ndiv))
        else:
            data = DataDecEnvNormAvg[TraceIdx, :].reshape((config['CommonParams']['Segments'], Ndiv))

        for SegIdx, sut in enumerate(data):
            # Bollinger Bands
            ResultsBB[TraceIdx,SegIdx], (PercentAbove, PercentBelow) = BB.BollingerBands(ref=ref[SegIdx,:], sut=sut,
                        std=config['CommonParams']['BB']['OPs']['NumStd'], period=config['CommonParams']['BB']['OPs']['Period'], Fs=1)
            if config['CommonParams']['BB']['OPs']['Visualization'] == True:
                TitleStr = 'BB:Trace {}, Segment {}'.format(TraceIdx,SegIdx)
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


    scaler = MinMaxScaler()
    ResultsCuNorm = np.zeros((ResultsCu.shape))
    ResultsBBNorm = scaler.fit_transform(ResultsBB)
    ResultsCuNorm[:,:,0] = scaler.fit_transform(ResultsCu[:,:,0]) # Mean
    ResultsCuNorm[:,:, 1] = scaler.fit_transform(ResultsCu[:,:, 1]) # Sigma
    ResultsMACDNorm = scaler.fit_transform(ResultsMACD)




    CentroidNorm = (np.mean(ResultsBBNorm,axis=0), np.mean(ResultsCuNorm[:,:,0],axis=0), np.mean(ResultsCuNorm[:,:,1],axis=0),
    np.mean(ResultsMACDNorm,axis=0))

    Visualize(F0=ResultsBBNorm,F1=ResultsCuNorm,F2=ResultsMACDNorm,Centroids=CentroidNorm,labels=parameters['UniqueLabels'],
              title=config['Ref']['OutputParams']['TitleStr'])




    if config['Ref']['OutputParams']['SaveResults'] == True:
        pfp_log.info('Saving results to {}{}'.format(config['Ref']['OutputParams']['Path'],config['Ref']['OutputParams']['FileName']))

        # Write Results to File
        EnvStats = { 'Params':{
                                'Segments' :config['CommonParams']['Segments'],
                                'ChanNum': config['Ref']['InputParams']['ChanNum'],
                                'Average': config['CommonParams']['Average']
                                },
                    'Ref': ref,
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
        env.save_pickle(dir=config['Ref']['OutputParams']['Path'],filename=config['Ref']['OutputParams']['FileName'],obj=EnvStats)




if __name__ == '__main__':

    if configFile != []:
        with open(configFile) as json_file:
            config = json.load(json_file)
            json_file.close()

    run(config)