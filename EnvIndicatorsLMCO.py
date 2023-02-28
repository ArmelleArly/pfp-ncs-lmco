import numpy as np
import os, pickle
from matplotlib import pyplot as plt
from scipy.stats import skew
import pandas as pd


def save_pickle(obj=[], dir=[], filename=[]):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(dir=[], filename=[]):
    with open(dir + filename + '.pkl', 'rb') as f:
        return pickle.load(f)

#  Class
class BB:
        def __init__(self):
            self.name = 'BollingerBands'
            #self.std = std
            #self.period = period
            #self.ref = ref
            #self.sut = sut
            #self.ref_Rolling_mu = 0
            #self.sut_Rolling_mu = 0
            #self.ref_upperlmt = 0
            #self.ref_lowerlmt = 0

        def display(self):
            print("Indicator Name:", self.name)

        # Instance method
        def BollingerBands(self,ref, sut, std=2,  period=20, Fs = 1):

        # This is based on the Bollinger Bands. Bollinger Bands are envelopes at a standard deviation
        # level aboce and below a simple moving average of the signal. Because the distance of the bands is based on
        # standard deviation, they adjust to the variance in the signal.
        # Bolinger Bands use 2 parameters, Period and Standard Deviations, default values ae 20 for period and 2 for
        # standard deviation.
        # Returns three indicators.
        # Score : Number of samples that cross the Bollinger Bands, Normalized between 0-1
        # (PercentAbove,PercentBelow) : Percent of Signal above the upper Bollinger Band and Percent of singal below
        # the Bollinger Band.

            start_time = 0
            end_time = len(ref)
            time = np.arange(start_time, end_time, 1) / Fs

            # Signal Under Test, create a Pandas TIme Series
            sut_ts = pd.Series(sut,name='TS', index=pd.Index(time,name='time'))

            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=period)

            sut_Rolling_mu = sut_ts.rolling(period).mean()
            sut_Rolling_mu = sut_Rolling_mu.dropna()
            sut_Rolling_mu = sut_Rolling_mu.reset_index(drop=True)

            # Reference Waveform
            ref_ts = pd.Series(ref,name='TS',index=pd.Index(time,name='time'))

            ref_Rolling_mu = ref_ts.rolling(period).mean()
            ref_Rolling_std = ref_ts.rolling(period).std()
            # Discard the NaN's
            ref_Rolling_mu = ref_Rolling_mu.dropna()
            ref_Rolling_mu = ref_Rolling_mu.reset_index(drop=True)
            # Discard the NaN's
            ref_Rolling_std = ref_Rolling_std.dropna()
            ref_Rolling_std = ref_Rolling_std.reset_index(drop=True)

            # Calculate the Upper and Lower Sigma Bounds
            ref_upperlmt = ref_Rolling_mu + ref_Rolling_std*std
            ref_lowerlmt = ref_Rolling_mu - ref_Rolling_std*std

            # Get indexes of Signal Under Test where it is below the Lower Sigma Bound
            UpperLmtResults =  np.greater(sut_Rolling_mu,ref_upperlmt)
            # Get index where TRUE
            UpperTrueIdx = np.where(UpperLmtResults)
            # calculate the area where the Signal Under Test is above the Upper Sigma Bound
            AreaAboveUL = np.sum(sut_Rolling_mu.loc[UpperTrueIdx] - ref_upperlmt.loc[UpperTrueIdx]) * len(UpperTrueIdx)
            # Normalize the upper area measurement
            TotalAreaUL = np.sum(ref_upperlmt) * len(ref_upperlmt)
            PercentAbove = AreaAboveUL/TotalAreaUL

            # Get indexes of Signal Under Test where it is below the Lower Sigma Bound
            LowerLmtResults = np.less(sut_Rolling_mu, ref_lowerlmt)
            # Get index where TRUE
            LowerTrueIdx = np.where(LowerLmtResults)
            # calculate the area where the Signal Under Test is below the Lower Sigma Bound
            AreaBelowLL = np.sum(ref_Rolling_mu.loc[LowerTrueIdx] + sut_Rolling_mu.loc[LowerTrueIdx]) * len(LowerTrueIdx)
            # Normalize the lower area measurement
            TotalAreaLL = np.sum(ref_lowerlmt) * len(ref_lowerlmt)
            PercentBelow = AreaBelowLL / TotalAreaLL

            # Compute a score that is between 0-1 that indicates the percentage of the DUT is out of the sigma bounds.
            Score = (np.sum(UpperLmtResults)+np.sum(LowerLmtResults))/(len(UpperLmtResults) + len(LowerLmtResults))
            self.ref = ref_ts
            self.sut = sut_ts
            self.ref_Rolling_mu = ref_Rolling_mu
            self.sut_Rolling_mu = sut_Rolling_mu
            self.ref_upperlmt = ref_upperlmt
            self.ref_lowerlmt = ref_lowerlmt
            self.index = ref_Rolling_mu.index
            self.Fs = Fs

            return(Score, (PercentAbove,PercentBelow))

        def DisplayChart(self,title=''):
            plt.plot(self.ref.index/self.Fs, self.ref, 'b', label='Ref')
            plt.plot(self.sut.index/self.Fs, self.sut, 'g', label='SUT')
            plt.plot(self.index/self.Fs, self.ref_Rolling_mu, 'b--', label='Ref SMA')
            plt.plot(self.index/self.Fs, self.sut_Rolling_mu, 'g--', label='SUT SMA')

            # plt.plot(ts.shift(int(FilterLen/2)), 'g')
            plt.plot(self.index/self.Fs, self.ref_upperlmt, 'c--', label='UpperBand')
            plt.plot(self.index/self.Fs, self.ref_lowerlmt, 'k--', label='LowerBand')
            plt.title(title)
            plt.legend()
            plt.show(block=False)

            plt.figure(4)
            plt.plot(self.index / self.Fs, self.ref_Rolling_mu, 'b--', label='Ref SMA')
            plt.plot(self.index / self.Fs, self.sut_Rolling_mu, 'g--', label='SUT SMA')

            # plt.plot(ts.shift(int(FilterLen/2)), 'g')
            plt.plot(self.index / self.Fs, self.ref_upperlmt, 'c--', label='UpperBand')
            plt.plot(self.index / self.Fs, self.ref_lowerlmt, 'k--', label='LowerBand')
            plt.title(title)
            plt.legend()
            plt.show(block=True)
            debug = 1

#  Class
class MACD:
        def __init__(self):
            self.name = 'MACD'

        def display(self):
            print("Indicator Name:", self.name)


        def MACD(self, ref=[], sut=[],FastEMA=20, SlowEMA= 80, SigEMA= 60, Fs=1):
        # Moving Average Convergance Divergance indicator is a momentum oscillator primarily used for trends. It appears on
        # the chart as two lines that oscillate without boundaries. The crossover of the two (MACD and Ref Signal) lines
        # gives a a signal similar to moving averages with different periods. These two lines are formed using the MACD and
        # reference signal as a base to calculate an error measurement (e1). The A third line is added which is the Signal Under
        # Test, this SUT is measured against the MACD and an error measurement is calculated (e2).

            start_time = 0
            end_time = len(ref)
            time = np.arange(start_time, end_time, 1) / Fs
            # Create Time Series
            ref_ts = pd.Series(ref, name='TS', index=pd.Index(time, name='time'))

            # MACD Line creation, two exponential moving Averages
            refFastMA = ref_ts.ewm(span=FastEMA,adjust=False).mean()
            refSlowMA = ref_ts.ewm(span=SlowEMA, adjust=False).mean()
            refMACD = refFastMA - refSlowMA
            # The Signal Line is created with a N period EMA of the MCD line
            refSigLine = refMACD.ewm(span=SigEMA, adjust=False).mean()

            # Signal Under Test
            sut_ts = pd.Series(sut, name='TS', index=pd.Index(time, name='time'))
            sutFastMA = sut_ts.ewm(span=FastEMA, adjust=False).mean()
            sutSlowMA = sut_ts.ewm(span=SlowEMA, adjust=False).mean()
            sutMACD = sutFastMA - sutSlowMA
            # The Signal Line is created with a N period EMA of the MCD line
            sutSigLine = sutMACD.ewm(span=SigEMA, adjust=False).mean()

            # Compute the error between the MACD line and Ref and MACD line and Sut
            e1 = np.sum(np.abs(refMACD - refSigLine))
            e2 = np.sum(np.abs(refMACD - sutSigLine))
            self.refMACD = refMACD
            self.sutMACD = sutMACD
            self.refSigLine = refSigLine
            self.sutSigLine = sutSigLine
            self.Fs = Fs
            debug = 1
            return e1,e2

        def DisplayChart(self,title=''):
            plt.figure(2)
            plt.plot(self.refMACD.index,np.abs(self.refMACD - self.refSigLine), 'k', label= 'Ref Error')
            plt.plot(self.refMACD.index,np.abs(self.refMACD - self.sutSigLine), 'r', label= 'SUT Error')
            plt.legend(loc='upper left')
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
            plt.title(title)
            plt.minorticks_on()
            plt.show(block=False)

            plt.figure(4)
            plt.plot(self.refMACD.index, self.refMACD, label='Ref MACD', color='r')
            plt.plot(self.refSigLine.index, self.refSigLine, label='Ref Signal Line', color='b')
            plt.plot(self.sutSigLine.index, self.sutSigLine, label='SUT Signal Line', color='g')
            plt.legend(loc='upper left')
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
            plt.title(title)
            plt.minorticks_on()
            plt.show(block=True)
            debug =1


# Class
class CuSu:
        def __init__(self):
            self.name = 'CumulativeSumError'

        def display(self):
            print("Indicator Name:", self.name)

        def __scale_minmax(self, X, min=0.0, max=1.0):
            X_std = (X - X.min()) / (X.max() - X.min())
            X_scaled = X_std * (max - min) + min
            return X_scaled

        # Instance method
        def CuSu(self,ref=[], sut=[], FastEMA=20, SlowEMA= 80,Fs = 1):
            start_time = 0
            end_time = len(ref)
            time = np.arange(start_time, end_time, 1) / Fs
            # Create Time Series
            ref_ts = pd.Series(ref, name='TS', index=pd.Index(time, name='time'))

            # Reference two exponential moving Averages
            refFastMA = ref_ts.ewm(span=FastEMA, adjust=False).mean()
            refSlowMA = ref_ts.ewm(span=SlowEMA, adjust=False).mean()

            # Signal Under Test two exponential moving Averages
            sut_ts = pd.Series(sut, name='TS', index=pd.Index(time, name='time'))
            sutFastMA = sut_ts.ewm(span=FastEMA, adjust=False).mean()
            sutSlowMA = sut_ts.ewm(span=SlowEMA, adjust=False).mean()

            # Calculate the Cumlative Sums
            refFastCuSu = np.cumsum(refFastMA).values
            refSlowCuSu = np.cumsum(refSlowMA).values
            sutFastCuSu = np.cumsum(sutFastMA).values
            sutSlowCuSu = np.cumsum(sutSlowMA).values
            tmp = np.array([refFastCuSu,refSlowCuSu,sutFastCuSu,sutSlowCuSu])
            # Scale the data
            ScaledData = self.__scale_minmax(tmp, min=0.0, max=1.0)
            refFastCuSuN = ScaledData[0,:]
            refSlowCuSuN = ScaledData[1,:]
            sutFastCuSuN = ScaledData[2,:]
            sutSlowCuSuN = ScaledData[3,:]
            # Calculate the Errors
            refErrorCuSu = refFastCuSuN - refSlowCuSuN
            sutErrorCuSu = refFastCuSuN - sutSlowCuSuN
            SlowErrorCuSu = refSlowCuSuN - sutSlowCuSuN
            FastErrorCuSu = refFastCuSuN - sutFastCuSuN

            # Error Stats, (refError : refFastEMA -refSlowEMA), sutError : refFastEMA-sutFastEMA,
            # SlowError : refSlowEMA - sutSlowEMA, FastError : refFastEMA - sutFastEMA)
            refErrorCuSuStats = (refErrorCuSu.mean(), refErrorCuSu.std())
            sutErrorCuSuStats = (sutErrorCuSu.mean(), sutErrorCuSu.std())
            SlowErrorCuSuStats = (SlowErrorCuSu.mean(), SlowErrorCuSu.std())
            FastErrorCuSuStats = (FastErrorCuSu.mean(), FastErrorCuSu.std())


            # For the Display Chart
            # Scale the Errors for display
            tmp2 = np.array([refErrorCuSu, sutErrorCuSu, SlowErrorCuSu, FastErrorCuSu])
            # Scale the data
            ScaledData2 = self.__scale_minmax(tmp2, min=0.0, max=1.0)
            refErrorCuSuN = ScaledData2[0,:]
            sutErrorCuSuN = ScaledData2[1,:]
            SlowErrorCuSuN = ScaledData2[2,:]
            FastErrorCuSuN = ScaledData2[3,:]

            self.refErrorCuSu = refErrorCuSu
            self.sutErrorCuSu = sutErrorCuSu
            self.SlowErrorCuSu = SlowErrorCuSu
            self.FastErrorCuSu = FastErrorCuSu

            self.refErrorCuSuStatsN = (refErrorCuSuN.mean(), refErrorCuSuN.std(), skew(refErrorCuSuN))
            self. sutErrorCuSuStatsN = (sutErrorCuSuN.mean(), sutErrorCuSuN.std(), skew(sutErrorCuSuN))
            self.SlowErrorCuSuStatsN = (SlowErrorCuSuN.mean(), SlowErrorCuSuN.std(),skew(SlowErrorCuSuN))
            self.FastErrorCuSuStatsN = (FastErrorCuSuN.mean(), FastErrorCuSuN.std(),skew(FastErrorCuSuN))
            self.index = refFastMA.index
            return (refErrorCuSuStats,sutErrorCuSuStats,SlowErrorCuSuStats,FastErrorCuSuStats)

        def DisplayChart(self, title=''):
            plt.figure(2)
            plt.plot(self.index,self.refErrorCuSu, 'k', label='Ref:FastEMA-SlowEMA')
            plt.plot(self.index,self.sutErrorCuSu, 'r', label='refFastEMA-sutSlowEMA')
            plt.plot(self.index,self.FastErrorCuSu, 'b', label='refFastEMA-sutFastEMA')
            plt.plot(self.index,self.SlowErrorCuSu, 'g', label='refSlowEMA-sutSlowEMA')
            plt.title(title)
            plt.legend(loc='upper left')
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
            plt.minorticks_on()
            plt.show(block=False)

            Colors = ['r', 'b', 'k', 'g']
            Markers = ['o', '*', 'x', 's']
            plt.figure(4)
            ax = plt.axes(projection='3d')
            ax.scatter(self.refErrorCuSuStatsN [0],self.refErrorCuSuStatsN[1], self.refErrorCuSuStatsN[2],
                       label='Ref:FastEMA-SlowEMA', color='k',marker=Markers[0])
            ax.scatter(self.sutErrorCuSuStatsN[0], self.sutErrorCuSuStatsN[1], self.sutErrorCuSuStatsN[2],
                       label='refFastEMA-sutSlowEMA', color='r',marker=Markers[1])
            ax.scatter(self.FastErrorCuSuStatsN[0], self.FastErrorCuSuStatsN[1], self.FastErrorCuSuStatsN[2],
                       label='refFastEMA-sutFastEMA',color='b',marker=Markers[2])
            ax.scatter(self.SlowErrorCuSuStatsN[0], self.SlowErrorCuSuStatsN[1], self.SlowErrorCuSuStatsN[2],
                       label='refSlowEMA-sutSlowEMA',color='g',marker=Markers[3])
           # plt.legend(loc='upper left')
            ax.legend()
            ax.set_xlim3d(0, 1)
            ax.set_ylim3d(0, 1)
            ax.set_zlim3d(-1, 1)
            ax.set_xlabel('Mean')
            ax.set_ylabel('Std')
            ax.set_zlabel('Skew')
            ax.grid(b=True, which='major', color='k', linestyle='-')
            ax.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
            ax.minorticks_on()
            plt.title(title)
            plt.show(block=True  )
            debug = 1
