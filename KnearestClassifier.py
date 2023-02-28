# K - nearest Neighbor Classifier
from sklearn.model_selection import train_test_split
from collections import Counter
from matplotlib import pyplot as plt
import pickle,os,argparse, logging,types
import numpy as np
import sys, json
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D



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




def save_json(obj=[], dir=[], filename=[]):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + filename + '.json','w') as f:
        f.write(obj)



def save_pickle(obj=[], dir=[], filename=[]):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(dir=[], filename=[]):
    with open(dir + filename + '.pkl', 'rb') as f:
        return pickle.load(f)


# The previous approach took only the ranking of the neighbors according to their distance in account. We can improve
# the voting by using the actual distance. To this purpose we will write a new voting function:

def vote_distance_weights(neighbors, all_results=True):
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    for index in range(number_of_neighbors):
        dist = neighbors[index][1]
        label = int(neighbors[index][2])
        # Counter ({Key1:Value1, Key2:Value2})... ({Label0:Distance Metric, Label1:Distance Metri})
        class_counter[label] += 1 / (dist**2 + 1)
        debug = 1
    # Sort from most common to least common
    labels, votes = zip(*class_counter.most_common())
    # print(labels, votes)
    # class_counter.most_common(1), Returns a List with a tuple i.e. ({Key:Value})
    # class_counter.most_common(1)[0][0] returns the Key
    winner = class_counter.most_common(1)[0][0]
    # class_counter.most_common(1)[0][0] returns the Value
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
             class_counter[key] /= total
        return int(winner), class_counter.most_common()
    else:
        return int(winner), votes4winner / sum(votes)


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def Visualize(learn_data=[],learn_labels=[],c0=[],c1=[],FeatSelect = [],title=''):

    # Centroids
    if FeatSelect == 4:
        c0 = c0[0],c0[1], np.sum([c0[2],c0[3]])/2
        c1 = c1[0], c1[1], np.sum([c1[2], c1[3]])/2
    elif FeatSelect == 3:
        c0 = c0[0], c0[1], c0[2]/2
        c1 = c1[0], c1[1], c1[2]/2

    X = []
    for iclass in range(2):
        X.append([[], [], []])
        for i in range(len(learn_data)):
            if learn_labels[i] == iclass:
                X[iclass][0].append(learn_data[i][0])
                X[iclass][1].append(learn_data[i][1])
                X[iclass][2].append(sum(learn_data[i][2:])/2)


    colours = ("r", "g", "y")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for iclass in range(2):
        ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colours[iclass])
        # ax.scatter(X[iclass][0], X[iclass][1], c=colours[iclass])
    ax.scatter(c0[0],c0[1],c0[2], c='k', label= 'Centroid L0')
    ax.scatter(c1[0], c1[1],c1[2],c = 'b', label='Centroid L1')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
  #  AllOthersStr = np.unique(learn_labels) + 'Centroid L0'+'Centroid L2'
    AllOthersStr = ['Label 0','Label 1','Centroid Label 0','Centroid Lable 1']
    plt.title(title)
    ax.legend(AllOthersStr, numpoints=1, loc='upper right')
    plt.show(block = True)


def distance(instance1, instance2):
    """ Calculates the Eucledian distance between two instances"""
    return np.linalg.norm(np.subtract(instance1, instance2))


#  write a vote function. This functions uses the class Counter from collections to count the quantity of
# the classes inside of an instance list. This instance list will be the neighbors of course. The function vote returns
# the most common class:
def vote(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        # Counter Labels
        class_counter[neighbor[2]] += 1
        debug = 1
    return class_counter.most_common(1)[0][0]

# To pursue this strategy, we can assign weights to the neighbors in the following way: The nearest neighbor of an
# instance gets a weight, the second closest gets a weight of and then going on up to for the farthest away neighbor.
# This means that we are using the harmonic series as weights:


def vote_harmonic_weights(neighbors, all_results=True):
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    for index in range(number_of_neighbors):
        class_counter[neighbors[index][2]] += 1/(index+1)
    labels, votes = zip(*class_counter.most_common())
    #print(labels, votes)
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
             class_counter[key] /= total
        return winner, class_counter.most_common()
    else:
        return winner, votes4winner / sum(votes)


# 'vote_prob' is a function like 'vote' but returns the class name and the probability for this class:
def vote_prob(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
    labels, votes = zip(*class_counter.most_common())
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    return winner, votes4winner/sum(votes)


# 'vote_prob' is a function like 'vote' but returns the class name and the probability for this class:
def FinalVote_prob(results):
    class_counter = Counter()
    for SegKey, SegValue in results.items():
        for IdxKey, IdxValue in SegValue.items():
            class_counter[IdxValue['PredictedLabel']] += 1
            debug =1
        debug = 1
    labels, votes = zip(*class_counter.most_common())
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    return winner, votes4winner/sum(votes)



def get_neighbors(training_set,
                  labels,
                  test_instance,
                  test_label,
                  k,
                  distance):
    """
    get_neighors calculates a list of the k nearest neighbors
    of an instance 'test_instance'.
    The function returns a list of k 3-tuples.
    Each 3-tuples consists of (index, dist, label)
    where
    index    is the index from the training_set,
    dist     is the distance between the test_instance and the
             instance training_set[index]
    distance is a reference to a function used to calculate the
             distances
    """
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index], test_label))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors

def Visualization(dataDict=[],config=[]):

    # Label0Idxs = np.where(learn_labels==0)[0]
    # Label1Idxs = np.where(learn_labels == 1)[0]

    MarkerSize = 20 * 4 ** 2
    pfp_log.info('Plotting Results')
    for SegIdx in range(config['CommonParams']['Segments']):
        LabelStr = 'SegMLdata-{}'.format(SegIdx)

        Colors = ['r', 'b', 'k', 'g']
        Markers = ['o', '*', 'x', '.']
        fig = plt.figure(SegIdx)
        ax = plt.axes(projection='3d')
        # plt.title('Means: ' + V1TestList[ii])
        plt.title(config['Dut']['OutputParams']['TitleStr'] + ' Segment {}'.format(SegIdx))
      #  x1 = dataDict[LabelStr]['X_train'][np.where(dataDict[LabelStr]['y_train'][:] == 0)[0], :]

        ax.scatter(dataDict[LabelStr]['X_train'][np.where(dataDict[LabelStr]['y_train'][:] == 0)[0], 0],
                   dataDict[LabelStr]['X_train'][np.where(dataDict[LabelStr]['y_train'][:] == 0)[0], 1],
                   dataDict[LabelStr]['X_train'][np.where(dataDict[LabelStr]['y_train'][:] == 0)[0], 2],
                   c=Colors[0], marker=Markers[0])
        ax.scatter(dataDict[LabelStr]['X_train'][np.where(dataDict[LabelStr]['y_train'][:] == 1)[0], 0],
                   dataDict[LabelStr]['X_train'][np.where(dataDict[LabelStr]['y_train'][:] == 1)[0], 1],
                   dataDict[LabelStr]['X_train'][np.where(dataDict[LabelStr]['y_train'][:] == 1)[0], 2],
                   c=Colors[1], marker=Markers[1])

        # Centroids
        ax.scatter(dataDict[LabelStr]['Label0_X_trainCentroid'][0],dataDict[LabelStr]['Label0_X_trainCentroid'][1],
                   dataDict[LabelStr]['Label0_X_trainCentroid'][2],s=MarkerSize, c=Colors[0], marker='x')
        ax.scatter(dataDict[LabelStr]['Label1_X_trainCentroid'][0],dataDict[LabelStr]['Label1_X_trainCentroid'][1],
                   dataDict[LabelStr]['Label1_X_trainCentroid'][2], s=MarkerSize, c=Colors[1], marker='x')

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

def run(config= []):


    RefDict = load_pickle(config['Ref']['OutputParams']['Path'],config['Ref']['OutputParams']['FileName'])
    DutDict = load_pickle(config['Dut']['OutputParams']['Path'],config['Dut']['OutputParams']['FileName'])

    Results = {}
    # Here we need assign Labels to the Ref & Dut Features for the Nearest Neighbor algorithm to work. We assign
    # Label 0 to the Ref Features and Label 1 to the Dut Features

    RefFeatures = {'Data': np.array([[RefDict['ResultsBB'][i, :], RefDict['ResultsCu'][i, :, 0],
                                          RefDict['ResultsCu'][i, :, 1], RefDict['ResultsMACD'][i, :]] for
                                         i in range(len(RefDict['ResultsMACD']))]),
                       'Labels': np.zeros((len(RefDict['ResultsMACD'])), dtype=int)}


    DutFeatures = {'Data': np.array([[DutDict['ResultsBB'][i, :], DutDict['ResultsCu'][i, :, 0],
                                          DutDict['ResultsCu'][i, :, 1], DutDict['ResultsMACD'][i, :]] for i
                                         in range(len(RefDict['ResultsMACD']))]),
                       'Labels': np.ones((len(DutDict['ResultsMACD'])), dtype=int)}


    # Shuffle the Data
    np.random.seed(42)
    indices = np.random.permutation(len(RefFeatures['Data']))


    # if config['OutputParams']['Dataexplore'] == True:
    # This section is for Testing with both Labels, this gives insight to the data
    CombinedData =  np.concatenate((RefFeatures['Data'][indices], DutFeatures['Data'][indices]),axis=0)
    CombinedLabels =  np.concatenate((RefFeatures['Labels'][indices], DutFeatures['Labels'][indices]),axis=0)

    # Normalize (0-1) the data in each segment
    scaler = MinMaxScaler()
    for SegIdx in range(0,config['CommonParams']['Segments'],1):
        CombinedData[:,:,SegIdx] = scaler.fit_transform(CombinedData[:,:,SegIdx])



    # Split into Test
    SegMLdata = {}

    for SegIdx in range(0, config['CommonParams']['Segments'], 1):
        LabelStr = 'SegMLdata-{}'.format(SegIdx)
        SegMLdata[LabelStr] = {}
        X_train, X_test, y_train, y_test = train_test_split(CombinedData[:,:,SegIdx], CombinedLabels, test_size=0.33, random_state=42)

        SegMLdata[LabelStr]['X_train'] = X_train
        SegMLdata[LabelStr]['X_test'] = X_test
        SegMLdata[LabelStr]['y_train'] = y_train
        SegMLdata[LabelStr]['y_test'] = y_test

        #  Calculate centroid
        L0_train = X_train[np.where(y_train == 0)[0]]
        L0_test = X_test[np.where(y_test == 0)[0]]
        L1_train = X_train[np.where(y_train == 1)[0]]
        L1_test = X_test[np.where(y_test == 1)[0]]

        L0_trainCentroid = np.mean(L0_train[:,0]),np.mean(L0_train[:,1]),np.mean(L0_train[:,2]),np.mean(L0_train[:,3])
        L0_testCentroid = np.mean(L0_test[:,0]),np.mean(L0_test[:,1]),np.mean(L0_test[:,2]),np.mean(L0_test[:,3])
        L1_trainCentroid = np.mean(L1_train[:, 0]), np.mean(L1_train[:, 1]), np.mean(L1_train[:, 2]), np.mean(
            L1_train[:, 3])
        L1_testCentroid = np.mean(L1_test[:, 0]), np.mean(L1_test[:, 1]), np.mean(L1_test[:, 2]), np.mean(
            L1_test[:, 3])

        SegMLdata[LabelStr]['Label0_X_trainCentroid'] = L0_trainCentroid
        SegMLdata[LabelStr]['Label0_X_testCentroid'] = L0_testCentroid
        SegMLdata[LabelStr]['Label1_X_trainCentroid'] = L1_trainCentroid
        SegMLdata[LabelStr]['Label1_X_testCentroid'] = L1_testCentroid




    Visualization(dataDict=SegMLdata,config=config)
    # Visualize(learn_data=X_test, learn_labels=y_test,c0=L0_testCentroid,c1= L1_testCentroid,FeatSelect=FeatSelect,title="test data, seg {}".format(SegIdx))

    pfp_log.debug("")
    pfp_log.debug("")
    pfp_log.debug("The first samples of our train set in Segment {}:".format(SegIdx))
    pfp_log.debug(f"{'Trace index':7s}{'data':20s}{'label':3s}")
    for i in range(min(3,min(len(X_train),len(y_train)))):
        pfp_log.debug(f"{i:4d}   {X_train[i]}   {y_train[i]:3}")

    pfp_log.debug("The first samples of our test set:")
    pfp_log.debug(f"{'Trace index':7s}{'data':20s}{'label':3s}")
    for i in range(min(3,min(len(X_test),len(y_test)))):
        pfp_log.debug(f"{i:4d}   {X_test[i]}   {y_test[i]:3}")

    #    from random import shuffle
    #    shuffle(y_train) # use this to test Voting prob.
    # We will test 'vote_distance_weights' on our test data:
    SegResults = {}

    # Extract the label of Interest, Typically Dut label 1
    Label = 1
    idx = np.where(y_test == Label)[0]
    yy_test = y_test[idx]
    XX_test = X_test[idx]
    #  X_train = [L0_trainCentroid,L1_trainCentroid]
    #  y_train = [0,1]
    for i in range(len(XX_test)):
        # neighbors = Each 3 - tuples consists of (training_set[index], dist, label[index)
        # Get the N nearest neighbors to y_test[i] from X_train
        neighbors = get_neighbors(X_train, y_train, XX_test[i], yy_test[i], config['Knearest']['InputParams']['K'],
                                  distance=distance, )

        #  neighborsC = get_neighbors([L0_trainCentroid, L1_trainCentroid] ,[0, 1] ,XX_test[i], yy_test[i], config['InputParams']['K'], distance=distance)

        vdw = vote_distance_weights(neighbors, all_results=True)
        IdxKey = 'Idx{}'.format(i)


        if len(vdw[1]) == 1:
            SegResults[IdxKey] = {'PredictedLabel': vdw[0],
                                  'HighestScoreLabel': vdw[1][0][0],
                                  'HighScore':vdw[1][0][1]}
        elif len(vdw[1]) == 2:
            SegResults[IdxKey] = {'PredictedLabel': vdw[0],
                                  'HighestScoreLabel': vdw[1][0][0],
                                  'HighScore': vdw[1][0][1],
                                  'LowestScoreLabel': vdw[1][1][0],
                                  'LowScore': vdw[1][1][1]
                                  }





        #SegResults[IdxKey] = vdw
        pfp_log.debug("index: {}, result of distance weights vote:{} ".format(i,vote_distance_weights(neighbors, all_results=True)))

    s = 'Segment{}'.format(SegIdx)
    Results[s] = SegResults

    PredictedLabel, LabelVote = FinalVote_prob(Results)
    pfp_log.info('Predicted Label {} Percent Vote {}'.format(PredictedLabel,LabelVote))
    s = 'FinalVote'
    Results[s] = {'PredictedLabel':PredictedLabel,'PercentVote':LabelVote}
    if config['Knearest']['OutputParams']['SaveResults'] == True:
        pfp_log.info('Saving results to {}{}'.format(config['Knearest']['OutputParams']['Path'],config['Knearest']['OutputParams']['FileName']))
        Results3 = json.dumps(Results,indent=4)
        # Save JSON checks if the directory exist and creates it if not
        save_json(dir=config['Knearest']['OutputParams']['Path'],filename=config['Knearest']['OutputParams']['FileName'],obj=Results3)
    debug = 1


if __name__ == '__main__':

    if configFile != []:
        with open(configFile) as json_file:
            config = json.load(json_file)
            json_file.close()

    #    run(config)

    sys.exit(run(config))
