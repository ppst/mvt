#!/usr/bin/python3

import argparse
import math
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

if __name__ == 'train':
    import db
else:
    import mvt.db as db


__author__     = "Philip Stegmaier"
__contact__    = "https://github.com/ppst/mvt/issues"
__copyright__  = "Copyright (c) 2021, Philip Stegmaier"
__license__    = "https://en.wikipedia.org/wiki/MIT_License"
__maintainer__ = "Philip Stegmaier"
__version__    = "0.1.0"


class MultiVectorTrainer:
    """MultiVectorTrainer"""
    
    trace = []
    
    foldSelection = 0.1 # required to be > 0 and <= 0.5
    featureStart = 3
    minReposSize = 100
    iterNum = 0
    selectBest = True
    scaleFactor = 0.5
    
    n_estimators = 500
    n_jobs = multiprocessing.cpu_count()
    
    
    def get_default_model(self):
        """Returns a new default model (RandomForestClassifier).

        Returns
        -------
        RandomForestClassifier
            Default model type.
        """
        return RandomForestClassifier(n_estimators = self.n_estimators, n_jobs = self.n_jobs)
    
    
    def get_default_scale(self, iteration):
        """Returns the iteration-based scale for stochastic feature vector sampling.
        
        Parameters
        ----------
        iteration : int
            Iteration number
            
        Returns
        -------
        scale
            The factor to scale probability values in the exponent of the distribution function.
        """
        return self.scaleFactor * (iteration - 1) * (iteration - 1)
    
        
    def get_cmd_args(self):
        """Returns commandline arguments parsed by `argparse.ArgumentParser`
        """
        parser = argparse.ArgumentParser(description='Train ML model selecting from multiple feature vectors per data point')
    
        parser.add_argument('-d', '--data',
                            required = True,
                            type     = argparse.FileType('r'),
                            dest     = 'database',
                            help     = 'Input SQLite database')
        
        parser.add_argument('-m', '--model',
                            required = True,
                            dest     = 'modelFile',
                            help     = 'Output model file')
        
        parser.add_argument('-t', '--trace',
                            required = False,
                            dest     = 'traceFile',
                            help     = 'Output file for iteration info')
    
        parser.add_argument('-i', '--iterations',
                            required = False,
                            dest     = 'maxIterations',
                            type     = int,
                            default  = 100,
                            help     = 'Maximum number of iterations')
    
        parser.add_argument('-n', '--maxUnimproved',
                            required = False,
                            dest     = 'maxUnimproved',
                            type     = int,
                            default  = 50,
                            help     = 'Maximum number of iterations without improvement')
        
        parser.add_argument('-s', '--selectdist',
                            required = False,
                            dest = 'selectDist',
                            action='store_true',
                            help = 'Sample feature vectors from the distribution of vector probabilities for positioning')
        
        return parser.parse_args()
    
    
    def select_random_features(self, targetList, start, end, resultList, tsplits, traindb, coreNum):
        """Selects random feature vectors for list of data points.

        Parameters
        ----------
        targetList : array of arrays
        
        start : int
            Starting index in `targetList`
            
        end : int
            End index (not inclusive) in `targetList`
            
        resultList: list
            Output list
            
       tsplits : array of arrays of arrays
            Outer arrays correspond to classes, inner arrays contain data point indexes
        
        traindb : `db` object
            A `db` object connected to training database
            
        coreNum : int
            Number of the subprocess that called the function
        """
        vn = 0
        for t in range(start, end):
            x = targetList[t]
            labelNum = tsplits[x[0]][x[1]][0]
            vn += 1
            vnums = traindb.get_feature_vector_nums(labelNum)
            if len(vnums) == 0:
                raise ValueError("Error: received empty feature vector num list")
            resultList.append([x[0], x[1], random.choice(vnums)])
            if vn % 100 == 0:
                print(coreNum, vn, t)
        
        
    
    def initialize_vectors(self, tsplits, traindb):
        """Initializes training data points with randomly sampled feature vectors.
        
        Sets second element of inner arrays to database indexes of selected feature vectors.
        
        Parameters
        ----------
        tsplits : array of arrays of arrays
            Arrays correspond to classes. Inner arrays contain data point indexes as first element.
            
        traindb : `db` object
            A `db` object connected to training database
        """
        print("Initializing vectors")
        labelCount = len(tsplits[0]) + len(tsplits[1])
        coreGroups = self.get_core_groups(labelCount)
        mpiManager = multiprocessing.Manager()
        
        targetList = []
        resultList = mpiManager.list()
    
        for l in range(2):
            for t in range(len(tsplits[l])):
                targetList.append([l, t])
    
        processes = []
        grcount = 0
        coreNum = 1
        for cg in coreGroups:
            prc = multiprocessing.Process(target = self.select_random_features, args = (targetList, grcount, (grcount + cg), resultList, tsplits, traindb, coreNum))
            processes.append(prc)
            prc.start()
            grcount += cg
            coreNum += 1
    
        for prc in processes: prc.join()
        
        for r in resultList:
            tsplits[r[0]][r[1]][1] = r[2]
    

    def initialize_splits(self, labels):
        """Initializes training data splits.

        Parameters
        ----------
        labels : array of arrays
            Arrays correspond to classes 0, 1 and contain data point indexes (`label_num` database column)
            
        Returns
        -------
        splits
            Arrays containing initial assignment of data points to training and test sets
        """
        print("Initializing splits")
        trainSize = min(len(labels[0]), len(labels[1]))
        if not (self.foldSelection > 0.0 and self.foldSelection <= 0.5):
            raise ValueError("Error: invalid fold")
        reposSize = (int)(trainSize * self.foldSelection)
        if reposSize < self.minReposSize:
            raise ValueError("Error: invalid reposition size")
        print(trainSize, reposSize)
        retrainSize = trainSize - reposSize
        splits = [[[],[],[]], [[],[],[]]]
        for t in range(2):
            for x in labels[t]:
                if not x[1] == t:
                    raise ValueError("Error: label does not match index")
            labelNums = [x[0] for x in labels[t]]
            random.shuffle(labelNums)
            print(len(labelNums), retrainSize)
            splits[t][0] = [[x, -1] for x in labelNums[:retrainSize]]
            splits[t][1] = [[x, -1] for x in labelNums[retrainSize:trainSize]]
            if len(labelNums) > trainSize:
                splits[t][2] = [[x, -1] for x in labelNums[trainSize:]]
        return splits
        

    def make_training_set(self, tsplits, traindb):
        """Returns numpy array for model training

        Parameters
        ----------
        tsplits : array of arrays of arrays
            Lists with data point indexes
        
        traindb : `db` object
            A `db` object connected to training database
            
        Returns
        -------
        np.array
            Feature matrix for model training
        """
        print("Making training set")
        colnames = traindb.get_feature_columns()[self.featureStart:]
        features = []
        index = []
        for l in range(2):
            for p in tsplits[l]:
                features.append(traindb.get_feature_vector(p[0], p[1])[self.featureStart:])
                index.append(p[0])
        return np.array(pd.DataFrame(features, columns = colnames, index = index))
            

    def select_feature_vector(self, model, label, traindb, colnames, get_scale):
        """Selects feature vector for a data point using specified model.
        
        If `selectBest = False`, the feature vector is sampled according to probability of class = 1.
        Otherwise the first feature vector with highest probability is selected.

        Parameters
        ----------
        model : 
            Model used to select feature vector
        
        label : list
            Containing the data point database index (`label_num`) as first element
        
        traindb : `db` object
            A `db` object connected to training database
            
        colnames : list
            Feature column names
            
        get_scale : function
            Function that takes the iteration number as argument and returns a scale factor.
            Required for stochastic sampling only
        
        Returns
        -------
        index
            Database index of selected feature vector (column `feature_vector_num`)
        
        mxp
            Max. probability of class = 1 of specified data point
        
        label
            The predicted label for data point corresponding to `mxp`
        """
        vectors = traindb.get_feature_vectors(label[0])
        index = [x[2] for x in vectors]
        features = np.array(pd.DataFrame([x[self.featureStart:] for x in vectors], columns = colnames, index = index))
        pred = model.predict_proba(features)
        mxp = -1.0
        mxi = 0
        for r in range(len(pred)):
            if pred[r][1] > mxp:
                mxi = r
                mxp = pred[r][1]
        if not self.selectBest:
            dist = [0.0] * len(pred)
            part = 0.0
            for r in range(len(pred)):
                dist[r] = math.exp((pred[r][1] - mxp) * get_scale(self.iterNum))
                part += dist[r]
            prob = 0.0
            pr = random.random()
            for r in range(len(pred)):
                dist[r] = dist[r]/part
                if pr >= prob and pr <= dist[r] + prob:
                    mxi = r
                    break
                prob += dist[r]
        return index[mxi], mxp, (1 if mxp > 0.5 else 0)


    def select_feature_vectors(self, targetList, start, end, resultList, model, tsplits, traindb, colnames, get_scale, coreNum):
        """Selects feature vectors for data points using specified model.
        
        The output list contains arrays with class index, data point index, 
        selected feature vector index, predicted probability of class 1 and the predicted label.

        Parameters
        ----------
        targetList : array of arrays
            Data point labels and indexes
        
        start : int
            Starting index in `targetList`
            
        end : int
            End index (not inclusive) in `targetList`
            
        resultList: list
            Output list
            
        model:
            Model used to select feature vectors
            
        tsplits : array of arrays of arrays
            Outer arrays correspond to classes, inner arrays contain data point indexes
            
        traindb : `db` object
            A `db` object connected to training database
            
        colnames : list
            Feature column names
            
        get_scale : function
            Function that takes the iteration number as argument and returns a scale factor.
            Required for stochastic sampling only
            
        coreNum : int
            Number of the subprocess that called the function
        """
        for t in range(start, end):
            x = targetList[t]
            vnum, prob, label = self.select_feature_vector(model, tsplits[x[0]][x[1]], traindb, colnames, get_scale)
            resultList.append([x[0], x[1], vnum, prob, label])
            
            
    def get_core_groups(self, num):
        """Returns number of items to be assigned to cores

        Parameters
        ----------
        num : int
            Number of items to distribute
            
        Returns
        -------
        coreGroups
            a list of integers specifying the number of items per core
        """
        procNum = multiprocessing.cpu_count()
        labelPerCore = int(math.floor(num/procNum))
        coreGroups = [labelPerCore] * procNum
        cpcDiff = num - (labelPerCore * procNum)
        for c in range(cpcDiff):
            coreGroups[c] += 1
        print(coreGroups)
        return coreGroups
        
        
    def position_features(self, tsplits, traindb, model, get_scale):
        """Selects feature vectors for test data using specified model.

        Parameters
        ----------
        tsplits : array of arrays of arrays
            Test data points
            
        traindb : `db` object
            A `db` object connected to training database
        
        model :
            Model used to select feature vectors
        
        get_scale : function
            Function that returns the iteration-based scale factor for stochastic selection
            
        Returns
        -------
        labels
            list of known labels
            
        predLabels
            list of predicted labels
            
        nsplits
            Test data points with updated feature vector selection
        """
        print("Positioning features")
        
        labelCount = len(tsplits[0]) + len(tsplits[1])
        coreGroups = self.get_core_groups(labelCount)
        mpiManager = multiprocessing.Manager()
        colnames = traindb.get_feature_columns()[self.featureStart:]
        
        targetList = []
        resultList = mpiManager.list()
    
        for l in range(2):
            for t in range(len(tsplits[l])):
                targetList.append([l, t])
    
        processes = []
        grcount = 0
        coreNum = 1
        for cg in coreGroups:
            prc = multiprocessing.Process(target = self.select_feature_vectors, args = (targetList, grcount, (grcount + cg), resultList, model, tsplits, traindb, colnames,  get_scale, coreNum))
            processes.append(prc)
            prc.start()
            grcount += cg
            coreNum += 1
    
        for prc in processes: prc.join()
    
        labels = []
        predLabels = []
        nsplits = [[], []]
        for p in resultList:
            nsplits[p[0]].append([ tsplits[p[0]][p[1]][0], p[2] ])
            labels.append(p[0])
            predLabels.append(p[4])
            
        return labels, predLabels, nsplits
    

    def _update_splits(self, splits):
        """Updates training and test sets.
        
        Newly selected feature vectors are moved from test
        to training subsets.
        New test sets are created from training data replaced
        by new feature vectors and/or excess data points from the
        larger class.

        Parameters
        ----------
        splits : array of arrays of arrays
            Training data splits into training and test subsets for classes.
            As produced by `initialize_splits(labels)`
        """
        print("Updating splits")
        retrainSize = len(splits[0][0])
        reposSize   = len(splits[0][1])
        for l in range(2):
            mr = min(reposSize, retrainSize)
            moveRetrain = splits[l][0][:mr]
            splits[l][0] = splits[l][0][reposSize:] + splits[l][1]
            if len(splits[l][2]) == 0:
                splits[l][1] = moveRetrain
            else:
                extSize = len(splits[l][2])
                if extSize >= reposSize:
                    splits[l][1] = splits[l][2][:reposSize]
                    if extSize > reposSize:
                        splits[l][2] = splits[l][2][reposSize:] + moveRetrain
                    else:
                        splits[l][2] = moveRetrain
                else:
                    reposDiff = reposSize - extSize
                    splits[l][1] = splits[l][2] + moveRetrain[:reposDiff]
                    splits[l][2] = moveRetrain[reposDiff:]
    
    
    def iterate(self, splits, traindb, maxIterations, maxUnimproved, get_model, get_scale):
        """Performs training iterations and returns the model that achieved 
           highest accuracy.

        Parameters
        ----------
        splits : array of arrays of arrays
            Training data splits into training and test subsets for classes.
            As produced by `initialize_splits(labels)`
            
        traindb : a `db` object connected to the training database
        
        maxIterations : int
            Max. number of iterations
            
        maxUnimproved : int
            Max. number of iterations without improvement
            
        get_model : function
            Function that takes no parameters and returns a new model object
            
        get_scale : function
            Function that takes the iteration number as argument and returns a scale factor.
            Required for stochastic sampling only
            
        Returns
        -------
        bestModel
            The model that achieved highest accuracy on the test set
        """
        self.initialize_vectors([splits[0][0], splits[1][0]], traindb)
        labels = [0] * len(splits[0][0])  + [1] * len(splits[1][0])
        self.iterNum = 0
        unimproved = 0
        bestModel    = None
        bestAccuracy = 0.0
        
        while self.iterNum < maxIterations and unimproved < maxUnimproved:
            self.iterNum += 1
            print("Iteration", self.iterNum, unimproved)
            features = self.make_training_set([splits[0][0], splits[1][0]], traindb)
        
            print("Training random forest")
            model = get_model()
            model.fit(features, labels)
        
            reposLabels, predLabels, nextSplits = self.position_features([splits[0][1], splits[1][1]], traindb, model, get_scale)
                
            accuracy = accuracy_score(reposLabels, predLabels)
            
            if self.iterNum > 1:
                bestReposLabels, bestPredLabels, bestNextSplits = self.position_features([splits[0][1], splits[1][1]], traindb, bestModel, get_scale)
                bestAccuracy = accuracy_score(bestReposLabels, bestPredLabels)
            
            print(accuracy, "(", bestAccuracy, ")")
            
            if accuracy > bestAccuracy:
                bestModel = model
                bestAccuracy = accuracy
                splits[0][1] = nextSplits[0]
                splits[1][1] = nextSplits[1]
                unimproved = 0
                print(confusion_matrix(reposLabels, predLabels))
                print(classification_report(reposLabels, predLabels))
            
            else:
                splits[0][1] = bestNextSplits[0]
                splits[1][1] = bestNextSplits[1]
                unimproved += 1
                print(confusion_matrix(bestReposLabels, bestPredLabels))
                print(classification_report(bestReposLabels, bestPredLabels))
            
            self.trace.append([self.iterNum, accuracy, bestAccuracy, unimproved, get_scale(self.iterNum)])
            self._update_splits(splits)
        return bestModel
            
    
    def train(self, dbfile, maxIterations, maxUnimproved, get_model, get_scale):
        """Performs training with known labels and multiple feature vectors per example
           and returns the model that achieved highest accuracy.
        
        Parameters
        ----------
        dbfile : str
            SQLite parameter database file
            
        maxIterations : int
            Max. number of iterations
            
        maxUnimproved : int
            Max. number of iterations without improvement
            
        get_model : function
            Function that takes no parameters and returns a new model object
            
        get_scale :
            Function that takes the iteration number as argument and returns a scale factor.
            Required for stochastic sampling only
            
        Returns
        -------
        model
            The model that achieved highest accuracy on the test set
        """
        traindb = db.Database()
        print("Connecting to database")
        traindb.connect(dbfile)
        labels    = traindb.get_labels()
        print("0: %i, 1: %i, <%i>" % (len(labels[0]), len(labels[1]), len(labels)))
        splits = self.initialize_splits(labels)
        model = self.iterate(splits, traindb, maxIterations, maxUnimproved, get_model, get_scale)
        traindb.close()
        print("Disconnected")
        return model
    
    
    def dumpTrace(self, traceFile):
        """Writes contents of `trace` to file.

        Parameters
        ----------
        traceFile : str
            Path for output file
        """
        OF = open(traceFile, 'w')
        OF.write("Iteration\tAccuracy\tBest.accuracy\tUnimproved\tIteration.scale\n")
        for t in self.trace:
            OF.write("%i\t%.5f\t%.5f\t%i\t%.5f\n" % (t[0], t[1], t[2], t[3], t[4]))
        OF.close()
    
    
    def main(self):
        """Performs training according to parameters specified in `sys.argv`.
        """
        cmdArgs = self.get_cmd_args()
        self.selectBest = not cmdArgs.selectDist
        model = self.train(cmdArgs.database.name, cmdArgs.maxIterations, cmdArgs.maxUnimproved, self.get_default_model, self.get_default_scale)
        pickle.dump(model, open(cmdArgs.modelFile, 'wb'), pickle.HIGHEST_PROTOCOL)
        if not cmdArgs.traceFile == None and len(cmdArgs.traceFile) > 0:
            self.dumpTrace(cmdArgs.traceFile)


if __name__ == '__main__':
    MultiVectorTrainer().main()
