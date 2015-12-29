__author__ = 'vid, darko'

import jpype as jp
import common, os
from os.path import join, normpath, dirname
import sklearn.datasets.base as sk
import numpy as np


def importDatasetFromArff(arff, classIndex=None):
    '''Imports Dataset From an ARFF Textual Format

    :param arff: the data in ARFF textual format
    :param classIndex: the index of the class attribute
    :return: a dataset (Bunch)
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    tmp = common.TemporaryFile(suffix='.arff')
    tmp.writeString(arff)

    source = jp.JClass('weka.core.converters.ConverterUtils$DataSource')(tmp.name)
    instances = source.getDataSet()

    if classIndex is None:
        print 'Warning: class is set to the last attribute!'
        classIndex = instances.numAttributes() - 1
    elif classIndex == -1:
        classIndex = instances.numAttributes() - 1

    instances.setClassIndex(classIndex)
    return convertWekaInstancesToBunch(instances)




def exportDatasetToArff(bunch):
    '''Exports a dataset to an ARFF file

    :param bunch: dataset
    :return:
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    instances = convertBunchToWekaInstances(bunch)

    fileContents = instances.toString()

    # numpy.savetxt("foo.csv", csv, fmt='%.6f', delimiter=",")

    fileOut = open('myfile','w')
    fileOut.write(fileContents)
    fileOut.close()

    output_dict = {}
    output_dict['FileOut'] = fileOut
    return output_dict



def printWekaModel(wekaClassifier):
    '''Outputs textual information about a Weka model

    :param wekaClassifier: a WekaClassifier object
    :return: a textual representation of the Weka model
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    try:
        model = common.deserializeWekaObject(wekaClassifier.sclassifier)
    except:
        raise Exception("Only WEKA classifiers/models supported. Please provide a valid WEKA learner.")

    return model.toString()


def getAttributeList(smodel):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    return common.deserializeWekaObject(smodel).toString()


def loadUciDatasetWeka(file_name = "cpu.arff"):
    '''Loads a UCI dataset from the ones provided along with this package

    :param f: the filename of the UCI file
    :return: dataset (Bunch)
    '''
    file_names = ["breast-cancer.arff","contact-lenses.arff","cpu.arff","cpu.with.vendor.arff","credit-g.arff","diabetes.arff","glass.arff","ionosphere.arff","iris.2D.arff","iris.arff","labor.arff","ReutersCorn-test.arff","ReutersCorn-train.arff","ReutersGrain-test.arff","ReutersGrain-train.arff","segment-challenge.arff","segment-test.arff","soybean.arff","supermarket.arff","unbalanced.arff","vote.arff","weather.nominal.arff","weather.numeric.arff"]
    if not(file_name in file_names):
        raise Exception("Illegal dataset requested.")

    baseDir = normpath(dirname(__file__))
    wekaDir = normpath(join(baseDir, 'weka'))
    dataDir = normpath(join(wekaDir, 'data'))

    f = dataDir + os.sep + file_name
    fi = open(f,'r')
    classificationDataset = fi.read()
    fi.close()

    bunch = importDatasetFromArff(classificationDataset)

    return bunch

def convertWekaInstancesToBunch(instances):
    ''' Converts WEKA Instances to the scikit Bunch format

    :param instances: WEKA dataset (Instances)
    :return:
    '''

    if instances.classIndex() < 0:
        instances.setClassIndex(instances.numAttributes() - 1)

    targetAtt = instances.classAttribute()
    targetNames = []
    if targetAtt.isNominal():
        for j in range(0, targetAtt.numValues()):
            targetNames.append( targetAtt.value(j) )

    feature_names = []

    numSamples      = instances.numInstances()
    numAttributes   = instances.numAttributes()
    numTargets      = 1

    data = np.empty((numSamples, numAttributes-numTargets))
    target = np.empty((numSamples,), dtype=np.int)

    fdescr = instances.relationName()

    featureValueNames = []
    for j in range(0,numAttributes-numTargets):
        myatt = instances.attribute(j)
        # mtype = 1 if myatt.isNumeric() else 0
        mname = myatt.name()
        feature_names.append(mname)

        num_vals = myatt.numValues()
        fVals=[]
        for k in range(0,num_vals):
            fVals.append(myatt.value(k))

        featureValueNames.append(fVals)

    for i in range(0,numSamples):
        arr = []
        for j in range(0,numAttributes):
            arr.append( instances.get(i).value(j) )

        data[i] = np.asarray(arr[:-1], dtype=np.float)
        if targetAtt.isNominal():
            target[i] = np.asarray(arr[-1], dtype=np.int)
        else:
            target[i] = np.asarray(arr[-1], dtype=np.float)


    return sk.Bunch(data=data, target=target,
                 target_names=targetNames,
                 DESCR=fdescr,
                 feature_value_names=featureValueNames,
                 feature_names=feature_names)   #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])


def convertBunchToWekaInstances(bunch):
    '''Converts scikit Bunch format to WEKA Instances

    :param bunch: dataset in scikit Bunch format; looks for field 'feature_value_names' for nominal features
    :return: dataset in WEKA Instances format
    '''


    util = jp.JPackage("java.util")
    wekaCore = jp.JPackage("weka.core")

    listAttrs = util.ArrayList()

    num_samples = bunch.data.shape[0]       #num_samples = 5
    num_features = bunch.data.shape[1]       #numAtt = 3
    num_targets = 1

    # create the features
    for j in range(0,num_features):
        if bunch.has_key("feature_value_names") and len(bunch.feature_value_names[j])>0:
            attVals = util.ArrayList()
            for v in bunch.feature_value_names[j]:
                attVals.add(v)

            # Constructor for nominal attributes and string attributes.
            att = wekaCore.Attribute(bunch.feature_names[j],  attVals)

        else:
            # Constructor for a numeric attribute with a particular index.
            att = wekaCore.Attribute(bunch.feature_names[j], j)

        listAttrs.add(att)

    numAtt = num_features + num_targets
    if len(bunch.target_names) > 0:
        # nominal target
        attVals = util.ArrayList()
        for v in bunch.target_names:
            attVals.add(v)
        # Constructor for nominal attributes and string attributes.
        targetAtt = wekaCore.Attribute("nominalClass",  attVals)

    else:
        # numeric target
        targetAtt = wekaCore.Attribute("numericClass", numAtt-1)

    listAttrs.add(targetAtt)

    inst = wekaCore.Instances(bunch.DESCR, listAttrs, num_samples)

    for i in range(0,num_samples):
        my_row = wekaCore.DenseInstance(numAtt)
        for j in range(0,num_features):
            my_row.setValue(j, bunch.data[i][j])

        my_row.setValue(numAtt-1, jp.JDouble(bunch.target[i]))
        # my_row.setValue(numAtt-1, bunch.target[i])

        inst.add(jp.JObject(my_row, 'weka.core.Instance'))

    inst.setClassIndex(inst.numAttributes()-1)

    # print inst.toString()

    return inst
