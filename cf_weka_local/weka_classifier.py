__author__ = 'darkoa'

from cf_datamining.classifier import Classifier
import common
import utilities as ut
import jpype as jp

class WekaClassifier(Classifier):

    def __init__(self, sclassifier):
        if not jp.isThreadAttachedToJVM():
            jp.attachThreadToJVM()

        self.sclassifier = sclassifier

        # can't keep 'classifier', since it cannot serialize (pickle) it
        # self.classifier = common.deserializeWekaObject(sclassifier)


    def buildClassifier(self, data):
        """Builds a classifier

        :param data: bunch
        """
        if not jp.isThreadAttachedToJVM():
            jp.attachThreadToJVM()

        instances = ut.convertBunchToWekaInstances(data)

        classifier = common.deserializeWekaObject(self.sclassifier)

        if instances.classIndex() == -1:
            instances.setClassIndex(instances.numAttributes()-1)
            # raise ValueError('Class not set!')

        classifier.buildClassifier(instances)
        self.sclassifier = common.serializeWekaObject(classifier)


    def applyClassifier(self, data):
        """Applies a classifier on a dataset, and gets predictions

        :param data: bunch
        :return: bunch with targetPredicted
        """
        if not jp.isThreadAttachedToJVM():
            jp.attachThreadToJVM()

        instances = ut.convertBunchToWekaInstances(data)

        classifier = common.deserializeWekaObject(self.sclassifier)

        classIndex = instances.classIndex()
        if classIndex == -1:
            raise ValueError('Class not set!')

        predictions = []
        for instance in instances:
            label = int(classifier.classifyInstance(instance))
            predictions.append(label)

        data["targetPredicted"] = predictions
        return data

    def printClassifier(self):
        if not jp.isThreadAttachedToJVM():
            jp.attachThreadToJVM()

        try:
            classifier = common.deserializeWekaObject(self.sclassifier)
            return classifier.toString()
        except:
            raise Exception("Only WEKA classifiers/models supported. Please provide a valid WEKA learner.")

