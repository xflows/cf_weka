#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'daleksovski'

from cf_data_mining.classifier import Classifier
import common
import utilities as ut
import jpype as jp


class WekaClassifier(Classifier):
    def __init__(self, sclassifier):
        if not jp.isThreadAttachedToJVM():
            jp.attachThreadToJVM()

        self.sclassifier = sclassifier

    def build_classifier(self, data):
        """Builds a classifier

        :param data: bunch
        """
        if not jp.isThreadAttachedToJVM():
            jp.attachThreadToJVM()

        instances = ut.convert_bunch_to_weka_instances(data)

        classifier = common.deserialize_weka_object(self.sclassifier)

        if instances.classIndex() == -1:
            instances.setClassIndex(instances.numAttributes() - 1)
            # raise ValueError('Class not set!')

        classifier.buildClassifier(instances)
        self.sclassifier = common.serialize_weka_object(classifier)

    def apply_classifier(self, data):
        """Applies a classifier on a dataset, and gets predictions

        :param data: bunch
        :return: bunch with targetPredicted
        """
        if not jp.isThreadAttachedToJVM():
            jp.attachThreadToJVM()

        instances = ut.convert_bunch_to_weka_instances(data)

        classifier = common.deserialize_weka_object(self.sclassifier)

        class_index = instances.classIndex()
        if class_index == -1:
            raise ValueError('Class not set!')

        predictions = []
        for instance in instances:
            label = int(classifier.classifyInstance(instance))
            predictions.append(label)

        data["targetPredicted"] = predictions
        return data

    def print_classifier(self):
        if not jp.isThreadAttachedToJVM():
            jp.attachThreadToJVM()

        try:
            classifier = common.deserialize_weka_object(self.sclassifier)
            return classifier.toString()
        except:
            raise Exception("Only WEKA classifiers/models supported. Please provide a valid WEKA learner.")
