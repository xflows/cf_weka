#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'vid, daleksovski'

import jpype as jp
import common
import utilities as ut

MAPPING_REPORT_START = 'Attribute mappings:'

def apply_mapped_classifier_get_instances(weka_classifier, original_data, data):
    '''An advanced version of the Apply Classifier method.
    Addresses incompatible training and test data, and returns a dataset with predictions.

    :param weka_classifier: WekaClassifier object
    :param original_data: original training instances, bunch
    :param data: test instances, bunch
    :return: Dataset (Bunch) object with predictions and a textual report from the InputMappedClassifier class
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    try:
        classifier = common.deserialize_weka_object(weka_classifier.sclassifier)
    except:
        raise Exception("Only WEKA classifiers/models supported. Please provide a valid WEKA learner.")

    original_training_instances = ut.convert_bunch_to_weka_instances(original_data)
    instances = ut.convert_bunch_to_weka_instances(data)

    # serialize classifier with original instances to a file once again for the Mapped classifier
    tfile = common.TemporaryFile(flags='wb+')
    s = jp.JClass('weka.core.SerializationHelper')
    s.writeAll(tfile.name, [classifier, original_training_instances])

    # construct a MappedClassifier
    mapped_classifier = jp.JClass('weka.classifiers.misc.InputMappedClassifier')()
    mapped_classifier.setIgnoreCaseForNames(True)
    mapped_classifier.setTrim(True)
    # mapped_classifier.setSuppressMappingReport(True)
    # mc.setModelHeader(original_training_instances)
    mapped_classifier.setModelPath(tfile.name)

    predictions = []
    try:
        for instance in instances:
            label = int(mapped_classifier.classifyInstance(instance))
            predictions.append(label)

        data["targetPredicted"] = predictions
    except:
        raise Exception("Classifier not built. Please use the Build Classifier widget first.")

    report = mapped_classifier.toString()
    if MAPPING_REPORT_START in report:
        report = report[report.index(MAPPING_REPORT_START):]

    return data, report
