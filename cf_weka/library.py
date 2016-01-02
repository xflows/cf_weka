#!/usr/bin/env python
# -*- coding: utf-8 -*-
__authors__ = 'vid, daleksovski'

import classification
import evaluation
import utilities
import preprocessing

#
# CLASSIFICATION ALGORITHMS
#

def decision_tree_j48(input_dict):
    """Decision Tree learner J48"""
    return {'learner': classification.j48('-C 0.25 -M 2')}  # -C 0.25 -M 2


def naive_bayes(input_dict):
    """Naive Bayes learner"""
    return {'learner': classification.naive_bayes()}


def random_forest(input_dict):
    """Random Forest learner"""
    return {'learner': classification.random_forest()}


def multilayer_perceptron(input_dict):
    """MLP Neural-network learner"""
    return {'learner': classification.multilayer_perceptron()}


def smo(input_dict):
    """SVM learner"""
    return {'learner': classification.smo()}


def logistic_regression(input_dict):
    """Logistic Regression learner"""
    return {'learner': classification.logistic()}


def rules_zeror(input_dict):
    """rulesZeroR Rule learner"""
    return {'learner': classification.rules_zeror()}


def rules_jripper(input_dict):
    """Rule learner JRipper"""
    return {'learner': classification.rules_jrip()}


def knn(input_dict):
    """K-Nearest-Neighbours learner IBk"""
    return {'learner': classification.ibk()}


def random_tree(input_dict):
    """Random Tree learner"""
    return {'learner': classification.random_tree()}


def rep_tree(input_dict):
    return {'learner': classification.rep_tree()}


#
# PREPROCESSING
#

def feature_selection(input_dict):
    """Correlation-based Feature Subset Selection"""
    instances = input_dict['instances']
    output_dict = {}
    output_dict['selected'] = preprocessing.correlation_basedfeat_sel(instances)
    return output_dict


def normalize(input_dict):
    """Normalizes all numeric values in the given dataset"""
    instances = input_dict['instances']
    output_dict = {}
    # 1,0 -> normalize to [0,1]; 2,-1 then to [-1,1]
    output_dict['normalized'] = preprocessing.normalize(instances, '-S 2.0 -T -1.0')
    return output_dict


#
# EVALUATION
#

def apply_mapped_classifier_get_instances(input_dict):
    """An advanced version of the Apply Classifier method"""
    sclassifier = input_dict['classifier']
    soriginalInstances = input_dict['original_training_instances']
    sinstances = input_dict['instances']

    instances, report = evaluation.apply_mapped_classifier_get_instances(sclassifier, soriginalInstances, sinstances)

    output_dict = {'instances': instances, 'mapping_report': report}
    return output_dict

#
# UTILITIES
#
def export_dataset_to_arff(input_dict):
    """Export Dataset to an ARFF Textual Format"""
    return {}


def import_dataset_from_arff(input_dict):
    """Imports Dataset From an ARFF Textual Format"""
    arff = input_dict['arff']
    output_dict = {}
    output_dict['instances'] = utilities.import_dataset_from_arff(arff)
    return output_dict


def load_uci(input_dict):
    """Loads a UCI dataset"""
    arff_file = input_dict['filename']
    output_dict = {}
    output_dict['data'] = utilities.load_uci_dataset_weka(arff_file)
    return output_dict
