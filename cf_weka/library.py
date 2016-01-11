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
    p = input_dict['params']
    return {'learner': classification.j48(p)}  # '-C 0.25 -M 2'


def naive_bayes(input_dict):
    """Naive Bayes learner"""
    p = input_dict['params']
    return {'learner': classification.naive_bayes(p)}


def random_forest(input_dict):
    """Random Forest learner"""
    p = input_dict['params']
    return {'learner': classification.random_forest(p)}


def multilayer_perceptron(input_dict):
    """MLP Neural-network learner"""
    p = input_dict['params']
    return {'learner': classification.multilayer_perceptron(p)}


def smo(input_dict):
    """SVM learner"""
    p = input_dict['params']
    return {'learner': classification.smo(p)}


def logistic_regression_weka(input_dict):
    """Logistic Regression learner"""
    p = input_dict['params']
    return {'learner': classification.logistic(p)}


def rules_zeror(input_dict):
    """rulesZeroR Rule learner"""
    p = input_dict['params']
    return {'learner': classification.rules_zeror(p)}


def rules_jripper(input_dict):
    """Rule learner JRipper"""
    p = input_dict['params']
    return {'learner': classification.rules_jrip(p)}


def knn(input_dict):
    """K-Nearest-Neighbours learner IBk"""
    p = input_dict['params']
    return {'learner': classification.ibk(p)}


def random_tree(input_dict):
    """Random Tree learner"""
    p = input_dict['params']
    return {'learner': classification.random_tree(p)}


def rep_tree(input_dict):
    """Reduced Error Pruning tree"""
    p = input_dict['params']
    return {'learner': classification.rep_tree(p)}

def k_star(input_dict):
    """K* is an instance-based classifier, that is the class of a test instance is based upon the class of those training instances similar to it, as determined by some similarity function"""
    p = input_dict['params']
    return {'learner': classification.k_star(p)}

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

    arff_file_contents = utilities.export_dataset_to_arff(input_dict['dataset'])

    file_out = open('myfile', 'w')
    file_out.write(arff_file_contents)
    file_out.close()

    output_dict = {}
    output_dict['file_out'] = file_out
    return output_dict


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
