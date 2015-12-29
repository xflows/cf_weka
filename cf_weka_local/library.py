#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cf_weka_local.preprocessing

__author__ = 'vid, darkoa'

import classification
import evaluation
import utilities

#
# CLASSIFICATION ALGORITHMS
#

# Optimalne vrednosti parametrov: -C 0.5 -M 2 za vlake
def wekaLocalDecisionTreeJ48(input_dict):
    """Decision Tree learner J48"""
    return {'learner': classification.J48('-C 0.25 -M 2')} #-C 0.25 -M 2

def wekaLocalNB(input_dict): # =None sicer ne dela iz py konzole
    """Naive Bayes learner"""
    return {'learner': classification.naiveBayes()} # learner v widgetu!

def wekaLocalRandomForest(input_dict):
    """Random Forest learner"""
    return {'learner': classification.RandomForest()}

def wekaLocalMultilayerPerceptron(input_dict):
    """MLP Neural-network learner"""
    return {'learner': classification.Multilayer_Perceptron()}

def wekaLocalSMO(input_dict):
    """SVM learner"""
    return {'learner': classification.SMO()}


def wekaLocalLogisticRegression(input_dict):
    """Logistic Regression learner"""
    return {'learner': classification.Logistic()}

def wekaLocalRuleZeroR(input_dict):
    """rulesZeroR Rule learner"""
    return {'learner': classification.rulesZeroR()  }

def wekaLocalRuleJRipper(input_dict):
    """Rule learner JRipper"""
    return {'learner': classification.rulesJRip()}

def wekaLocalKnnIBk(input_dict):
    """K-Nearest-Neighbours learner IBk"""
    return {'learner': classification.IBk()}

def wekaLocalRandomTree(input_dict):
    """Random Tree learner"""
    return {'learner': classification.RandomTree()}

def wekaLocalREPTree(input_dict):
    return {'learner': classification.REPTree()}


#
# PREPROCESSING
#

def wekaLocalFeatureSelection(input_dict):
    """Correlation-based Feature Subset Selection"""
    instances = input_dict['instances']
    output_dict = {}
    output_dict['selected'] = cf_weka_local.preprocessing.correlationBasedfeatSel(instances)
    return output_dict

def wekaLocalNormalize(input_dict):
    """Normalizes all numeric values in the given dataset"""
    instances = input_dict['instances']
    output_dict = {}
    # 1,0 -> normalizira na [0,1]; 2,-1 pa na [-1,1]
    output_dict['normalized'] = cf_weka_local.preprocessing.normalize(instances, '-S 2.0 -T -1.0')
    return output_dict

#
# EVALUATION
#

def wekaLocalApplyMappedClassifierGetInstances(input_dict):
    """An advanced version of the Apply Classifier method"""
    sclassifier = input_dict['classifier']
    soriginalInstances = input_dict['original_training_instances']
    sinstances = input_dict['instances']

    instances, report = evaluation.apply_mapped_classifier_get_instances(sclassifier, soriginalInstances, sinstances)

    output_dict = {'instances': instances, 'mapping_report': report}
    return output_dict


def wekaLocalCrossValidate(input_dict):
    """K-Fold Cross Validation"""
    nfolds = int( input_dict["folds"] )

    slearner = input_dict['learner']
    bunch = input_dict['instances']

    accuracy, conf_matrix, acc_by_class, summary = evaluation.cross_validate(slearner, bunch, nfolds=10)
    return {'accuracy':accuracy,
            'confusion_matrix':conf_matrix,
            'accuracy_by_class':acc_by_class,
            'summary':summary}

#
# UTILITIES
#
def wekaLocalExportDatasetToARFF(input_dict):
    """Export Dataset to an ARFF Textual Format"""
    return {}

def wekaLocalImportDatasetFromARFF(input_dict):
    """Imports Dataset From an ARFF Textual Format"""
    arff = input_dict['arff']
    output_dict = {}
    output_dict['instances'] = utilities.importDatasetFromArff(arff)
    return output_dict

def wekaLocalGetAttributeList(input_dict):
    arff_file = input_dict['arff_file']
    output_dict = {}
    output_dict['attr_list'] = utilities.getAttributeList(arff_file)
    return output_dict

def wekaLocalLoadUCI(input_dict):
    """Loads a UCI dataset"""
    arff_file = input_dict['filename']
    output_dict = {}
    output_dict['data'] = utilities.loadUciDatasetWeka(arff_file)
    return output_dict


#
# CLUSTERING
#

# def weka_local_Build_Clusterer(input_dict):
#     """Builds a clusterer using a learner and data instances"""
#     slearner = input_dict['learner']
#     sinstances = input_dict['instances']
#
#     clusterer = evaluation.build_clusterer(slearner, sinstances)
#
#     output_dict = {'clusterer': clusterer}
#     return output_dict
#
# def weka_local_Apply_Clusterer(input_dict):
#     return {}
#
# def weka_simple_kmeans(input_dict):
#     num_clusters = input_dict['k']
#     return {'learner': classification.SimpleKMeans()}
