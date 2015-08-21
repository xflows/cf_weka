#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cf_weka_local.preprocessing

__author__ = 'vid, darko'

import classification
import evaluation
import utilities

#
# CLASSIFICATION ALGORITHMS
#

# Optimalne vrednosti parametrov: -C 0.5 -M 2 za vlake
def weka_local_J48(input_dict):
    """Decision Tree learner J48"""
    return {'learner': classification.J48_learner('-C 0.25 -M 2')} #-C 0.25 -M 2

def weka_local_NB(input_dict): # =None sicer ne dela iz py konzole
    """Naive Bayes learner"""
    return {'learner': classification.Naive_Bayes()} # learner v widgetu!

def weka_local_JRipper(input_dict):
    """Rule learner JRipper"""
    return {'learner': classification.JRip()}

# def weka_local_LibSVM(input_dict):
#     return {'learner': classification.LibSVM()} # not working

def weka_local_RandomForest(input_dict):
    """Random Forest learner"""
    return {'learner': classification.RandomForest()}

def weka_local_Multilayer_Perceptron(input_dict):
    """MLP Neural-network learner"""
    return {'learner': classification.Multilayer_Perceptron()}

def weka_local_SMO(input_dict):
    """SVM learner"""
    return {'learner': classification.SMO()}


def weka_local_Logistic(input_dict):
    """Logistic Regression learner"""
    return {'learner': classification.Logistic()}

def weka_local_ZeroR(input_dict):
    """ZeroR Rule learner"""
    return {'learner': classification.ZeroR()  }

def weka_local_knn_IBk(input_dict):
    """K-Nearest-Neighbours learner IBk"""
    return {'learner': classification.knn_IBk()}

def weka_local_Random_Tree(input_dict):
    """Random Tree learner"""
    return {'learner': classification.RandomTree()}

def weka_local_REP_Tree(input_dict):
    return {'learner': classification.REPTree()}

#
# PREPROCESSING
#

def weka_local_FeatureSelection(input_dict):
    """Correlation-based Feature Subset Selection"""
    instances = input_dict['instances']
    output_dict = {}
    output_dict['selected'] = cf_weka_local.preprocessing.FeatSel(instances)
    return output_dict

def weka_local_Normalize(input_dict):
    """Normalizes all numeric values in the given dataset"""
    instances = input_dict['instances']
    output_dict = {}
    # 1,0 -> normalizira na [0,1]; 2,-1 pa na [-1,1]
    output_dict['normalized'] = cf_weka_local.preprocessing.normalize(instances, '-S 2.0 -T -1.0')
    return output_dict

#
# EVALUATION
#

def weka_local_Build_Classifier(input_dict):
    """Builds a classifier using a learner and data instances"""
    slearner = input_dict['learner']
    sinstances = input_dict['instances']

    classifier = evaluation.build_classifier(slearner, sinstances)

    output_dict = {'classifier': classifier}
    return output_dict
# end

def weka_local_ApplyMappedClassifierGetInstances(input_dict):
    """An advanced version of the Apply Classifier method"""
    sclassifier = input_dict['classifier']
    soriginalInstances = input_dict['original_training_instances']
    sinstances = input_dict['instances']

    instances, report = evaluation.apply_mapped_classifier_get_instances(sclassifier, soriginalInstances, sinstances)

    output_dict = {'instances': instances, 'mapping_report': report}
    return output_dict
# end

def weka_local_Apply_Classifier(input_dict):
    """The Apply Classifier method: calculates predictions for given test instances"""
    sclassifier = input_dict['classifier']
    sinstances = input_dict['instances']

    instances_out = evaluation.apply_classifier(sclassifier, sinstances)

    output_dict = {'instances_out': instances_out}
    return output_dict

def weka_local_Cross_Validate(input_dict):
    """K-Fold Cross Validation"""
    nfolds = int( input_dict["folds"] )

    slearner = input_dict['learner']
    sinstances = input_dict['instances']

    accuracy, conf_matrix, acc_by_class, summary = evaluation.cross_validate(slearner, sinstances, nfolds=10)
    return {'accuracy':accuracy,
            'confusion_matrix':conf_matrix,
            'accuracy_by_class':acc_by_class,
            'summary':summary}

#
# UTILITIES
#
def weka_local_WekaInstancesToArff(input_dict):
    """Export Dataset to an ARFF Textual Format"""
    instances = input_dict['instances']
    output_dict = {}
    output_dict['arff'] = utilities.weka_instances_to_arff(instances)
    return output_dict
# end

def weka_local_ArffToWekaInstances(input_dict):
    """Imports Dataset From an ARFF Textual Format"""
    arff = input_dict['arff']
    output_dict = {}
    output_dict['instances'] = utilities.arff_to_weka_instances(arff)
    return output_dict

def weka_local_PrintModel(input_dict):
    """Outputs textual information about a Weka model"""
    model = input_dict['model']
    output_dict = {}
    output_dict['model_as_string'] = utilities.print_model(model)
    return output_dict

def weka_local_GetAttributeList(input_dict):
    arff_file = input_dict['arff_file']
    output_dict = {}
    output_dict['attr_list'] = utilities.get_attr_list(arff_file)
    return output_dict

def weka_local_LoadUCI(input_dict):
    """Loads a UCI dataset"""
    arff_file = input_dict['filename']
    output_dict = {}
    output_dict['data'] = utilities.load_uci(arff_file)
    return output_dict
