#!/usr/bin/env python
# -*- coding: utf-8 -*- 

__author__ = 'darkoa'

import re
import classification # dodal
import evaluation # dodal
import utilities # dodal


# def test1(input_dict):
#     """aaa """
#     output_dict={}
#     output_dict['bayesout'] = 0
#     return output_dict


def weka_local_create_integers(input_dict):
    intStr = input_dict['intStr']
    intList = []
    for i in re.findall(r'\w+', intStr):
        try:
            intList.append(int(i))
        except:
            pass
    if input_dict['sort'].lower() == "true":
        intList.sort()
    return {'intList':intList}

def weka_local_sum_integers(input_dict):
    intList = input_dict['intList']
    return {'sum':sum(intList)}

def weka_local_pre_filter_integers(input_dict):
    return input_dict

def weka_local_post_filter_integers(postdata,input_dict,output_dict):
    intListOut = postdata['intListOut']
    intList = []
    for i in intListOut:
        try:
            intList.append(int(i))
        except:
            pass
    return {'intList': intList}

def weka_local_pre_display_summation(input_dict):
    return {}

#
# CLASSIFICATION ALGORITHMS
#

# Optimalne vrednosti parametrov: -C 0.5 -M 2 za vlake
def weka_local_J48(input_dict):
    return {'learner': classification.J48_learner('-C 0.25 -M 2')} #-C 0.25 -M 2

def weka_local_NB(input_dict): # =None sicer ne dela iz py konzole
    return {'learner': classification.Naive_Bayes()} # learner v widgetu!

def weka_local_JRipper(input_dict):
    return {'learner': classification.JRip()}

def weka_local_LibSVM(input_dict):
    return {'learner': classification.LibSVM()}

def weka_local_RandomForest(input_dict):
    return {'learner': classification.RandomForest()}

def weka_local_Multilayer_Perceptron(input_dict):
    return {'learner': classification.Multilayer_Perceptron()}

def weka_local_SMO(input_dict):
    return {'learner': classification.SMO()}


def weka_local_Logistic(input_dict):
    return {'learner': classification.Logistic()}

def weka_local_ZeroR(input_dict):
    return {'learner': classification.Logistic()}

def weka_local_knn_IBk(input_dict):
    return {'learner': classification.knn_IBk()}

def weka_local_Random_Tree(input_dict):
    return {'learner': classification.RandomTree()}

def weka_local_REP_Tree(input_dict):
    return {'learner': classification.REPTree()}


def weka_local_AttributeSelection(input_dict):
    instances = input_dict['instances']
    output_dict = {}
    output_dict['selected'] = classification.AttSel(instances)
    return output_dict

def weka_local_Normalize(input_dict):
    instances = input_dict['instances']
    output_dict = {}
    # 1,0 -> normalizira na [0,1]; 2,-1 pa na [-1,1]
    output_dict['normalized'] = classification.normalize(instances, '-S 2.0 -T -1.0')
    return output_dict

#
# EVALUATION
#

def weka_local_Build_Classifier(input_dict):
    learner = input_dict['learner']
    instances = input_dict['instances']

    classifier = evaluation.build_classifier(learner, instances)

    output_dict = {'classifier': classifier}
    return output_dict
# end

def weka_local_ApplyMappedClassifierGetInstances(input_dict):
    sclassifier = input_dict['classifier']
    soriginalInstances = input_dict['original_training_instances']
    sinstances = input_dict['instances']

    instances, report = evaluation.apply_mapped_classifier_get_instances(sclassifier, soriginalInstances, sinstances)

    output_dict = {'instances': instances, 'mapping_report': report}
    return output_dict
# end

def weka_local_Apply_Classifier(input_dict):
    sclassifier = input_dict['classifier']
    sinstances = input_dict['instances']

    instances_out = evaluation.apply_classifier(sclassifier, sinstances)

    output_dict = {'instances_out': instances_out}
    return output_dict



#
# UTILITIES
#
def weka_local_WekaInstancesToArff(input_dict):
    instances = input_dict['instances']
    output_dict = {}
    output_dict['arff'] = utilities.weka_instances_to_arff(instances)
    return output_dict
# end

def weka_local_ArffToWekaInstances(input_dict):
    arff = input_dict['arff']
    output_dict = {}
    output_dict['instances'] = utilities.arff_to_weka_instances(arff)
    return output_dict

# zakaj no model build?
def weka_local_PrintModel(input_dict):
    model = input_dict['model']
    output_dict = {}
    output_dict['model_as_string'] = utilities.print_model(model)
    return output_dict

def weka_local_GetAttributeList(input_dict):
    arff_file = input_dict['arff_file']
    output_dict = {}
    output_dict['attr_list'] = utilities.get_attr_list(arff_file)
    return output_dict


