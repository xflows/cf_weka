#!/usr/bin/env python
# -*- coding: utf-8 -*-
__authors__ = 'vid, daleksovski'

import jpype as jp

import common
from weka_classifier import WekaClassifier


def j48(params=None):
    '''Weka decision tree learner J48

    :param params: parameters in textual form to pass to the J48 Weka class (e.g. "-C 0.25 -M 2")
    :return: a WekaClassifier object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.trees.J48')()
    model.setOptions(common.parse_options(params))

    return WekaClassifier(common.serialize_weka_object(model))


def naive_bayes(params=None):
    '''Naive Bayes classifier provided by Weka. Naive Bayes is a simple probabilistic classifier based on applying the Bayes' theorem.

    :param params: parameters in textual form to pass to the NaiveBayes Weka class
    :return: a WekaClassifier object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.bayes.NaiveBayes')()
    model.setOptions(common.parse_options(params))
    return WekaClassifier(common.serialize_weka_object(model))


def k_star(params=None):
    '''Instance-Based learner K* by Weka

    :param params: parameters in textual form to pass to the KStar Weka class
    :return: a WekaClassifier object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.lazy.KStar')()
    model.setOptions(common.parse_options(params))
    return WekaClassifier(common.serialize_weka_object(model))


def rep_tree(params=None):
    '''A REP Tree, which is a fast decision tree learner. Builds a decision/regression tree using information gain/variance and prunes it using reduced-error pruning

    :param params: parameters in textual form to pass to the REPTree Weka class
    :return: a WekaClassifier object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.trees.REPTree')()
    model.setOptions(common.parse_options(params))
    return WekaClassifier(common.serialize_weka_object(model))


def random_tree(params=None):
    '''A tree that considers K randomly chosen attributes at each node, and performs no pruning

    :param params: parameters in textual form to pass to the RandomTree Weka class
    :return: a WekaClassifier object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.trees.RandomTree')()
    model.setOptions(common.parse_options(params))
    return WekaClassifier(common.serialize_weka_object(model))


def random_forest(params=None):
    '''Random Forest learner by Weka

    :param params: parameters in textual form to pass to the RandomForest Weka class
    :return: a WekaClassifier object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.trees.RandomForest')()
    model.setOptions(common.parse_options(params))
    return WekaClassifier(common.serialize_weka_object(model))


def multilayer_perceptron(params=None):
    '''Feedforward artificial neural network, using backpropagation to classify instances

    :param params: parameters in textual form to pass to the MultilayerPerceptron Weka class
    :return: a WekaClassifier object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.functions.MultilayerPerceptron')()
    model.setOptions(common.parse_options(params))
    return WekaClassifier(common.serialize_weka_object(model))


def smo(params=None):
    '''A support vector classifier, trained using the Sequential Minimal Optimization (SMO) algorithm

    :param params: parameters in textual form to pass to the SMO Weka class
    :return: a WekaClassifier object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.functions.SMO')()
    model.setOptions(common.parse_options(params))
    return WekaClassifier(common.serialize_weka_object(model))


def logistic(params=None):
    '''Logistic regression by Weka

    :param params: parameters in textual form to pass to the Logistic Weka class
    :return: a WekaClassifier object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.functions.Logistic')()
    model.setOptions(common.parse_options(params))
    return WekaClassifier(common.serialize_weka_object(model))


def ibk(params=None):
    '''K-nearest neighbours classifier by Weka

    :param params: parameters in textual form to pass to the IBk Weka class
    :return: a WekaClassifier object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.lazy.IBk')()
    model.setOptions(common.parse_options(params))
    return WekaClassifier(common.serialize_weka_object(model))


# ++++++++++++ rules ++++++++++++

def rules_jrip(params=None):
    '''The RIPPER rule learner by Weka

    :param params: parameters in textual form to pass to the JRip Weka class
    :return: a WekaClassifier object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.rules.JRip')()
    model.setOptions(common.parse_options(params))
    return WekaClassifier(common.serialize_weka_object(model))


def rules_zeror(params=None):
    '''Weka's rulesZeroR classifier: predicts the mean (for a numeric class) or the mode (for a nominal class).

    :param params: parameters in textual form to pass to the rulesZeroR Weka class
    :return: a WekaClassifier object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.rules.ZeroR')()
    model.setOptions(common.parse_options(params))
    return WekaClassifier(common.serialize_weka_object(model))
