__author__ = 'vid, darko'

import jpype as jp

import common


def J48_learner(params=None):
    '''Weka decision tree learner J48

    :param params: parameters in textual form to pass to the J48 Weka class (e.g. "-C 0.25 -M 2")
    :return: serialized Weka J48 object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.trees.J48')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end


def Naive_Bayes(params=None):
    '''Naive Bayes classifier provided by Weka. Naive Bayes is a simple probabilistic classifier based on applying the Bayes' theorem.

    :param params: parameters in textual form to pass to the NaiveBayes Weka class
    :return: serialized Weka NaiveBayes object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.bayes.NaiveBayes')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)

def JRip(params=None):
    '''The RIPPER rule learner by Weka

    :param params: parameters in textual form to pass to the JRip Weka class
    :return: serialized Weka JRip object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.rules.JRip')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end




def KStar(params=None):
    '''Instance-Based learner K* by Weka

    :param params: parameters in textual form to pass to the KStar Weka class
    :return: serialized Weka KStar object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.lazy.KStar')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end


def REPTree(params=None):
    '''A REP Tree, which is a fast decision tree learner. Builds a decision/regression tree using information gain/variance and prunes it using reduced-error pruning

    :param params: parameters in textual form to pass to the REPTree Weka class
    :return: serialized Weka REPTree object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.trees.REPTree')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end


def RandomTree(params=None):
    '''A tree that considers K randomly chosen attributes at each node, and performs no pruning

    :param params: parameters in textual form to pass to the RandomTree Weka class
    :return: serialized Weka RandomTree object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.trees.RandomTree')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)



def RandomForest(params=None):
    '''Random Forest learner by Weka

    :param params: parameters in textual form to pass to the RandomForest Weka class
    :return: serialized Weka RandomForest object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.trees.RandomForest')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)



def Multilayer_Perceptron(params=None):
    '''Feedforward artificial neural network, using backpropagation to classify instances

    :param params: parameters in textual form to pass to the MultilayerPerceptron Weka class
    :return: serialized Weka MultilayerPerceptron object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.functions.MultilayerPerceptron')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)



def SMO(params=None):
    '''A support vector classifier, trained using a sequential minimal optimization (SMO) algorithm

    :param params: parameters in textual form to pass to the SMO Weka class
    :return: serialized Weka SMO object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.functions.SMO')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)


def Logistic(params=None):
    '''Logistic regression by Weka

    :param params: parameters in textual form to pass to the Logistic Weka class
    :return: serialized Weka Logistic object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()
    
    model = jp.JClass('weka.classifiers.functions.Logistic')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end

def knn_IBk(params=None):
    '''K-nearest neighbours classifier by Weka

    :param params: parameters in textual form to pass to the IBk Weka class
    :return: serialized Weka IBk object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.lazy.IBk')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)

def ZeroR(params=None):
    '''Weka's ZeroR classifier: predicts the mean (for a numeric class) or the mode (for a nominal class).

    :param params: parameters in textual form to pass to the ZeroR Weka class
    :return: serialized Weka ZeroR object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()
    
    model = jp.JClass('weka.classifiers.rules.ZeroR')()
    #model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)










