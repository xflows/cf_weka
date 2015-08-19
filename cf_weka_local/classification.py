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
# end


def LibSVM(params=None):
    '''Support Vector Machine learner provided by Weka

    :param params: parameters in textual form to pass to the LibSVM Weka class
    :return: serialized Weka LibSVM object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()
    svm = jp.JClass('weka.classifiers.functions.LibSVM')()
    svm.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(svm)
# end


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
# end


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
# end


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
# end


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
# end

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
# end


def FeatSel(instances):
    """Correlation-based Feature Subset Selection, as implemented by the CfsSubsetEval class of Weka

    :param instances: serialized Weka Instances object
    :return: serialized Weka Instances object
    """


    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    # Instances data!
    data = common.deserializeWekaObject(instances)

    Filter = jp.JClass('weka.filters.Filter')
    #attsel_Filter = Filter()

    AttributeSelection = jp.JClass('weka.filters.supervised.attribute.AttributeSelection')
    attsel_filter = AttributeSelection()

    CfsSubsetEval = jp.JClass('weka.attributeSelection.CfsSubsetEval')
    attsel_eval = CfsSubsetEval()

    GreedyStepwise = jp.JClass('weka.attributeSelection.BestFirst')
    attsel_search = GreedyStepwise()

    
    #attsel_search.setSearchBackwards(True) # True, true
    attsel_filter.setEvaluator(attsel_eval)
    attsel_filter.setSearch(attsel_search)
    attsel_filter.setInputFormat(data) 

    return  common.serializeWekaObject(Filter.useFilter(data, attsel_filter))
# end

def normalize(instances, params=None):
    '''Normalizes all numeric values in the given dataset (apart from the class attribute, if set)

    :param instances: serialized Weka Instances object
    :param params: parameters in textual form to pass to the Normalize Weka class
    :return: serialized Weka Instances object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    # Instances data!
    data = common.deserializeWekaObject(instances)

    Filter = jp.JClass('weka.filters.Filter')
    #attsel_Filter = Filter()

    Normalize = jp.JClass('weka.filters.unsupervised.attribute.Normalize')
    normalize_filter = Normalize()
    normalize_filter.setOptions(common.parseOptions(params))
    normalize_filter.setInputFormat(data)

    return  common.serializeWekaObject(Filter.useFilter(data, normalize_filter))









