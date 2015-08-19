__author__ = 'vid'

import jpype as jp
import common


def J48_learner(params=None):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.trees.J48')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end


def Naive_Bayes(params=None):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.bayes.NaiveBayes')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end


def LibSVM(params=None):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()
    svm = jp.JClass('weka.classifiers.functions.LibSVM')()
    svm.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(svm)
# end


def JRip(params=None):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.rules.JRip')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end




def KStar(params=None):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.lazy.KStar')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end


def REPTree(params=None):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.trees.REPTree')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end


def RandomTree(params=None):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.trees.RandomTree')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end


def RandomForest(params=None):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.trees.RandomForest')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end


def Multilayer_Perceptron(params=None):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.functions.MultilayerPerceptron')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end


def SMO(params=None):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.functions.SMO')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end

def Logistic(params=None):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()
    
    model = jp.JClass('weka.classifiers.functions.Logistic')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end

def knn_IBk(params=None):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    model = jp.JClass('weka.classifiers.lazy.IBk')()
    model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)

def ZeroR(params=None):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()
    
    model = jp.JClass('weka.classifiers.rules.ZeroR')()
    #model.setOptions(common.parseOptions(params))
    return common.serializeWekaObject(model)
# end


def AttSel(instances, params=None):
    """Correlation-based Feature Subset Selection

    :param instances:
    :param params:
    :return:
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

    #print "Izbrani atributi: "
    #print Filter.useFilter(data, attsel_filter)

    return  common.serializeWekaObject(Filter.useFilter(data, attsel_filter))
# end

def normalize(instances, params=None):
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
# end

#    Normalize norm = new Normalize();
#    norm.setInputFormat(training_data_filter1); ???
#    Instances processed_training_data = Filter.useFilter(training_data_filter1, norm);











