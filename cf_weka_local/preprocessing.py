import jpype as jp
from cf_weka_local import common, utilities

__author__ = 'darkoa'


def correlationBasedfeatSel(bunch):
    """Correlation-based Feature Subset Selection, as implemented by the CfsSubsetEval class of Weka

    :param bunch: dataset
    :return: new dataset
    """


    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    # Instances data!
    data = utilities.convertBunchToWekaInstances(bunch)

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

    newInstances = Filter.useFilter(data, attsel_filter)

    return utilities.convertWekaInstancesToBunch(newInstances)


def normalize(bunch, params=None):
    '''Normalizes all numeric values in the given dataset (apart from the class attribute, if set)

    :param bunch: dataset
    :param params: parameters in textual form to pass to the Normalize Weka class
    :return: dataset
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    # Instances data!
    data = utilities.convertBunchToWekaInstances(bunch)

    Filter = jp.JClass('weka.filters.Filter')

    Normalize = jp.JClass('weka.filters.unsupervised.attribute.Normalize')
    normalize_filter = Normalize()
    normalize_filter.setOptions(common.parseOptions(params))
    normalize_filter.setInputFormat(data)

    newInstances = Filter.useFilter(data, normalize_filter)

    return utilities.convertWekaInstancesToBunch(newInstances)
