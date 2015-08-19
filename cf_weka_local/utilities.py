__author__ = 'vid, darko'

import jpype as jp
import common, os
from os.path import join, normpath, dirname


def arff_to_weka_instances(arff, classIndex=None):
    '''Imports Dataset From an ARFF Textual Format

    :param arff: the data in ARFF textual format
    :param classIndex: the index of the class attribute
    :return: serialized Weka Instances object
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    tmp = common.TemporaryFile(suffix='.arff')
    tmp.writeString(arff)

    source = jp.JClass('weka.core.converters.ConverterUtils$DataSource')(tmp.name)
    instances = source.getDataSet()

    if classIndex is None:
        print 'Warning: class is set to the last attribute!'
        classIndex = instances.numAttributes() - 1
    elif classIndex == -1:
        classIndex = instances.numAttributes() - 1

    instances.setClassIndex(classIndex)
    return common.serializeWekaObject(instances)
# end

def weka_instances_to_arff(sdata):
    '''Export Dataset to an ARFF Textual Format

    :param sdata: serialized Weka Instances object
    :return:
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    instances = common.deserializeWekaObject(sdata)
    return instances.toString()
# end

def print_model(smodel):
    '''Outputs textual information about a Weka model

    :param smodel: a serialized Weka model
    :return: a textual representation of the Weka model
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    return common.deserializeWekaObject(smodel).toString()
# end

def get_attr_list(smodel):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    return common.deserializeWekaObject(smodel).toString()
# end

def load_uci(f = "cpu.arff"):
    '''Loads a UCI dataset from the ones provided along with this package

    :param f: the filename of the UCI file
    :return: serialized Weka Instances object
    '''
    base_dir = normpath(dirname(__file__))
    weka_dir = normpath(join(base_dir, 'weka'))
    data_dir = normpath(join(weka_dir, 'data'))

    import cf_base.library as l

    # f = "cpu.arff"
    f = data_dir + os.sep + f

    res_dict = l.load_to_string({'file':f})

    tmp = common.TemporaryFile(suffix='.arff')
    tmp.writeString( res_dict['string'] )

    source = jp.JClass('weka.core.converters.ConverterUtils$DataSource')(tmp.name)
    instances = source.getDataSet()

    classIndex=None

    if classIndex is None:
        print 'Warning: class is set to the last attribute!'
        classIndex = instances.numAttributes() - 1
    elif classIndex == -1:
        classIndex = instances.numAttributes() - 1

    instances.setClassIndex(classIndex)
    return common.serializeWekaObject(instances)

