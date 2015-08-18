__author__ = 'vid'

import jpype as jp
import common


def arff_to_weka_instances(arff, classIndex=None):
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
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    instances = common.deserializeWekaObject(sdata)
    return instances.toString()
# end

def print_model(smodel):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    return common.deserializeWekaObject(smodel).toString()
# end

def get_attr_list(smodel):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    return common.deserializeWekaObject(smodel).toString()
# end
