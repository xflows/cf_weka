__author__ = 'vid'

import jpype as jp
import common

MAPPING_REPORT_START = 'Attribute mappings:'


def build_classifier(slearner, sinstances):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    learner = common.deserializeWekaObject(slearner)
    instances = common.deserializeWekaObject(sinstances)

    if instances.classIndex() == -1:
        raise ValueError('Class not set!')

    learner.buildClassifier(instances)
    return common.serializeWekaObject(learner)
# end


def apply_classifier(sclassifier, sinstances):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    classifier = common.deserializeWekaObject(sclassifier)
    instances = common.deserializeWekaObject(sinstances)

    classes = []
    classIndex = instances.classIndex()
    if classIndex == -1:
        raise ValueError('Class not set!')
    classAttribute = instances.classAttribute()
    for instance in instances:
        label = int(classifier.classifyInstance(instance))
        classes.append(classAttribute.value(label))

    return classes
# end


def sapply_mapped_classifier(sclassifier, soriginalInstances, sinstances):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    classifier = common.deserializeWekaObject(sclassifier)
    original_training_instances = common.deserializeWekaObject(soriginalInstances)
    instances = common.deserializeWekaObject(sinstances)

    # serialize classifier with original instances to a file once again for the Mapped classifier
    tfile = common.TemporaryFile(flags='wb+')
    s = jp.JClass('weka.core.SerializationHelper')
    s.writeAll(tfile.name, [classifier, original_training_instances])

    # construct a MappedClassifier
    mappedClassifier =  jp.JClass('weka.classifiers.misc.InputMappedClassifier')()
    mappedClassifier.setIgnoreCaseForNames(True)
    mappedClassifier.setTrim(True)
    #mappedClassifier.setSuppressMappingReport(True)
    #mc.setModelHeader(original_training_instances)
    mappedClassifier.setModelPath(tfile.name)

    # use the mapped classifier on new data
    classes = []
    classIndex = instances.classIndex()
    if classIndex == -1:
        raise ValueError('Class not set!')
    classAttribute = instances.classAttribute()
    for instance in instances:
        label = int(mappedClassifier.classifyInstance(instance))
        classes.append(classAttribute.value(label))

    report = mappedClassifier.toString()
    if MAPPING_REPORT_START in report:
        report = report[report.index(MAPPING_REPORT_START):]

    return classes, report
# end


def apply_mapped_classifier_get_instances(sclassifier, soriginalInstances, sinstances):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    classifier = common.deserializeWekaObject(sclassifier)
    original_training_instances = common.deserializeWekaObject(soriginalInstances)
    instances = common.deserializeWekaObject(sinstances)

    # serialize classifier with original instances to a file once again for the Mapped classifier
    tfile = common.TemporaryFile(flags='wb+')
    s = jp.JClass('weka.core.SerializationHelper')
    s.writeAll(tfile.name, [classifier, original_training_instances])

    # construct a MappedClassifier
    mappedClassifier = jp.JClass('weka.classifiers.misc.InputMappedClassifier')()
    mappedClassifier.setIgnoreCaseForNames(True)
    mappedClassifier.setTrim(True)
    #mappedClassifier.setSuppressMappingReport(True)
    #mc.setModelHeader(original_training_instances)
    mappedClassifier.setModelPath(tfile.name)

    # use the mapped classifier on new data
    classIndex = instances.classIndex()
    if classIndex == -1:
        raise ValueError('Class not set!')
    classAttribute = instances.classAttribute()
    for instance in instances:
        # Thrown to indicate that an array has been accessed with an illegal index. 
        # The index is either negative or greater than or equal to the size of the array.
        label = int(mappedClassifier.classifyInstance(instance)) # 20-fold trains Tezave zaradi premalo primerov v foldu?
        instance.setClassValue(classAttribute.value(label))

    report = mappedClassifier.toString()
    if MAPPING_REPORT_START in report:
        report = report[report.index(MAPPING_REPORT_START):]

    return common.serializeWekaObject(instances), report
# end


def apply_classifier_get_instances(sclassifier, sinstances):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    classifier = common.deserializeWekaObject(sclassifier)
    instances = common.deserializeWekaObject(sinstances)

    classes = []
    classIndex = instances.classIndex()
    if classIndex == -1:
        raise ValueError('Class not set!')
    classAttribute = instances.classAttribute()
    for instance in instances:
        label = int(classifier.classifyInstance(instance))
        instance.setClassValue(classAttribute.value(label))

    return common.serializeWekaObject(instances)
# end

def cross_validate(slearner, sinstances, nfolds=10):
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    learner = common.deserializeWekaObject(slearner)
    instances = common.deserializeWekaObject(sinstances)

    buff = jp.JClass('java.lang.StringBuffer')()
    out = jp.JClass('weka.classifiers.evaluation.output.prediction.PlainText')()
    out.setSuppressOutput(True)
    out.setBuffer(buff)

    rnd = jp.JClass('java.util.Random')()
    cv = jp.JClass('weka.classifiers.Evaluation')(instances)
    args = jp.JArray(jp.JClass('java.lang.Object'), 1)(1)
    args[0] = out

    cv.crossValidateModel(learner, instances, nfolds, rnd, args)
    # accuracy, conf. matrix, acc. by class, summary
    return cv.pctCorrect(), cv.toMatrixString(), cv.toSummaryString(True), cv.toClassDetailsString()
# end

