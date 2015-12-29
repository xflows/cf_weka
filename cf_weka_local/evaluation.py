__author__ = 'vid, darko'

import jpype as jp
import common
import utilities as ut

MAPPING_REPORT_START = 'Attribute mappings:'

def build_clusterer(slearner, sinstances):
    '''The Build Classifier method

    :param slearner: serialized learner
    :param sinstances: serialized Instances object
    :return: serialized Classifier object
    '''

    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    learner = common.deserializeWekaObject(slearner)
    instances = common.deserializeWekaObject(sinstances)

    learner.buildClusterer(instances)
    return common.serializeWekaObject(learner)


def apply_mapped_classifier_get_instances(wekaClassifier, originalData, data):
    '''An advanced version of the Apply Classifier method.
    Addresses incompatible training and test data, and returns a dataset with predictions.

    :param wekaClassifier: WekaClassifier object
    :param originalData: original training instances, bunch
    :param data: test instances, bunch
    :return: ???Instances object with predictions and a textual report from the InputMappedClassifier class
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    try:
        classifier = common.deserializeWekaObject(wekaClassifier.sclassifier)
    except:
        raise Exception("Only WEKA classifiers/models supported. Please provide a valid WEKA learner.")

    original_training_instances = ut.convertBunchToWekaInstances(originalData)
    instances = ut.convertBunchToWekaInstances(data)

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

    predictions = []
    try:
        for instance in instances:
            label = int(mappedClassifier.classifyInstance(instance))
            predictions.append(label)

        data["targetPredicted"] = predictions
    except:
        raise Exception("Classifier not built. Please use the Build Classifier widget first.")

    report = mappedClassifier.toString()
    if MAPPING_REPORT_START in report:
        report = report[report.index(MAPPING_REPORT_START):]

    return data, report
# end


def cross_validate(wekaClassifier, bunch, nfolds=10):
    '''K-Fold Cross Validation

    :param wekaClassifier: a WekaClassifier object
    :param bunch:
    :param nfolds: the number of folds
    :return: numeric accuracy, conf.matrix (text), textual accuracy by class, textual summary
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    try:
        learner = common.deserializeWekaObject(wekaClassifier.sclassifier)
    except:
        raise Exception("Only WEKA classifiers/models supported. Please provide a valid WEKA learner.")

    # instances = common.deserializeWekaObject(sinstances)
    instances = ut.convertBunchToWekaInstances(bunch)

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

