#!/usr/bin/env python
# -*- coding: utf-8 -*-
__authors__ = 'vid, daleksovski'

import jpype as jp
import common, os
from os.path import join, normpath, dirname
import sklearn.datasets.base as sk
import numpy as np


def import_dataset_from_arff(arff, class_index=None):
    '''Imports Dataset From an ARFF Textual Format

    :param arff: the data in ARFF textual format
    :param classIndex: the index of the class attribute
    :return: a dataset (Bunch)
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    tmp = common.TemporaryFile(suffix='.arff')
    tmp.writeString(arff)

    source = jp.JClass('weka.core.converters.ConverterUtils$DataSource')(tmp.name)
    instances = source.getDataSet()

    if class_index is None:
        print 'Warning: class is set to the last attribute!'
        class_index = instances.numAttributes() - 1
    elif class_index == -1:
        class_index = instances.numAttributes() - 1

    instances.setClassIndex(class_index)
    return convert_weka_instances_to_bunch(instances)


def export_dataset_to_arff(bunch):
    '''Exports a dataset to an ARFF file

    :param bunch: dataset
    :return:
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    instances = convert_bunch_to_weka_instances(bunch)

    file_contents = instances.toString()

    # numpy.savetxt("foo.csv", csv, fmt='%.6f', delimiter=",")

    fileOut = open('myfile', 'w')
    fileOut.write(file_contents)
    fileOut.close()

    output_dict = {}
    output_dict['FileOut'] = fileOut
    return output_dict


def print_weka_model(weka_classifier):
    '''Outputs textual information about a Weka model

    :param weka_classifier: a WekaClassifier object
    :return: a textual representation of the Weka model
    '''
    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    try:
        model = common.deserialize_weka_object(weka_classifier.sclassifier)
    except:
        raise Exception("Only WEKA classifiers/models supported. Please provide a valid WEKA learner.")

    return model.toString()


def load_uci_dataset_weka(file_name="cpu.arff"):
    '''Loads a UCI dataset from the ones provided along with this package

    :param f: the filename of the UCI file
    :return: dataset (Bunch)
    '''
    file_names = ["breast-cancer.arff", "contact-lenses.arff", "cpu.arff", "cpu.with.vendor.arff", "credit-g.arff",
                  "diabetes.arff", "glass.arff", "ionosphere.arff", "iris.2D.arff", "iris.arff", "labor.arff",
                  "ReutersCorn-test.arff", "ReutersCorn-train.arff", "ReutersGrain-test.arff",
                  "ReutersGrain-train.arff", "segment-challenge.arff", "segment-test.arff", "soybean.arff",
                  "supermarket.arff", "unbalanced.arff", "vote.arff", "weather.nominal.arff", "weather.numeric.arff"]
    if not (file_name in file_names):
        raise Exception("Illegal dataset requested.")

    base_dir = normpath(dirname(__file__))
    weka_dir = normpath(join(base_dir, 'weka'))
    data_dir = normpath(join(weka_dir, 'data'))

    f = data_dir + os.sep + file_name
    fi = open(f, 'r')
    classification_dataset = fi.read()
    fi.close()

    bunch = import_dataset_from_arff(classification_dataset)

    return bunch


def convert_weka_instances_to_bunch(instances):
    ''' Converts WEKA Instances to the scikit Bunch format

    :param instances: WEKA dataset (Instances)
    :return:
    '''

    if instances.classIndex() < 0:
        instances.setClassIndex(instances.numAttributes() - 1)

    target_att = instances.classAttribute()
    target_names = []
    if target_att.isNominal():
        for j in range(0, target_att.numValues()):
            target_names.append(target_att.value(j))

    feature_names = []

    num_samples = instances.numInstances()
    num_attributes = instances.numAttributes()
    num_targets = 1

    data = np.empty((num_samples, num_attributes - num_targets))
    target = np.empty((num_samples,), dtype=np.int)

    fdescr = instances.relationName()

    feature_value_names = []
    for j in range(0, num_attributes - num_targets):
        myatt = instances.attribute(j)
        # mtype = 1 if myatt.isNumeric() else 0
        mname = myatt.name()
        feature_names.append(mname)

        num_vals = myatt.numValues()
        f_vals = []
        for k in range(0, num_vals):
            f_vals.append(myatt.value(k))

        feature_value_names.append(f_vals)

    for i in range(0, num_samples):
        arr = []
        for j in range(0, num_attributes):
            arr.append(instances.get(i).value(j))

        data[i] = np.asarray(arr[:-1], dtype=np.float)
        if target_att.isNominal():
            target[i] = np.asarray(arr[-1], dtype=np.int)
        else:
            target[i] = np.asarray(arr[-1], dtype=np.float)

    return sk.Bunch(data=data, target=target,
                    target_names=target_names,
                    DESCR=fdescr,
                    feature_value_names=feature_value_names,
                    feature_names=feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])


def convert_bunch_to_weka_instances(bunch):
    '''Converts a dataset from a scikit-learn Bunch format to a Weka Instances object

    :param bunch: dataset in scikit Bunch format; looks for field 'feature_value_names' for nominal features
    :return: dataset in Weka Instances format
    '''

    util = jp.JPackage("java.util")
    weka_core = jp.JPackage("weka.core")

    list_attrs = util.ArrayList()

    num_samples = bunch.data.shape[0]  # num_samples = 5
    num_features = bunch.data.shape[1]  # numAtt = 3
    num_targets = 1

    # create the features
    for j in range(0, num_features):
        if bunch.has_key("feature_value_names") and len(bunch.feature_value_names[j]) > 0:
            att_vals = util.ArrayList()
            for v in bunch.feature_value_names[j]:
                att_vals.add(v)

            # Constructor for nominal attributes and string attributes.
            att = weka_core.Attribute(bunch.feature_names[j], att_vals)

        else:
            # Constructor for a numeric attribute with a particular index.
            att = weka_core.Attribute(bunch.feature_names[j], j)

        list_attrs.add(att)

    num_att = num_features + num_targets
    if len(bunch.target_names) > 0:
        # nominal target
        att_vals = util.ArrayList()
        for v in bunch.target_names:
            att_vals.add(v)
        # Constructor for nominal attributes and string attributes.
        target_att = weka_core.Attribute("nominalClass", att_vals)

    else:
        # numeric target
        target_att = weka_core.Attribute("numericClass", num_att - 1)

    list_attrs.add(target_att)

    inst = weka_core.Instances(bunch.DESCR, list_attrs, num_samples)

    for i in range(0, num_samples):
        my_row = weka_core.DenseInstance(num_att)
        for j in range(0, num_features):
            my_row.setValue(j, bunch.data[i][j])

        my_row.setValue(num_att - 1, jp.JDouble(bunch.target[i]))

        inst.add(jp.JObject(my_row, 'weka.core.Instance'))

    inst.setClassIndex(inst.numAttributes() - 1)

    return inst
