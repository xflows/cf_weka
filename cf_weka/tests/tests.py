#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'daleksovski'

import os
import unittest
from os.path import join, normpath, dirname

import cf_weka.classification as c
import cf_weka.utilities as ut


class CFWekaTests(unittest.TestCase):
    def test_classification_learners(self):
        """ Tests creating classification learners from the classification.py file
        :return: a list of learners, i.e. WekaClassifier objects
        """

        lrn = []
        num_exceptions = 0
        try:
            lrn.append(c.logistic())
            lrn.append(c.j48())
            lrn.append(c.rep_tree())
            lrn.append(c.random_forest())
            lrn.append(c.random_tree())
            lrn.append(c.rules_zeror())
            lrn.append(c.rules_jrip())
            lrn.append(c.ibk())
            lrn.append(c.k_star())
            lrn.append(c.naive_bayes())
            lrn.append(c.multilayer_perceptron())
            lrn.append(c.smo())

        except Exception, e:
            num_exceptions += 1
            print "Exception: " + str(e)

        self.assertIs(num_exceptions, 0)
        return lrn

    # --------------------------------------------------------------------------------------------------------------------------

    def test_classification_models(self):
        """ Tests building classification models using provided learners
        :return: True if all tests pass
        """

        lrn_arr = self.test_classification_learners()
        for lrn in lrn_arr:
            try:

                base_dir = normpath(dirname(__file__)) + '/../'
                weka_dir = normpath(join(base_dir, 'weka'))
                data_dir = normpath(join(weka_dir, 'data'))

                # f = data_dir + os.sep + 'cpu.with.vendor.arff'  # numeric target, with both numeric and nominal features
                f = data_dir + os.sep + 'breast-cancer.arff'  # nominal target
                fi = open(f, 'r')
                classification_dataset = fi.read()
                fi.close()

                data = ut.import_dataset_from_arff(classification_dataset)

                lrn.buildClassifier(data)

                data_new = lrn.applyClassifier(data)

                self.assertIsNotNone(data_new['targetPredicted'])
            except Exception, e:
                print "Exception: " + str(e)
