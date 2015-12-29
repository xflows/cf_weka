__author__ = 'darkoa'

import unittest
import os
from os.path import join, normpath, dirname
import cf_weka_local.classification as c
import cf_weka_local.utilities as ut

class CF_Weka_Local_Tests(unittest.TestCase):

    def testClassificationLearners(self):
        """ Tests creating classification learners from the classification.py file
        :return: a list of learners, i.e. WekaClassifier objects
        """

        lrn = []
        try:
            lrn.append( c.Logistic()   )
            lrn.append( c.J48() )
            lrn.append( c.REPTree() )
            lrn.append( c.RandomForest() )
            lrn.append( c.RandomTree() )
            lrn.append( c.rulesZeroR() )
            lrn.append( c.rulesJRip() )
            lrn.append( c.IBk() )
            lrn.append( c.KStar() )
            lrn.append( c.naiveBayes() )
            lrn.append( c.Multilayer_Perceptron() )
            lrn.append( c.SMO() )

        except Exception, e:
            print "Exception: " + str(e)

        self.assertIs(len(lrn), 12)
        return lrn


# --------------------------------------------------------------------------------------------------------------------------

    def testClassificationModels(self):
        """ Tests building classification models using provided learners
        :return: True if all tests pass
        """

        lrn_arr =  self.testClassificationLearners()
        for lrn in lrn_arr:
            try:

                baseDir = normpath(dirname(__file__)) + '/../'
                wekaDir = normpath(join(baseDir, 'weka'))
                dataDir = normpath(join(wekaDir, 'data'))

                # f = data_dir + os.sep + 'cpu.with.vendor.arff'  # numeric target, but both numeric and nominal features
                f = dataDir + os.sep + 'breast-cancer.arff'    # nominal target
                fi = open(f,'r')
                classification_dataset = fi.read()
                fi.close()

                data = ut.importDatasetFromArff(classification_dataset)

                lrn.buildClassifier(data)

                dataNew = lrn.applyClassifier(data)

                self.assertIsNotNone(dataNew['targetPredicted'])
            except Exception, e:
                print "Exception: " + str(e)
