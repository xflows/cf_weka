__author__ = 'darkoa'

import unittest, os
# import cf_base.library as b
import cf_weka_local.classification as c
import cf_weka_local.utilities as ut
import cf_weka_local.evaluation as ev
from os.path import join, normpath, dirname

class CF_Weka_Local_Tests(unittest.TestCase):

    def testClassificationLearners(self):
        """ Tests creating classification learners from the classification.py file
        :return: a list of learners
        """

        lrn = []
        try:
            lrn.append( c.Logistic()   )
            lrn.append( c.J48_learner() )
            lrn.append( c.REPTree() )
            lrn.append( c.RandomForest() )
            lrn.append( c.RandomTree() )
            lrn.append( c.ZeroR() )
            lrn.append( c.JRip() )
            lrn.append( c.knn_IBk() )
            lrn.append( c.KStar() )
            lrn.append( c.Naive_Bayes() )
            lrn.append( c.Multilayer_Perceptron() )
            lrn.append( c.SMO() )

        except Exception, e:
            print "Exception: " + e

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
                clf = None

                base_dir = normpath(dirname(__file__)) + '/../'
                weka_dir = normpath(join(base_dir, 'weka'))
                data_dir = normpath(join(weka_dir, 'data'))

                f = data_dir + os.sep + 'breast-cancer.arff'
                fi = open(f,'r')
                classification_dataset = fi.read()
                fi.close()

                classification_dataset = ut.arff_to_weka_instances(classification_dataset)


                clf = ev.build_classifier(lrn, classification_dataset)

            except Exception, e:
                clf = None
                print "Exception: " + e

            self.assertIsNotNone(clf)




    # def test_uci(self):
    #     """ Tests
    #     :return: True if all tests pass
    #     """
    #
    #     clf = None
    #
    #     # import cf_weka_local.utilities as u
    #
    #     res = 1 #u.load_uci("cpu.arff")
    #
    #     self.assertIsNotNone(res)
