__author__ = 'vid'

from base64 import b64encode, b64decode
from temputils import TemporaryFile
import jpype as jp
#from jpype import java as java
from os.path import join, normpath, dirname



BASE = normpath(dirname(__file__))
JARDIR = normpath(join(BASE, 'weka'))
CLASSPATH = '-Djava.ext.dirs=' + JARDIR # + '-Djava.class.path=' + JARDIR
if not jp.isJVMStarted():
    jp.startJVM(jp.getDefaultJVMPath(), CLASSPATH)

#a = jp.JClass('weka.core.ClassloaderUtil')
#a.addFile('blah.jar')


def parseOptions(opString):
    return opString.replace(',', ' ').split() if opString != None else []


def serializeWekaObject(obj):
    s = jp.JClass('weka.core.SerializationHelper')
    tfile = TemporaryFile(flags='wb+')
    s.write(tfile.name, obj)
    return b64encode(tfile.fp.read())
# end


def deserializeWekaObject(objString):
    d = jp.JClass('weka.core.SerializationHelper')
    tfile = TemporaryFile(flags='wb+')
    tfile.writeString(b64decode(objString))
    return d.read(tfile.name)
# end


def loadInstancesFromString(dataString):
    tmp = TemporaryFile(suffix='.arff')
    tmp.writeString(dataString)
    source = jp.JClass('weka.core.converters.ConverterUtils$DataSource')(tmp.name)
    instances = source.getDataSet()
    return instances
# end


