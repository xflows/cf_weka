#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'daleksovski'

from django.shortcuts import render
import jpype as jp

def wekaLocalExportDatasetToARFF(request,input_dict,output_dict,widget):
    import utilities as ut
    from cf_core import helpers

    if not jp.isThreadAttachedToJVM():
        jp.attachThreadToJVM()

    bunch = input_dict['instances']
    output_dict = {}

    instances = ut.convert_bunch_to_weka_instances(bunch)

    destination = helpers.get_media_root()+'/'+str(request.user.id)+'/'+str(widget.id)+'.arff'

    f = open(destination, 'w')
    s = instances.toString()
    f.write(s)
    f.close()

    filename = str(request.user.id)+'/'+str(widget.id)+'.arff'
    output_dict['filename'] = filename

    return render(request, 'visualizations/string_to_file.html',{'widget':widget,'input_dict':input_dict,'output_dict':output_dict})
