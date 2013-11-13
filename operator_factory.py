#!/usr/bin/python -tt
from __future__ import generators
from py_operators import *
import random

class OperatorFactory(object):
    """
    Base class for defining custom operators
    """
    factories = {}
    def add_factory(str_id, op_factory):
        OperatorFactory.factories[str_id] = op_factory
    add_factory = staticmethod(add_factory)
    def create_operator(ps_parameters,str_section):
        str_id = ps_parameters.get_section_dict(str_section)['name']
        if not OperatorFactory.factories.has_key(str_id):
            OperatorFactory.factories[str_id] = \
              eval(str_id + '.Factory()')
        return OperatorFactory.factories[str_id]. \
               create(ps_parameters,str_section)
    create_operator = staticmethod(create_operator)