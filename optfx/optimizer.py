#! /usr/bin/python3
# -*- coding: utf-8 -*-
''' Optimizer interfaces for nlopt.
'''
from attr import attr
from dolfinx import fem
import ufl
import numpy as np
from .utils import to_numpy, from_numpy
from .abstract import Module
try:
    import nlopt as nl
except ImportError:
    raise ImportError('Nlopt must be installed.')

def opt(problem: Module, initials: list, wrt: list, setting: dict, params: dict, algorithm='LD_MMA') -> tuple:
    '''
    Optimizer based on nlopt.

    Args:
        problem (Module): Defined physics.
        initials (list): List of fem.Functions that interpolated the initial values. 
        wrt (list): Index.
        setting (dict): Setting for the optimizer. (see nlopt reference)
        params (dict): Parameters for the optimizer. (see nlopt reference)
        algorithm (str, optional): nlopt algorithm. Defaults to 'LD_MMA'.

    Returns:
        tuple: tuple of solutions as fem.Function
    '''    
    problem_size = 0
    for initial in initials:
        problem_size += initial.vector.size
    optimizer = nl.opt(getattr(nl, algorithm), problem_size)

    split_index = []
    index = 0
    for initial in initials:
        index += initial.vector.size
        split_index.append(index)

    def eval(x, grad):
        xs = np.split(x, split_index)
        xs_fenics = [from_numpy(i, j) for i, j in zip(xs, initials)]
        cost = problem.forward(xs_fenics)
        grad[:] = np.concatenate(problem.backward_objective(wrt))
        return np.float64(cost)

    constraints = []
    for attribute in dir(problem):
        if attribute.startswith('constraint'):
            constraints.append(attribute)

    if constraints:
        for constraint in constraints:
            def const(x, grad):
                measure_form = getattr(problem, constraint)()
                measure = fem.assemble_scalar(fem.form(measure_form))
                grad[:] = np.concatenate(problem.backward_constraint(constraint, wrt))
                return np.float64(measure)
            optimizer.add_inequality_constraint(const, 1e-8)

    optimizer.set_min_objective(eval)
    for set in setting:
        getattr(optimizer, set)(setting[set])
    for param in params:
        optimizer.set_param(param, params[param])
    initial_numpy = np.concatenate([to_numpy(i) for i in initials])
    solution_numpy = optimizer.optimize(initial_numpy)
    solution_fenics = [from_numpy(i, j) for i, j in zip(np.split(solution_numpy, split_index), initials)]
    return tuple(solution_fenics)
