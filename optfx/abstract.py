#! /usr/bin/python3
# -*- coding: utf-8 -*-
'''Abstract modules for the definition of functions.'''

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from dolfinx import fem
import ufl
from .utils import to_numpy
from .adjoint import compute_sensitivities

class FX(metaclass=ABCMeta):
    def __compute_sensitivities(self, objective: ufl.Form, controls: list, wrt: Iterable | None):
        if isinstance(wrt, Iterable):
            sens_numpy = []
            for i in range(len(controls)):
                if i not in wrt:
                    sens_numpy.append(to_numpy(controls[i])*0.0)
                else:
                    control = controls[i]
                    sens_numpy.append(to_numpy(compute_sensitivities(objective, control)))
                return sens_numpy
        else:
            raise ValueError("wrt must be iterable.")
        
    @abstractmethod
    def problem(self, controls):
        raise NotImplementedError('')

    def forward(self, controls) -> float:
        self.controls_fenics = controls
        self.objective = self.problem(self.controls_fenics)
        value = fem.assemble_scalar(fem.form(self.objective))
        return value
    
    def backward_objective(self, wrt=None):
        sens = self.__compute_sensitivities(self.objective, self.controls_fenics, wrt)
        return sens
    
    def backward_constraint(self, target, wrt=None):
        sens = self.__compute_sensitivities(getattr(self, target)(), self.controls_fenics, wrt)
        return sens