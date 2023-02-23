#! /usr/bin/python3
# -*- coding: utf-8 -*-
'''Abstract modules for the definition of functions.'''

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from dolfinx import fem
import numpy as np
import ufl
from .utils import to_numpy
from .adjoint import compute_sensitivities

class Module(metaclass=ABCMeta):
    '''
    Core module of optfx. Users shoud inherit this module in your physics model.
    Your defined the physics class must have the *problem* method.

    Args:
        metaclass (_type_, optional): ABCmeta. Do not change.
    '''    
    def __compute_sensitivities(self, objective: ufl.Form, controls: list, wrt: Iterable | None) -> list:
        '''
        Internal method. Comute sensitivities using the auto diff.

        Args:
            objective (ufl.Form): Objective
            controls (list): Control varibals
            wrt (Iterable | None): Index for differentiation

        Raises:
            ValueError

        Returns:
            list: List of numpy array 
        '''    
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
    def problem(self, controls: list) -> None:
        '''
        User difines own physics model in this method. 

        Args:
            controls (list): Controls

        Raises:
            NotImplementedError
        '''        
        raise NotImplementedError('')

    def forward(self, controls: list) -> float:
        '''
        Forward calculation. You must return the ufl.form of objective (assumed the scalar value)

        Args:
            controls (list): Controls

        Returns:
            float: Objective values
        '''        
        self.controls_fenics = controls
        self.objective = self.problem(self.controls_fenics)
        value = fem.assemble_scalar(fem.form(self.objective))
        return value
    
    def backward_objective(self, wrt: list) -> np.ndarray:
        '''
        Backword calculation for the objective w.r.t. wrt index. 

        Args:
            wrt (list): Index.

        Returns:
            np.ndarray: sensitivities
        '''        
        sens = self.__compute_sensitivities(self.objective, self.controls_fenics, wrt)
        return sens
    
    def backward_constraint(self, target, wrt=None):
        '''
        Backword target constraint for the objective w.r.t. wrt index. 

        Args:
            target (str): Target.
            wrt (list): Index.

        Returns:
            np.ndarray: sensitivities
        '''      
        sens = self.__compute_sensitivities(getattr(self, target)(), self.controls_fenics, wrt)
        return sens