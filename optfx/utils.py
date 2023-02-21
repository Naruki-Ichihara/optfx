#! /usr/bin/python3
# -*- coding: utf-8 -*-
'''Utiltes for optfx.'''

from dolfinx import fem
import numpy as np
from mpi4py import MPI

def to_numpy(fenics_var: fem.Constant | fem.Function) -> np.ndarray:
    """Converting from fenicsx variables to numpy ndarray.

    Args:
        fenics_var (fem.Constant | fem.Function): fenicsx variable

    Raises:
        ValueError

    Returns:
        np.ndarray: Return the numpy.ndarray
    """    
    if isinstance(fenics_var, fem.Constant):
        return np.asarray(fenics_var.value)
    if isinstance(fenics_var, fem.Function):
        fenics_vec = fenics_var.vector
        data = fenics_vec
        return fenics_vec.getArray()
    raise ValueError("Cannot convert " + str(type(fenics_var)))

def from_numpy(numpy_array: np.ndarray, func_temp: fem.Function) -> fem.Function:
    """Converting from numpy array to fenicsx variables based on the function space.

    Args:
        numpy_array (np.ndarray): Array to convert into fenicsx
        func_space (fem.FunctionSpace): Base-function space

    Raises:
        ValueError

    Returns:
        fem.Function: Return the fenicsx.fem.Function
    """    
    func_space = func_temp.function_space
    u = type(func_temp)(func_space)
    np_size = numpy_array.size
    fenics_size = u.vector.size
    if np_size != fenics_size:
        err_msg = (
            "Cannot convert numpy array to Function: Wrong size {} vs {}".format(np_size, fenics_size)
        )
        raise ValueError(err_msg)

    range_begin, range_end = u.vector.owner_range
    numpy_array = np.asarray(numpy_array)
    local_array = numpy_array.reshape(fenics_size)[range_begin:range_end]
    u.vector.setArray(local_array)
    u.vector.assemble()
    return u