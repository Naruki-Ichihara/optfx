#! /usr/bin/python3
# -*- coding: utf-8 -*-
'''Auto differentation for fenicsx'''

from dolfinx import fem
import ufl

def compute_sensitivities(form: ufl.Form, control: fem.Function) -> fem.Function:
    """Compute sensitivities using Auto grad

    Args:
        form (ufl.Form): Functional for derivative
        control (fem.Function): w.r.t. variable

    Returns:
        fem.Function: derivative
    """    
    func = fem.petsc.assemble_vector(fem.form(ufl.derivative(form, control)))
    u = fem.Function(control.function_space)
    u.vector.setArray(func)
    u.vector.assemble()
    return u
