from dolfinx import fem
from dolfinx import mesh as msh
import ufl
import optfx as fx
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

m = 0.3    # Target rate of the material amount
p = 5      # Penalty parameter
eps = 1e-3 # Material lower bound
R = 0.025    # Helmholtz filter radius
n = 256    # Resolution

def k(a):
    return eps + (1 - eps) * a ** p

mesh = msh.create_unit_square(comm, 8, 8, msh.CellType.quadrilateral)
V = fem.FunctionSpace(mesh, ('Lagrange', 1))

uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

import numpy
# Create facet to cell connectivity required to determine boundary facets
tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)
boundary_facets = msh.exterior_facet_indices(mesh.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Function(V)
f.interpolate(lambda x: x[0]*0+1)
source = fem.Constant(mesh, -6.0)

class Poisson(fx.FX):
    def problem(self, controls):
        rho = controls[0]
        a = ufl.dot(ufl.grad(u), ufl.grad(v))*rho**3 * ufl.dx
        L = ufl.dot(source, v) * ufl.dx
        problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        uh = problem.solve()
        E = ufl.dot(ufl.grad(uh), ufl.grad(uh))*rho**3 * ufl.dx
        self.volume = rho*ufl.dx
        return E
    def constraint_volume(self):
        return self.volume
    
poisson = Poisson()
print(poisson.forward([f]))
print(poisson.backward_objective(wrt=[0]))
print(poisson.backward_constraint('constraint_volume', wrt=[0]))
