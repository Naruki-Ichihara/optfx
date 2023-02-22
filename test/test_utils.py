import optfx as fx
from dolfinx.fem import FunctionSpace, Function
from dolfinx import fem
from dolfinx import mesh
from mpi4py import MPI
import numpy as np
import ufl

def test_to_from_numpy():
    comm = MPI.COMM_WORLD
    test_domain = mesh.create_unit_square(comm, 100, 100, mesh.CellType.quadrilateral)
    V = FunctionSpace(test_domain, ("Lagrange", 1))
    u = Function(V)
    sample = np.random.rand(10201)*10
    x = fx.from_numpy(sample, u)
    true_values = x.vector.getArray()
    result = fx.to_numpy(x)
    assert np.allclose(true_values, result)

def test_adjoint(): # In-progress
    dx = ufl.Measure("dx", metadata={"quadrature_degree": 4})
    comm = MPI.COMM_WORLD
    test_domain = mesh.create_unit_square(comm, 100, 100, mesh.CellType.quadrilateral)
    sample = np.random.rand(10201)*10
    V = FunctionSpace(test_domain, ("Lagrange", 1))
    u = Function(V)
    u.interpolate(fem.Constant(test_domain, 1.0))
    f = 0.5*u*u*dx
    df = fx.compute_sensitivities(f, u)
    print(fx.to_numpy(u))
    print(fx.to_numpy(df))

test_adjoint()