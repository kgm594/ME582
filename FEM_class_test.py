from ME582.FEM.FEMClass_1D import FEM1D
from sympy import *
import numpy as np

q1 = FEM1D()
q1.general_de(A_input='-1.',
              B_input='1',
              C_input='3.',
              D_input='0.',
              E_input='-1',
              precision=10,
              prints=False)

q1.construct_the_weak_form(order=1)

q1.create_1d_cartesian_grid_by_num(0, 1, 50)

q1.construct_matrices()

q1.add_boundary_conditions(3, 0, '2.5 - fi /2.')
q1.add_boundary_conditions(41, 1, '')

q1.pre_process(prints=False)

q1.construct_solver_matrix()


q11 = FEM1D()
q11.general_de(A_input='-1.',
              B_input='1',
              C_input='3.',
              D_input='0.',
              E_input='-1',
              precision=10,
              prints=False)

q11.construct_the_weak_form(order=1)

q11.create_1d_cartesian_grid_by_num(1, 2, 50)

q11.construct_matrices()

q11.add_boundary_conditions(42, 1, '2.5 - fi /2.')
q11.add_boundary_conditions(1, 2, '-10.')

q11.pre_process(prints=False)

q11.construct_solver_matrix(prints=False)

q11.solve()