import numpy as np
from collections import OrderedDict
from math import *
from sympy import *


class FEM1D:

    # global matrix parameters
    k_matrix_global = 0
    f_vector_global = 0
    fi_vector_global = 0
    fi_vector_global_map = 0
    b_vector_global = 0

    mesh_global = []

    def __init__(self):

        self.domain = OrderedDict()
        self.mesh = []
        self.nodes = OrderedDict()
        self.nodes_loc = OrderedDict()
        self.elements = OrderedDict()
        self.elements_loc = OrderedDict()
        self.de_parameters = OrderedDict()
        self.bc = OrderedDict()
        self.l2g = []
        self.solution = []
        self.stiffness = []
        self.forcing = []
        self.boundary = []

        # problem differential equation (DE)
        self.de = 0
        # coefficients of derivatives of the DE
        self.d2T = 0
        self.dT = 0
        self.T = 0
        self.f = 0

        # weak form construction parameters
        self.A = 0
        self.B = 0
        self.C = 0
        self.D = 0
        self.E = 0
        self.F = 0
        self.G = 0

        # weak form local coordinates for elements construction parameters
        self.A_e = 0
        self.B_e = 0
        self.C_e = 0
        self.D_e = 0
        self.E_e = 0
        self.F_e = 0
        self.G_e = 0

        # the weak form of the DE
        self.weak_form_k_e_equation = 0
        self.weak_form_f_e_equation = 0
        self.k_matrix = 0
        self.f_vector = 0
        self.fi_vector = 0
        self.b_vector = 0

        # preprocess matrices
        self.k_solve = 0
        self.f_solve = 0
        self.fi_map = []
        self.b_solve = 0

        self.b_type = []
        self.b_values = 0

        # The basis functions in terms of ksi (local coordinates)
        self.s1_e = 0
        self.ds1_e = 0

        return

    def print_mesh(self):
        print(self.mesh)
        for i in self.nodes.keys():
            print(i, self.nodes[i])

        print()

        for i in self.elements.keys():
            print(i, self.elements[i])

        print()
        self.construct_l2g()

        print(self.l2g)
        return

    def set_problem_domain(self, lower_bound, higher_bound):

        self.domain['a'] = float(lower_bound)
        self.domain['b'] = float(higher_bound)

        return

    def create_1d_cartesian_grid_by_num(self, lower_bound, upper_bound, num_element):

        self.domain['a'] = float(lower_bound)
        self.domain['b'] = float(upper_bound)
        self.mesh = np.linspace(float(lower_bound), float(upper_bound), num_element + 1)
        self.mesh_global.append(self.mesh)

        for i in range(len(self.mesh)):
            self.nodes[i + 1] = {}
            self.nodes[i + 1]['id'] = i+1
            self.nodes[i+1]['x'] = round(float(self.mesh[i]), 8)
            self.nodes[i+1]['n_id_init'] = i+1
            if i != len(self.mesh)-1:
                self.elements[i + 1] = {}
                self.elements[i+1]['id'] = i+1
                self.elements[i+1]['nodes'] = [int(i+1), int(i+2)]
                self.elements[i+1]['h'] = abs(round(float(self.mesh[i] - self.mesh[i+1]), 8))
                self.elements[i+1]['j'] = abs(round(float(self.mesh[i] - self.mesh[i+1]), 8)) / 2.
                self.elements[i+1]['id_init'] = i+1

        for i in self.nodes.keys():
            self.nodes_loc[self.nodes[i]['x']] = self.nodes[i]

        return

    def construct_l2g(self):

        for i in self.elements.keys():
            self.l2g.append(self.elements[i]['nodes'])

        return

    def steady_heat_transfer(self, Area, Premeter, length, diameter,
                             conductivity, specific_heat, T_infinity,
                             num_elements, precision, prints):

        init_printing(use_unicode=True)
        x, T, k, A, d, P, h, T_inf = symbols('x, T, k, A, d, P, h, T_inf')

        # defining the functions
        T = Function('T')

        d = eval(diameter)
        k = eval(conductivity)
        h = eval(specific_heat)
        T_inf = eval(T_infinity)
        A = eval(Area)
        P = eval(Premeter)

        self.set_problem_domain(0, length)
        self.create_1d_cartesian_grid_by_num(num_elements)
        self.de = simplify(-diff(k*A*diff(T(x), x), x) + h * P * (T(x)-T_inf)).expand()

        self.d2T = self.de.expand().coeff(Derivative(T(x), x, x))
        self.dT = self.de.expand().coeff(Derivative(T(x), x))
        self.T = self.de.expand().coeff(T(x))
        self.f = simplify(-(self.de - (self.d2T * Derivative(T(x), x, x) +
                                       self.dT * Derivative(T(x), x) + self.T * T(x))))

        # construct the weak form

        residual, w, weighted_residual = symbols('residual, w, weighted_residual')

        residual = self.de
        weighted_residual = simplify(w(x) * residual)

        # do the integration by parts with 10 sg

        a = weighted_residual.expand().coeff(Derivative(T(x), x, x))
        b = diff(a, x)
        self.D = N(-b.expand().coeff(w(x)), precision)
        self.A = N(-b.expand().coeff(Derivative(w(x), x)), precision)
        self.B = N(simplify(self.dT), precision)
        self.C = N(simplify(self.T), precision)
        self.E = N(simplify(self.B + self.D), precision)
        self.F = N(simplify(self.f), precision)
        self.G = N(simplify(self.A), precision)

        if prints:

            print()
            print('The DE is: \n\t', self.de)
            print()
            print('d2T/dx2 coefficients: \n\t', self.d2T)
            print()
            print('dT/dx coefficients: \n\t', self.dT, '\t It will become zero after IBP')
            print()
            print('T(x) coefficients: \n\t', self.T)
            print()
            print('f(x) coefficients: \n\t', self.f)
            print()
            print('The weighted residual is \n\t', weighted_residual.expand())
            print()
            print('After manipulations for integration by parts (IBP)')
            print('coefficients of LHS dw/dx * dT/dx: \n\t', self.A)
            print()
            print('coefficients of LHS w * dT/dx: \n\t', self.E, '\t It is always zero due to IBP')
            print()
            print('coefficients of LHS w * T: \n\t', self.C)
            print()
            print('coefficients of RHS F: \n\t', self.F)
            print()
            print('coefficients of RHS dT/dx in SV * : \n\t', self.G)
            print()

        return

    def general_de(self, A_input, B_input, C_input, D_input, E_input, precision, prints):

        init_printing(use_unicode=True)
        x, fi, A, B, C, D, E = symbols('x, fi, A, B, C, D, E')

        # defining the functions
        fi = Function('fi')

        A = eval(A_input)
        B = eval(B_input)
        C = eval(C_input)
        D = eval(D_input)
        E = eval(E_input)

        self.de = simplify(A * diff(B * diff(fi(x), x), x) + C * diff(fi(x), x) + D * fi(x) + E).expand()

        self.d2fi = self.de.expand().coeff(Derivative(fi(x), x, x))
        self.dfi = self.de.expand().coeff(Derivative(fi(x), x))
        self.fi = self.de.expand().coeff(fi(x))
        self.f = simplify(-(self.de - (self.d2fi * Derivative(fi(x), x, x) +
                                       self.dfi * Derivative(fi(x), x) + self.fi * fi(x))))

        # construct the weak form

        residual, w, weighted_residual = symbols('residual, w, weighted_residual')

        residual = self.de
        weighted_residual = simplify(w(x) * residual)

        # do the integration by parts with 10 sg

        a = weighted_residual.expand().coeff(Derivative(fi(x), x, x))
        b = diff(a, x)
        self.D = N(-b.expand().coeff(w(x)), precision)
        self.A = N(-b.expand().coeff(Derivative(w(x), x)), precision)
        self.B = N(simplify(self.dfi), precision)
        self.C = N(simplify(self.fi), precision)
        self.E = N(simplify(self.B + self.D), precision)
        self.F = N(simplify(self.f), precision)
        self.G = N(simplify(self.A), precision)

        x, ksi, Je, xes, xee = symbols('x,ksi,Je,xes,xee')
        # change in variables for local element system

        self.A_e = simplify(self.A.subs(x, 0.5 * (ksi * Je * 2. + (xes + xee))))
        self.B_e = simplify(self.B.subs(x, 0.5 * (ksi * Je * 2. + (xes + xee))))
        self.C_e = simplify(self.C.subs(x, 0.5 * (ksi * Je * 2. + (xes + xee))))
        self.D_e = simplify(self.D.subs(x, 0.5 * (ksi * Je * 2. + (xes + xee))))
        self.E_e = simplify(self.E.subs(x, 0.5 * (ksi * Je * 2. + (xes + xee))))
        self.F_e = simplify(self.F.subs(x, 0.5 * (ksi * Je * 2. + (xes + xee))))
        self.G_e = simplify(self.G.subs(x, 0.5 * (ksi * Je * 2. + (xes + xee))))

        if prints:

            print()
            print('The DE is: \n\t', self.de)
            print()
            print('d2fi/dx2 coefficients: \n\t', self.d2fi)
            print()
            print('dfi/dx coefficients: \n\t', self.dfi)
            print()
            print('fi(x) coefficients: \n\t', self.fi)
            print()
            print('f(x) coefficients: \n\t', self.f)
            print()
            print('The weighted residual is \n\t', weighted_residual.expand())
            print()
            print('After manipulations for integration by parts (IBP)')
            print('coefficients of LHS dw/dx * dfi/dx: \n\t', self.A)
            print()
            print('coefficients of LHS w * dfi/dx: \n\t', self.E)
            print()
            print('coefficients of LHS w * fi: \n\t', self.C)
            print()
            print('coefficients of RHS F: \n\t', self.F)
            print()
            print('coefficients of RHS dfi/dx in SV * : \n\t', self.G)
            print()

        return

    def construct_first_order_shape_functions(self):

        self.s1_e = lambda i, ksi: 0.5 * (1 - ksi) if i == 1 else 0.5 * (1 + ksi) if i == 2 else 0
        self.ds1_e = lambda i, ksi: -0.5 if i == 1 else 0.5 if i == 2 else 0

        return

    def construct_the_weak_form(self, order):

        file_path = '/home/ariya/Desktop/codes/ME582/FEM/equation.txt'

        if order == 1:
            self.construct_first_order_shape_functions()

            # construct the solution string
            k_part1 = '(' + str(self.A_e) + ') * (self.ds1_e(i,ksi) * self.ds1_e(j,ksi) / (Je**2)) '
            k_part2 = '(' + str(self.E_e) + ') * (self.s1_e(i,ksi) * self.ds1_e(j,ksi) / (Je)) '
            k_part3 = '(' + str(self.C_e) + ') * (self.s1_e(i,ksi) * self.s1_e(j,ksi)) '
            k_ij = '(' + k_part1 + ' + ' + k_part2 + ' + ' + k_part3 + ') * Je'
            self.weak_form_k_e_equation = k_ij

            f_ij = '(' + str(self.F_e) + ') * (self.s1_e(i,ksi)) * Je '
            self.weak_form_f_e_equation = f_ij

        fid = open(file_path, 'w')
        fid.write(self.weak_form_k_e_equation + '\n')
        fid.write(self.weak_form_f_e_equation + '\n')
        fid.close()

        # now we have strings that are functions of ksi and element shape and Galarkin basis functions

        return

    def three_pt_GQ_k(self, x_e_start, x_e_end, jacobian, ii, jj):

        integral_result = 0
        xes = x_e_start
        xee = x_e_end
        Je = jacobian
        i = ii
        j = jj

        for k in [[5./9., sqrt(3./5.)], [8./9., 0], [5./9., -sqrt(3./5.)]]:
            ksi = k[1]
            weight = k[0]
            integral_result += weight * eval(self.weak_form_k_e_equation)

        return integral_result

    def three_pt_GQ_f(self, x_e_start, x_e_end, jacobian, ii):

        integral_result = 0
        xes = x_e_start
        xee = x_e_end
        Je = jacobian
        i = ii

        for k in [[5./9., sqrt(3./5.)], [8./9., 0], [5./9., -sqrt(3./5.)]]:
            ksi = k[1]
            weight = k[0]
            integral_result += weight * eval(self.weak_form_f_e_equation)

        return integral_result

    def construct_matrices(self, prints=False):

        # initialize k
        nn = len(self.nodes.keys())
        ne = len(self.elements.keys())
        self.k_matrix = np.zeros((nn, nn))
        self.f_vector = np.zeros((nn, 1))
        self.fi_vector = np.zeros((nn, 1))
        self.b_vector = np.zeros((nn, 1))
        self.b_type = [0] * nn
        self.b_values = [0] * nn

        self.construct_l2g()

        for kk in range(1, ne + 1):
            local_node_1 = self.elements[kk]['nodes'][0]
            local_node_2 = self.elements[kk]['nodes'][1]
            xestart = self.nodes[local_node_1]['x']
            xeend = self.nodes[local_node_2]['x']
            jacobian = self.elements[kk]['j']

            for ii in range(1, 3):
                for jj in range(1, 3):
                    GQ_k_result = self.three_pt_GQ_k(xestart, xeend, jacobian, ii, jj)
                    self.k_matrix[self.l2g[kk-1][ii-1]-1][self.l2g[kk-1][jj-1]-1] += GQ_k_result

                GQ_f_result = self.three_pt_GQ_f(xestart, xeend, jacobian, ii)
                self.f_vector[self.l2g[kk - 1][ii - 1] - 1][0] += GQ_f_result
        if prints:
            print('K = \n', self.k_matrix)
            print('F = \n', self.f_vector)
            print('B = \n', self.b_vector)

        return

    def add_boundary_conditions(self, bc_type, node_loc, expression):

        self.b_type[self.nodes_loc[node_loc]['id']-1] = bc_type
        self.b_values[self.nodes_loc[node_loc]['id']-1] = expression

        return

    def pre_process(self, prints=False):

        def get_the_normal(a):

            if a == self.domain['a']:
                normal_val = -1.
            elif a == self.domain['b']:
                normal_val = 1.
            else:
                normal_val = 0.

            return normal_val

        j = 0   # keeps track of the ones removed

        # there was a problem here only the references were saved in the new array
        # solved it using copy()
        self.k_solve = self.k_matrix.copy()
        self.f_solve = self.f_vector.copy()
        self.b_solve = self.b_vector.copy()

        for i in range(len(self.b_type)):
            x, func, fi = symbols('x, func, fi')
            if self.b_type[i] in [1, 1., '1', '1.']:

                # remove the ith row of all the elements
                a = self.k_solve.copy()
                a = np.delete(a, i - j, 0)
                self.k_solve = a.copy()
                a = self.f_solve.copy()
                a = np.delete(a, i - j, 0)
                self.f_solve = a.copy()
                a = self.b_solve.copy()
                a = np.delete(a, i - j, 0)
                self.b_solve = a.copy()

                # add the value to fi
                fi_val = eval(self.b_values[i])
                self.fi_vector[i] = float(fi_val)
                self.f_solve -= (self.k_solve[:, i-j:i-j+1]) * fi_val

                # remove the ith column of k
                a = self.k_solve.copy()
                a = np.delete(a, i - j, 1)
                self.k_solve = a.copy()

                j += 1

            elif self.b_type[i] in [2, 2., '2', '2.']:
                self.fi_map.append(i)
                bc_normal = get_the_normal(self.nodes[i+1]['x'])
                f = eval(self.b_values[i])
                b = self.G * bc_normal * f
                x = self.nodes[i+1]['x']
                self.b_vector[i] = eval(str(b))
                self.b_solve[i-j] = eval(str(b))

            elif self.b_type[i] in [3, 3., '3', '3.']:
                self.fi_map.append(i)
                bc_normal = get_the_normal(self.nodes[i+1]['x'])
                f = eval(self.b_values[i-j])
                b = self.G * bc_normal * f
                beta = b.expand().coeff(fi)
                gamma = b - beta * fi

                x = self.nodes[i+1]['x']
                self.b_vector[i] = eval(str(gamma))
                self.b_solve[i-j] = eval(str(gamma))
                self.k_solve[i-j, i-j] -= eval(str(beta))

            elif self.b_type[i] in [0., 0, 41, 41., '41', '41.']:
                # primary variables to be solve 0 for the same material, 41 for connection
                self.fi_map.append(i)
            
        if prints:
            print('k_mat\n', self.k_matrix)
            print('k_sol\n', self.k_solve)
            print('f_mat\n', self.f_vector)
            print('f_sol\n', self.f_solve)
            print('b_mat\n', self.b_vector)
            print('b_sol\n', self.b_solve)
            print('fi_mat\n', self.fi_vector)
            print('fi_map\n', self.fi_map)

        return

    def construct_solver_matrix(self, prints=False):

        if type(self.k_matrix_global) is int:

            FEM1D.k_matrix_global = self.k_solve.copy()
            FEM1D.f_vector_global = self.f_solve.copy()
            FEM1D.b_vector_global = self.b_solve.copy()
            FEM1D.fi_vector_global = self.fi_vector.copy()
            FEM1D.fi_vector_global_map = self.fi_map

        else:
            l2 = self.k_solve.shape[0]
            l1 = FEM1D.k_matrix_global.shape[0]

            new_mat = np.zeros((l1+l2-1, l1+l2-1))
            new_mat[:l1, :l1] += FEM1D.k_matrix_global[:l1, :l1]
            new_mat[l1-1:, l1-1:] += self.k_solve
            FEM1D.k_matrix_global = new_mat.copy()

            new_mat = np.zeros((l1 + l2-1, 1))
            new_mat[:l1, 0:1] += FEM1D.f_vector_global[:l1, 0:1]
            new_mat[l1-1:, 0:1] += self.f_solve[:l2, 0:1]
            FEM1D.f_vector_global = new_mat.copy()

            new_mat = np.zeros((l1 + l2-1, 1))
            new_mat[:l1, 0:1] += FEM1D.b_vector_global[:l1, 0:1]
            new_mat[l1-1:, 0:1] += self.b_solve[:l2, 0:1]
            FEM1D.b_vector_global = new_mat.copy()

            l3 = FEM1D.fi_vector_global.shape[0]
            l4 = self.fi_vector.shape[0]

            new_mat = np.zeros((l3 + l4-1, 1))
            new_mat[:l3, 0:1] = FEM1D.fi_vector_global[:l3, 0:1]
            new_mat[l3-1:, 0:1] = self.fi_vector[:l4, 0:1]
            FEM1D.fi_vector_global = new_mat.copy()

            new_mat = []
            new_mat.extend(FEM1D.fi_vector_global_map)
            for i in range(1, len(self.fi_map)):
                new_mat.append(self.fi_map[i]+FEM1D.fi_vector_global_map[-1])
            FEM1D.fi_vector_global_map = new_mat

        if prints:
            print(FEM1D.k_matrix_global)
            print(FEM1D.f_vector_global)
            print(FEM1D.b_vector_global)
            print(FEM1D.fi_vector_global)
            print(FEM1D.fi_vector_global_map)

        return

    def solve(self):

        solution_1 = np.matmul(np.linalg.inv(FEM1D.k_matrix_global), FEM1D.f_vector_global)
        solution_2 = np.matmul(np.linalg.inv(FEM1D.k_matrix_global), FEM1D.b_vector_global)
        solution = solution_1 + solution_2

        l1 = solution.shape[0]

        for i in range(l1):
            index = FEM1D.fi_vector_global_map[i]
            FEM1D.fi_vector_global[index, 0] = solution[i, 0]

        print(FEM1D.fi_vector_global)

        return

