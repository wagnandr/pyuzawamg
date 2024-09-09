import argparse
import dolfin as df
import petsc4py as p4py
import numpy as np
import math
from block import block_mat, block_vec, block_assemble
from block.block_base import block_base
from block.iterative import Richardson, ConjGrad, LGMRES, MinRes, MinRes2, iterative
from block.algebraic.petsc import LU, SOR, AMG, Diag, InvDiag

from pyuzawamg.solvers import (
    MGSolverBlock,
    create_mesh_hierarchy,
    create_prolongation_hierarchy,
)

from pyuzawamg.util import (
    CBCBlockWrapper,
    power_iteration
)


class MyAMG(iterative):
    '''
    Simple wrapper for an algebraic multigrid.
    '''
    def __init__(self, 
                 A, 
                 tolerance=1e-5, 
                 initial_guess=None, 
                 iter=None, # ?
                 maxiter=200, 
                 name=None, 
                 show=1, 
                 rprecond=None, # ?
                 nonconvergence_is_fatal=False, # ?
                 retain_guess=False, # ?
                 relativeconv=False, # ?
                 callback=None): # ?
        iterative.__init__(self, A[0], None, tolerance, initial_guess, iter, maxiter, name, show, rprecond, nonconvergence_is_fatal, retain_guess, relativeconv, callback)
        self.A = A
        self.solver = None

    @staticmethod
    def _create_solver(A):
        import pyamg
        #ml = pyamg.ruge_stuben_solver(A)
        #ml = pyamg.rootnode_solver(A)
        '''
        ml = pyamg.smoothed_aggregation_solver(
            A,
            strength=('classical', {'theta': .75}),
            symmetry='nonsymmetric',
            presmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 3}),
            postsmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 3})
        )
        '''
        ml = pyamg.ruge_stuben_solver(
            A,
            #CF=('RS', {'second_pass': True}),
            #interpolation='classical',
            strength=('classical', {'theta': .75}),
            CF=('RS', {'second_pass': True}),
            presmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 3}),
            postsmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 3}),
            max_levels=2,
            #coarse_solver='lu'
            coarse_solver='splu'
            )
        #ml = pyamg.pairwise_solver(A)
        return ml

    @staticmethod
    def _convert_petsc_mat_to_scipy(A):
        import scipy as sp
        ai, aj, av = df.as_backend_type(A).mat().getValuesCSR()
        return sp.sparse.csr_matrix((av, aj, ai))

    def method(self, B, A, x, b, tolerance, maxiter, progress, relativeconv=False, shift=0, callback=None):
        print()
        if self.solver is None:
            self.AA = MyAMG._convert_petsc_mat_to_scipy(A[0][0])
            self.solver = self._create_solver(self.AA)
        bb = b[0][:]
        x0 = x.copy()[0][:]
        self.residual = []
        def cb(x, **kwds):
            norm = np.linalg.norm(bb -  self.AA @ x)
            self.residual.append(norm)
            try:
                print(self.residual[-1]/self.residual[-2], self.residual)
            except:
                pass
        cb(x0)
        res2 = []
        xx, info = self.solver.solve(bb, x0=x0, tol=1e-20, return_info=True, cycle='V', maxiter=1, callback=cb, residuals=res2)
        #print(solver)
        x = x.copy()
        x[0][:] = xx[:]
        self.residual_rates = (np.array(self.residual[1:]) / np.array(self.residual[-1])).tolist()
        print('1 ', self.residual)
        print('2 ', res2)
        return x, self.residual_rates, [], []


def assemble_biharmonic_operator_a(Km, K, M, eps):
    '''
    Assembles the operator
        diag(M) + 2 Km + eps^2 K diag(M)^{-1} Km,
    where diag(M) is the diagonal of M.
    '''
    Km_pc = df.as_backend_type(Km).mat()
    K_pc = df.as_backend_type(K).mat()
    M_pc = df.as_backend_type(M).mat()
    diag_M = M_pc.getDiagonal()
    idiag_M = 1. / diag_M
    iM_Km = df.as_backend_type(Km.copy()).mat()
    iM_Km.diagonalScale(idiag_M, None)
    R_pc = 2. * Km_pc + (eps**2) * K_pc * iM_Km
    R_pc.setDiagonal(diag_M, p4py.PETSc.InsertMode.ADD)
    R = df.PETScMatrix(R_pc)
    return R


def assemble_2(M, K, eps):
    '''
    Assembles the operator
        diag(M) + eps^2
    '''
    diag_M = df.as_backend_type(M).mat().getDiagonal()
    K = df.as_backend_type(K.copy()).mat()
    R = (eps**2) * K
    R.setDiagonal(2*diag_M, p4py.PETSc.InsertMode.ADD)
    return df.PETScMatrix(R)


def assemble_biharmonic_operator_b(Km, K, M, eps):
    '''
    Assembles the operator
        Md + 2 Km + eps^2 Km Md^{-1} K,
    where Md is the diagonal of M.
    '''
    Km_pc = df.as_backend_type(Km).mat()
    M_pc = df.as_backend_type(M).mat()
    diag_M = M_pc.getDiagonal()
    idiag_M = 1. / diag_M
    iM_K = df.as_backend_type(K.copy()).mat()
    iM_K.diagonalScale(idiag_M, None)
    R_pc = 2. * Km_pc + (eps**2) * Km_pc * iM_K
    R_pc.setDiagonal(diag_M, p4py.PETSc.InsertMode.ADD)
    R = df.PETScMatrix(R_pc)
    return R


def assemble_laplace_operator_a(Km, K, M, eps):
    K_pc = df.as_backend_type(K).mat()
    M_pc = df.as_backend_type(M).mat()
    R = df.PETScMatrix(K_pc + M_pc)
    return R


def assemble_laplace_operator_b(Km, K, M, eps):
    Km_pc = df.as_backend_type(Km).mat()
    M_pc = df.as_backend_type(M).mat()
    R = df.PETScMatrix(Km_pc + M_pc)
    return R


def assemble_system(V, m, eps, assembly_operator):
    phi, psi = df.TrialFunction(V), df.TestFunction(V)
    Km = df.assemble(m * df.inner(df.grad(phi), df.grad(psi)) * df.dx)
    K = df.assemble(df.inner(df.grad(phi), df.grad(psi)) * df.dx)
    M = df.assemble(phi*psi*df.dx)
    A = assembly_operator(Km, K, M, eps)
    b = df.assemble(df.Constant(0) * psi * df.dx)
    return block_mat([[A]]), block_vec([b])


def solve_problem(V, A, b, solver):
    x = A.create_vec()
    np.random.seed(0)
    x.randomize()
    np.random.seed(1)
    b.randomize()

    solver.num_iterations = 40

    res_norm_init = (b - A * x).norm()
    print(res_norm_init)
    residual_rate_list = []
    solver_new = solver(initial_guess=x)
    x = solver_new * b
    res_norm_final = (b - A * x).norm()
    print(f'resnorm {res_norm_init}, {res_norm_final}, {res_norm_final/res_norm_init}')

    residual_rate_list = np.array(solver_new.residual_rates)[4:]
    print(f'{np.mean(residual_rate_list)}, {residual_rate_list.prod()**(1./len(residual_rate_list))}')

    # write vectorial solution into functions
    u = df.Function(V, name='u')
    u.vector()[:] = x[0][:]

    # output to files
    file_u = df.File('output/biharmonic_u.pvd')
    file_u << u, 0

    return residual_rate_list


def _demo(
        smoothing_steps,
        w_cycles,
        num_levels,
        eps,
        operator,
        mobility,
        mobility_parameter,
        application,
        mg_type
):

    presmoothing_steps = postsmoothing_steps = smoothing_steps
    assembly_operator = assembly_operators[operator]
    m = mobilities[mobility](mobility_parameter)

    # create the coarse mesh:
    N = 2
    mesh_coarse = df.RectangleMesh(df.Point(-4,-1), df.Point(4,1), 4*N, N, 'left')

    # setup a mesh hierarchy
    meshes = create_mesh_hierarchy(mesh_coarse, num_levels)
    # create spaces
    V = list(map(lambda m: [df.FunctionSpace(m, 'P', 1)], meshes))

    print(f'dimension = {V[0][0].dim()}')

    # prolongation operators
    P = create_prolongation_hierarchy(V) 

    # assembled systems
    assembly_results = [assemble_system(V_, m, df.Constant(eps), assembly_operator) for V_, in V]
    A = [A for A, b in assembly_results]
    b = [b for A, b in assembly_results]

    # coarse grid solver
    Ainv_coarse = block_mat([[LU(A[-1][0,0])]])

    presmoother = [CBCBlockWrapper(Richardson(precond=block_mat([[SOR(A_[0,0], parameters={"petsc_sor_forward": 1, "ksp_monitor": 1})]]), A=A_, maxiter=1, tolerance=1e-16, show=0)) for A_ in A]
    postsmoother = [CBCBlockWrapper(Richardson(precond=block_mat([[SOR(A_[0,0], parameters={"petsc_sor_backward": 1})]]), A=A_, maxiter=1, tolerance=1e-16, show=0)) for A_ in A]

    # setup full solver

    if mg_type == 'gmg':
        solver = MGSolverBlock(
            presmoother=presmoother,
            postsmoother=postsmoother,
            coarse_grid_solver=Ainv_coarse,
            A=A,
            P=P,
            presmoothing_steps=presmoothing_steps,
            postsmoothing_steps=postsmoothing_steps,
            w_cycles=w_cycles,
            maxiter=1,
            tolerance=1e-16)
    elif mg_type == 'amg':
        solver = MyAMG(A[0])

    return applications[application](V[0][0], A[0], b[0], solver, verbose=True)


assembly_operators = {
    'biharmonic_a': assemble_biharmonic_operator_a,
    'biharmonic_b': assemble_biharmonic_operator_b,
    'laplace_a': assemble_laplace_operator_a, 
    'laplace_b': assemble_laplace_operator_b, 
}

mobilities = {
    'constant': lambda c: df.Constant(c),
    'jump': lambda c: df.Expression(f'(x[0] <= 0) ? 1 : {c}', degree=0),
    'tanh': lambda c: df.Expression(f'0.5*(tanh(x[0]/{c}) + 1)', degree=0)
}

applications = {
    'power_iteration': lambda V, A, b, solver, verbose: (power_iteration(A, solver, verbose), V.dim()),
    'solve_problem': lambda V, A, b, solver, verbose: (solve_problem(V, A, b, solver), V.dim())
}


def _cli():
    # problem properties
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(required=True)

    run_parser = subparsers.add_parser('mg:run')
    run_parser.add_argument('--smoothing-steps', type=int, default=3)
    run_parser.add_argument('--w-cycles', type=int, default=1)
    run_parser.add_argument('--num-levels', type=int, default=6)
    run_parser.add_argument('--eps', type=float, default=1.)
    run_parser.add_argument('--operator', type=str, choices=assembly_operators.keys(), default=list(assembly_operators.keys())[0])
    run_parser.add_argument('--mobility', type=str, choices=mobilities.keys(), default=list(mobilities.keys())[0])
    run_parser.add_argument('--mobility-parameter', type=float, default=1.)
    run_parser.add_argument('--application', type=str, choices=applications.keys(), default=list(applications.keys())[0])
    run_parser.add_argument('--mg-type', type=str, choices=['amg', 'gmg'], default='gmg')
    run_parser.set_defaults(func=_cli_run)

    const_parser = subparsers.add_parser('mg:study_constant')
    const_parser.add_argument('--operator', type=str, choices=assembly_operators.keys(), default=list(assembly_operators.keys())[0])
    const_parser.add_argument('--mg-type', type=str, choices=['amg', 'gmg'], default='gmg')
    const_parser.set_defaults(func=_cli_parameter_study_constant)

    const_parser = subparsers.add_parser('mg:study_variable')
    const_parser.add_argument('--operator', type=str, choices=assembly_operators.keys(), default=list(assembly_operators.keys())[0])
    const_parser.add_argument('--mobility', type=str, choices=('tanh', 'jump'), required=True)
    const_parser.add_argument('--mg-type', type=str, choices=['amg', 'gmg'], default='gmg')
    const_parser.set_defaults(func=_cli_parameter_study_variable)

    args = parser.parse_args()
    args.func(args)


def _cli_run(args):
    _demo(
        smoothing_steps=args.smoothing_steps,
        w_cycles=args.w_cycles,
        num_levels=args.num_levels,
        eps=args.eps,
        operator=args.operator,
        mobility=args.mobility,
        mobility_parameter=args.mobility_parameter,
        application=args.application,
        mg_type=args.mg_type
    )


def _cli_parameter_study_constant(args):
    list_eps = [1., 1e-1, 1e-2, 1e-4]
    list_num_levels = [4, 5, 6, 7, 8]
    list_mobility = [1, 1e-2]

    list_tables= []
    for mobility in list_mobility:
        table = []
        for num_levels in list_num_levels:
            row = []
            for eps in list_eps:
                radius, dof = _demo(
                    smoothing_steps=3,
                    w_cycles=1,
                    num_levels=num_levels,
                    eps=eps,
                    operator=args.operator,
                    mobility='constant',
                    mobility_parameter=mobility,
                    application='power_iteration',
                    mg_type=args.mg_type
                )
                row.append(radius)
            table.append(row)
        list_tables.append(np.c_[list_num_levels, table])
        print(list_tables)
    
    for table,mob in zip(list_tables, list_mobility):
        import tabulate
        print(f'  mobility = {mob}')
        tab = tabulate.tabulate(table, headers=['level'] + [f'eps={eps}' for eps in list_eps])
        print(tab)
        print()
        tab = tabulate.tabulate(table, headers=['level'] + [f'eps={eps}' for eps in list_eps], tablefmt="latex")
        print(tab)
        print()

def _cli_parameter_study_variable(args):
    list_eps = [1., 1e-1, 1e-2, 1e-4]
    list_num_levels = [4, 5, 6, 7, 8]
    list_mobility = [1, 1e-2, 1e-4, 1e-6]

    list_tables= []
    for mobility in list_mobility:
        table = []
        for num_levels in list_num_levels:
            row = []
            for eps in list_eps:
                radius, dof = _demo(
                    smoothing_steps=3,
                    w_cycles=1,
                    num_levels=num_levels,
                    eps=eps,
                    operator=args.operator,
                    #mobility='jump',
                    mobility=args.mobility,
                    mobility_parameter=mobility,
                    application='power_iteration',
                    mg_type=args.mg_type
                )
                row.append(radius)
            table.append(row)
        list_tables.append(np.c_[list_num_levels, table])
        print(list_tables)
    
    for table,mob in zip(list_tables, list_mobility):
        import tabulate
        tab = tabulate.tabulate(table, headers=['level'] + [f'eps={eps}' for eps in list_eps])
        print(f'  1 vs {mob}')
        print(tab)
        print()
        tab = tabulate.tabulate(table, headers=['level'] + [f'eps={eps}' for eps in list_eps], tablefmt="latex")
        print(tab)
        print()


if __name__ == '__main__':
    _cli()
