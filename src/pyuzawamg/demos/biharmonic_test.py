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

def assemble_2_(Km, K, M, eps):
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


def _run():
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
    run_parser.set_defaults(func=_run_demo)

    const_parser = subparsers.add_parser('mg:study_constant')
    const_parser.add_argument('--operator', type=str, choices=assembly_operators.keys(), default=list(assembly_operators.keys())[0])
    const_parser.add_argument('--mg-type', type=str, choices=['amg', 'gmg'], default='gmg')
    const_parser.set_defaults(func=_parameter_study_constant)

    const_parser = subparsers.add_parser('mg:study_variable')
    const_parser.add_argument('--operator', type=str, choices=assembly_operators.keys(), default=list(assembly_operators.keys())[0])
    const_parser.add_argument('--mobility', type=str, choices=('tanh', 'jump'), required=True)
    const_parser.add_argument('--mg-type', type=str, choices=['amg', 'gmg'], default='gmg')
    const_parser.set_defaults(func=_parameter_study_jump)

    const_parser = subparsers.add_parser('schur:run')
    const_parser.add_argument('--smoothing-steps', type=int, default=3)
    const_parser.add_argument('--w-cycles', type=int, default=1)
    const_parser.add_argument('--num-levels', type=int, default=6)
    const_parser.add_argument('--maxitermg', type=int, default=1)
    const_parser.add_argument('--tolmg', type=float, default=1e-16)
    const_parser.add_argument('--eps', type=float, default=1.)
    const_parser.add_argument('--precond-type', type=str, choices=('S1r', 'S1l', 'S1s', 'Smg', 'S1slr', 'S1srl', 'S2r', 'S2l', 'S2s', 'Sd', 'S1lgs'), required=True)
    const_parser.add_argument('--operator', type=str, choices=('real-schur', 'approx-schur'), required=True)
    const_parser.add_argument('--mobility', type=str, choices=mobilities.keys(), default=list(mobilities.keys())[0])
    const_parser.add_argument('--mobility-parameter', type=float, default=1.)
    const_parser.add_argument('--verbose', action='store_true')
    const_parser.set_defaults(func=_run_precond)

    const_parser = subparsers.add_parser('schur:study')
    const_parser.add_argument('--smoothing-steps', type=int, default=3)
    const_parser.add_argument('--w-cycles', type=int, default=1)
    const_parser.add_argument('--maxitermg', type=int, default=1)
    const_parser.add_argument('--tolmg', type=float, default=1e-16)
    const_parser.add_argument('--precond-type', type=str, choices=('S1r', 'S1l', 'S1s', 'Smg', 'Smg_gs', 'Smg_nogs', 'S2r', 'S2l', 'S2s', 'Sd', 'S1lgs'), required=True)
    const_parser.add_argument('--operator', type=str, choices=('real-schur', 'approx-schur'), required=True)
    const_parser.add_argument('--mobility', type=str, choices=mobilities.keys(), default=list(mobilities.keys())[0])
    const_parser.add_argument('--verbose', action='store_true')
    const_parser.set_defaults(func=_schur_parameter_study)

    saddle_parser = subparsers.add_parser('saddle:run')
    saddle_parser.add_argument('--smoothing-steps', type=int, default=3)
    saddle_parser.add_argument('--w-cycles', type=int, default=1)
    saddle_parser.add_argument('--num-levels', type=int, default=6)
    saddle_parser.add_argument('--maxitermg', type=int, default=1)
    saddle_parser.add_argument('--eps', type=float, default=1.)
    saddle_parser.add_argument('--tolmg', type=float, default=1e-16)
    saddle_parser.add_argument('--precond-type', type=str, choices=('S1r', 'S1l', 'S1s', 'S2r', 'S2l', 'S2s', 'Sd', 'S1lgs'), required=True)
    saddle_parser.add_argument('--operator', type=str, choices=('real-schur', 'approx-schur'), required=True)
    saddle_parser.add_argument('--mobility', type=str, choices=mobilities.keys(), default=list(mobilities.keys())[0])
    saddle_parser.add_argument('--mobility-parameter', type=float, default=1.)
    saddle_parser.add_argument('--verbose', action='store_true')
    saddle_parser.set_defaults(func=_saddle_run)

    args = parser.parse_args()
    args.func(args)


def _run_precond(args):
    _precond(
        num_levels=args.num_levels,
        eps=args.eps,
        mobility=args.mobility,
        mobility_parameter=args.mobility_parameter,
        args=args
    )


def _schur_parameter_study(args):
    #list_eps = [1., 1e-2, 1e-4, 1e-6]
    #list_eps = [1., 1e-1, 1e-2, 1e-4]
    list_eps = [1., 1e-1, 1e-2, 1e-4]
    #list_num_levels = [2, 3, 4]
    list_num_levels = [4, 5, 6, 7, 8]
    #list_num_levels = [4, 5]
    #list_mobility = [1, 1e-2, 1e-4, 1e-6]
    #list_mobility = [1, 1e-4]
    if args.mobility == "tanh":
        list_mobility = [1., 1e-1, 1e-2]
    else:
        list_mobility = [1, 1e-4, 1e-8]

    list_tables= []
    for mobility in list_mobility:
        table = []
        for num_levels in list_num_levels:
            row = []
            for eps in list_eps:
                num_iter, conv = _precond(
                    num_levels=num_levels,
                    eps=eps,
                    mobility=args.mobility,
                    mobility_parameter=mobility,
                    args=args
                )
                row.append(num_iter if conv else '-')
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


def create_multigrid(V, m, eps, presmoothing_steps, postsmoothing_steps, w_cycles, maxitermg, tolmg, assembly_operator, show=0):
    # prolongation operators
    P = create_prolongation_hierarchy(V) 

    assembly_results = [assemble_system(V_, m, df.Constant(eps), assembly_operator) for V_, in V]
    A = [A for A, b in assembly_results]
    b = [b for A, b in assembly_results]

    # coarse grid solver
    Ainv_coarse = block_mat([[LU(A[-1][0,0])]])

    presmoother = [CBCBlockWrapper(Richardson(precond=block_mat([[SOR(A_[0,0], parameters={"petsc_sor_forward": 1, "ksp_monitor": 1})]]), A=A_, maxiter=1, tolerance=1e-16, show=0)) for A_ in A]
    postsmoother = [CBCBlockWrapper(Richardson(precond=block_mat([[SOR(A_[0,0], parameters={"petsc_sor_backward": 1, "ksp_monitor": 1})]]), A=A_, maxiter=1, tolerance=1e-16, show=0)) for A_ in A]


    # setup full solver
    return MGSolverBlock(
        presmoother=presmoother,
        postsmoother=postsmoother,
        coarse_grid_solver=Ainv_coarse,
        A=A,
        P=P,
        presmoothing_steps=presmoothing_steps,
        postsmoothing_steps=postsmoothing_steps,
        w_cycles=w_cycles,
        show=show,
        maxiter=maxitermg,
        tolerance=tolmg)

def create_gs_smoother(V, m, eps, smoothing_steps, assembly_operator):
    A, b = assemble_system(V[0][0], m, df.Constant(eps), assembly_operator)
    #return block_mat([[Richardson(precond=block_mat([[SOR(A[0,0], parameters={"petsc_sor_backward": 1, "ksp_monitor": 1})]]), A=A, maxiter=1, tolerance=1e-16, show=0)]])
    return Richardson(precond=block_mat([[SOR(A[0,0], parameters={"petsc_sor_symmetric": 1, "ksp_monitor": 1})]]), A=A, maxiter=smoothing_steps, tolerance=1e-16, show=1)
    #return Richardson(precond=block_mat([[LU(A[0,0])]]), A=A, maxiter=smoothing_steps, tolerance=1e-16, show=1)
    #return block_mat([[SOR(A[0,0], parameters={"petsc_sor_forward": 1, "ksp_monitor": 1, "pc_sor_lits": smoothing_steps, "ksp_monitor": 1})]])


class unblock(block_base):
    def matvec(self, x):
        return x[0].copy()

class block(block_base):
    def matvec(self, x):
        return block_vec([x.copy()])

class wrap(block_base):
    def __init__(self, towrap):
        self.towrap = towrap

    def matvec(self, x):
        xx = block_vec([x])



def _saddle_run(args):
    mobility = args.mobility
    mobility_parameter = args.mobility_parameter
    num_levels = args.num_levels
    eps = args.eps

    N = 2
    mesh_coarse = df.RectangleMesh(df.Point(-4,-1), df.Point(4,1), 4*N, N, 'left')

    m = mobilities[mobility](mobility_parameter)

    # setup a mesh hierarchy
    meshes = create_mesh_hierarchy(mesh_coarse, num_levels)
    # create spaces
    V = list(map(lambda m: [df.FunctionSpace(m, 'P', 1)], meshes))

    W = [V[0][0], V[0][0]]

    u, p = map(df.TrialFunction, W)
    v, q = map(df.TestFunction, W)

    eps2 = df.Constant(eps**2)
    a = [[0, 0], [0, 0]]

    a[0][0] = eps2 *df.inner(df.grad(u), df.grad(v))*df.dx + df.inner(u, v)*df.dx
    a[0][1] = df.inner(p, v)*df.dx
    a[1][0] = df.inner(u, q)*df.dx
    a[1][1] = -m * df.inner(df.grad(p), df.grad(q)) * df.dx

    L = [df.inner(df.Constant(0), v)*df.dx, df.inner(df.Constant(0), q)*df.dx]

    A, b = map(block_assemble, (a, L))

    K = df.assemble(df.inner(df.grad(p), df.grad(q)) * df.dx)
    M = df.assemble(p*q*df.dx)
    Km = df.assemble(m * df.inner(df.grad(p), df.grad(q)) * df.dx)
    Mdiag = Diag(M)
    Minvdiag = InvDiag(M)
    MepK = assemble_2(M, K, eps)
    iBa = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_biharmonic_operator_a)
    iBb = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_biharmonic_operator_b)
    iMepK = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_2_)
    #S1_l = unblock() * iBa * MepK * Minvdiag
    #S1_l = iBa * MepK * Minvdiag
    S1_r = block_mat([[Minvdiag]]) * block_mat([[MepK]]) * iBb
    #iBa = AfterIteration(iBa, iBa.list_A[-1])
    S1_l = iBa * block_mat([[MepK]]) * block_mat([[Minvdiag]])
    #S2_l = block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]]) - block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]]) * block_mat([[Km]]) * S1_l
    #S2_r = block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]]) - S1_r * block_mat([[Km]]) * block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]])


    iBb = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_biharmonic_operator_b)
    S_approx = block_mat([[Km + Mdiag * LU(MepK) * Mdiag]])
    S_approx2 = block_mat([[Km + Mdiag * unblock() * iMepK * block() * Mdiag]])
    #S_real = block_mat([[Km + M * LU(df.assemble(2 * p*q*df.dx + eps**2 * df.inner(df.grad(p), df.grad(q)) * df.dx)) * M ]])

    if args.precond_type == 'S1r':
        S_p = S1_r
    elif args.precond_type == 'S1l':
        S_p = S1_l
    elif args.precond_type == 'Sd':
        S_p = ConjGrad(S_approx, precond=S1_l, robustresidual=True)
    else:
        raise RuntimeError('Unknown precond_type ' + args.precond_type)
    
    S_p = S1_r * S_approx2 * S1_l

    P = block_mat([[LU(A[0,0]), 0], [0, unblock() * S_p * block()]])

    x = A.create_vec()
    np.random.seed(0)
    x.randomize()
    xx = x.copy()

    def mycb(**args):
        res = xx - A * args['x']
        res2 = P * res
        resn = res.norm()
        res2n = res2.inner(res)
        print(args, resn, math.sqrt(res2n))

    if args.verbose:
        callback = mycb 
    else:
        callback = lambda **args: args 
    solver = MinRes(A, precond=P, callback=callback, maxiter=100, relativeconv=True, show=1, tolerance=1e-8)

    bb = solver * x

    print(x[0][:])
    print(bb[0][:])
    print( (x - A * bb).norm())
    print( (P*(x - A * bb)).norm())

    print(solver.iterations, solver.converged)

    pass


class CBCBlockWrapper2:
    """
    Small wrapper to convert a block solver in our solver format.
    """
    def __init__(self, solver):
        self.solver = solver

    def mult(self, src, dst):
        #print(f'src , {src.norm()}')
        res = self.solver * src
        #print(f'res , {res.norm()}')
        #print(dst, res)
        #print(dst[0].size())
        #print(res[0].size())
        dst[0][:] = res[0][:]

class AfterIteration(block_base):
    """
    Small wrapper to convert a block solver in our solver format.
    """
    def __init__(self, solver, operator):
        self.solver = solver
        self.operator = operator

    def matvec(self, src):
        res = self.solver * src
        d = src - self.operator * res
        w = self.solver * d 
        res = res + w 
        d = src - self.operator * res
        w = self.solver * d 
        res = res + w 
        d = src - self.operator * res
        w = self.solver * d 
        res = res + w 
        d = src - self.operator * res
        w = self.solver * d 
        res = res + w 
        return res

def _precond(
    num_levels,
    eps,
    mobility,
    mobility_parameter,
    args
):

    N = 2
    mesh_coarse = df.RectangleMesh(df.Point(-4,-1), df.Point(4,1), 4*N, N, 'left')

    m = mobilities[mobility](mobility_parameter)

    # setup a mesh hierarchy
    meshes = create_mesh_hierarchy(mesh_coarse, num_levels)
    # create spaces
    V = list(map(lambda m: [df.FunctionSpace(m, 'P', 1)], meshes))

    print(f'dimension = {V[0][0].dim()}')

    # prolongation operators
    P = create_prolongation_hierarchy(V) 

    phi, psi = df.TrialFunction(V[0][0]), df.TestFunction(V[0][0])
    Km = df.assemble(m * df.inner(df.grad(phi), df.grad(psi)) * df.dx)
    K = df.assemble(df.inner(df.grad(phi), df.grad(psi)) * df.dx)
    M = df.assemble(phi*psi*df.dx)
    MpKm = df.assemble(m * phi*psi*df.dx + m * df.inner(df.grad(phi), df.grad(psi)) * df.dx)
    Mdiag = Diag(M)
    Minvdiag = InvDiag(M)
    #MepK = df.assemble(2 * phi*psi*df.dx + (eps**2) * df.inner(df.grad(phi), df.grad(psi)) * df.dx)
    MepK = assemble_2(M, K, eps)
    # 2 * M + (eps**2) * K 
    #S = block_mat([[Km + Mdiag * LU(MepK) * Mdiag]])
    S_real = block_mat([[Km + M * LU(df.assemble(2 * phi*psi*df.dx + eps**2 * df.inner(df.grad(phi), df.grad(psi)) * df.dx)) * M ]])
    #op = df.assemble(2 * phi*psi*df.dx + eps**2 * df.inner(df.grad(phi), df.grad(psi)) * df.dx)
    #S_real = block_mat([[Km + M * AfterIteration(LU(op), op) * M ]])
    S_approx = block_mat([[Km + Mdiag * LU(MepK) * Mdiag]])
    #S_approx = block_mat([[Km + Mdiag * AfterIteration(LU(MepK), MepK) * Mdiag]])

    if args.operator == 'real-schur':
        S = S_real 
    elif args.operator == 'approx-schur':
        S = S_approx 
    else:
        raise RuntimeError('Unknown operator type ' + args.operator)

    iMepK = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_2_)
    S_approx2 = block_mat([[Km + Mdiag * unblock() * iMepK * block() * Mdiag]])

    def create_S_approx2(V, usegs=False):
        if usegs:
            iMepK = create_gs_smoother(V, m, eps, args.smoothing_steps, assemble_2_)
        else:
            iMepK = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_2_)
        #iMepK = AfterIteration(iMepK, MepK)
        phi, psi = df.TrialFunction(V[0][0]), df.TestFunction(V[0][0])
        M = df.assemble(phi*psi*df.dx)
        Mdiag = Diag(M)
        Km = df.assemble(m * df.inner(df.grad(phi), df.grad(psi)) * df.dx)
        S_approx2 = block_mat([[Km + Mdiag * unblock() * iMepK * block() * Mdiag]])
        S_approx2.create_vec = lambda dim=1: block_vec([Km.create_vec(dim)])
        return S_approx2

    iBa = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_biharmonic_operator_a)
    #iBa = AfterIteration(iBa, iBa.A) 
    iBb = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_biharmonic_operator_b)
    #iBb = AfterIteration(iBb, iBb.A) 

    iBa_gs = create_gs_smoother(V, m, eps, args.smoothing_steps, assemble_biharmonic_operator_a)
    iBb_gs = create_gs_smoother(V, m, eps, args.smoothing_steps, assemble_biharmonic_operator_b)


    def create_smoother_S1_l_gs(V, m, eps, usegs=True):
        phi, psi = df.TrialFunction(V[0][0]), df.TestFunction(V[0][0])
        K = df.assemble(df.inner(df.grad(phi), df.grad(psi)) * df.dx)
        M = df.assemble(phi*psi*df.dx)
        Minvdiag = InvDiag(M)
        if usegs:
            iBa_gs = create_gs_smoother(V, m, eps, args.smoothing_steps, assemble_biharmonic_operator_a)
        else:
            iBa_gs = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_biharmonic_operator_a)
        MepK = assemble_2(M, K, eps)
        S1_l_gs = iBa_gs * block_mat([[MepK]]) * block_mat([[Minvdiag]])
        #S1_l_gs = block_mat([[MepK]]) * block_mat([[Minvdiag]])
        print('A ', MepK.size(0), MepK.size(1))
        S1_l_gs.name = f"> {MepK.size(0)} {MepK.size(1)}"
        return S1_l_gs

    def create_smoother_S1_r_gs(V, m, eps, usegs=True):
        phi, psi = df.TrialFunction(V[0][0]), df.TestFunction(V[0][0])
        K = df.assemble(df.inner(df.grad(phi), df.grad(psi)) * df.dx)
        M = df.assemble(phi*psi*df.dx)
        Minvdiag = InvDiag(M)
        # iBb_gs = create_gs_smoother(V, m, eps, args.smoothing_steps, assemble_biharmonic_operator_b)
        if usegs:
            iBb_gs = create_gs_smoother(V, m, eps, args.smoothing_steps, assemble_biharmonic_operator_b)
        else:
            iBb_gs = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_biharmonic_operator_b)
        MepK = assemble_2(M, K, eps)
        S1_r_gs = block_mat([[Minvdiag]]) * block_mat([[MepK]]) * iBb_gs
        #S1_r_gs = block_mat([[Minvdiag]]) * block_mat([[MepK]]) 
        return S1_r_gs


    # Ba = assemble_biharmonic_operator_a(Km, K, M, eps)
    # Bb = assemble_biharmonic_operator_b(Km, K, M, eps)
    #S_l = LGMRES(Ba, tolerance=1e-16, maxiter=800) * MepK * Minvdiag
    #S_r = Minvdiag * MepK * LGMRES(Bb, tolerance=1e-16, maxiter=800)

    #S_l = LU(Ba) * MepK * Minvdiag
    #S_r = Minvdiag * MepK * LU(Bb) 

    S1_l = iBa * block_mat([[MepK]]) * block_mat([[Minvdiag]])
    S1_l_gs = iBa_gs * block_mat([[MepK]]) * block_mat([[Minvdiag]])
    S1_r = block_mat([[Minvdiag]]) * block_mat([[MepK]]) * iBb

    #S2_l = block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]]) * (block_mat([[1]]) - block_mat([[Km]]) * S1_l)
    S2_l = block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]]) - block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]]) * block_mat([[Km]]) * S1_l
    #S2_r = (block_mat([[1]]) - S1_r * block_mat([[Km]]) ) * block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]]) 
    S2_r = block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]]) - S1_r * block_mat([[Km]]) * block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]])

    def create_coarse_grid_solver(V):
        phi, psi = df.TrialFunction(V[0][0]), df.TestFunction(V[0][0])
        Km = df.assemble(m * df.inner(df.grad(phi), df.grad(psi)) * df.dx)
        K = df.assemble(df.inner(df.grad(phi), df.grad(psi)) * df.dx)
        M = df.assemble(phi*psi*df.dx)
        Mdiag = Diag(M)
        Minvdiag = InvDiag(M)
        #Minvdiag = InvDiag(M)
        MepK = assemble_2(M, K, eps)
        S_approx = block_mat([[Km + Mdiag * LU(MepK) * Mdiag]])
        iBa = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_biharmonic_operator_a)
        print()
        # iBa = AfterIteration(iBa, )
        S1_l = iBa * block_mat([[MepK]]) * block_mat([[Minvdiag]])
        #S_p = ConjGrad(S_approx, precond=S1_l, robustresidual=True)
        S_p = ConjGrad(S_approx, precond=S1_l, robustresidual=True, tolerance=1e-12, maxiter=50, callback=lambda **a:print(a), relativeconv=True)
        return S_p
    
    class twogrid(block_base):
        def __init__(self, A, coarse, P, S, Sl, Sr):
            self.A = A
            self.coarse = coarse
            self.P = P
            self.S = S
            self.Sl = Sl
            self.Sr = Sr

        def matvec(self, x):
            xx = x.copy() 
            aa = self.Sl * (xx)
            r = xx - self.S*aa
            rc = self.coarse.A.create_vec()
            self.P.tmult(r, rc) 
            dc = self.coarse * rc
            d = block_vec([S[0,0].A.create_vec()])
            self.P.mult(dc, d)
            aa = aa + d
            aa = aa + self.Sr * (xx - self.S * aa)
            return aa


    assembly_results = [assemble_system(V_, m, df.Constant(eps), assemble_biharmonic_operator_a) for V_, in V]
    A = [A for A, b in assembly_results]

    if args.precond_type == 'S1r':
        S_p = S1_r
    elif args.precond_type == 'S1l':
        S_p = S1_l
    elif args.precond_type == 'S1lgs':
        S_p = S1_l_gs
    elif args.precond_type == 'S1s':
        theta = 0.5
        S_p = (1.0-theta) * S1_l + theta *S1_r
    elif args.precond_type == 'Smg_gs':
        k_start = len(V)-1
        S_p_coarse = create_coarse_grid_solver(V[k_start:])
        for k in range(k_start, 0, -1):
            print(f'creating_k {k}')
            S_p_coarse = twogrid(
                A[k-1],
                S_p_coarse,
                P[k-1], 
                create_S_approx2(V[k-1:], usegs=True),
                create_smoother_S1_l_gs(V[k-1:], m, eps,usegs=True),
                create_smoother_S1_r_gs(V[k-1:], m, eps,usegs=True)) 
        S_p = S_p_coarse
    elif args.precond_type == 'Smg':
        k_start = len(V)-1
        S_p_coarse = create_coarse_grid_solver(V[k_start:])
        for k in range(k_start, 0, -1):
            print(f'creating_k {k}')
            S_p_coarse = twogrid(
                A[k-1],
                S_p_coarse,
                P[k-1], 
                create_S_approx2(V[k-1:], usegs=False),
                create_smoother_S1_l_gs(V[k-1:], m, eps,usegs=True),
                create_smoother_S1_r_gs(V[k-1:], m, eps,usegs=True)) 
                #create_smoother_S1_l_gs(V[k-1:], m, eps,usegs=False)) 
        S_p = S_p_coarse
    elif args.precond_type == 'Smg_nogs':
        k_start = len(V)-1
        S_p_coarse = create_coarse_grid_solver(V[k_start:])
        for k in range(k_start, 0, -1):
            print(f'creating_k {k}')
            S_p_coarse = twogrid(
                A[k-1],
                S_p_coarse,
                P[k-1], 
                create_S_approx2(V[k-1:], usegs=False),
                create_smoother_S1_l_gs(V[k-1:], m, eps,usegs=False),
                create_smoother_S1_r_gs(V[k-1:], m, eps,usegs=False)) 
                #create_smoother_S1_l_gs(V[k-1:], m, eps,usegs=False)) 
        S_p = S_p_coarse
    elif args.precond_type == 'S1slr':
        S_p = S1_l * S * S1_r
    elif args.precond_type == 'S1srl':
        S_p = S1_r * S * S1_l
    elif args.precond_type == 'S2l':
        S_p = S2_l
    elif args.precond_type == 'S2r':
        S_p = S2_r
    elif args.precond_type == 'S2s':
        S_p = 0.5 * S2_r + 0.5 * S2_l
    elif args.precond_type == 'Sd':
        S_p = ConjGrad(S_approx, precond=S1_l, robustresidual=True)
    else:
        raise RuntimeError('Unknown precond_type ' + args.precond_type)

    #S_l = AMG(Ba) * MepK * Minvdiag
    #S_r = Minvdiag * MepK * AMG(Bb)

    x = S.create_vec()
    np.random.seed(0)
    x.randomize()

    print(eps**2)
    if args.verbose:
        callback = lambda *args, **kwargs: print(args, kwargs, ((x - S * kwargs['x'])).norm() if kwargs else '')
        #callback = lambda a, b, c: print('v> ', a,b,c)
    else:
        callback = lambda *args, **kargs: args 
    solver = ConjGrad(S, precond=S_p, callback=callback, maxiter=100, relativeconv=True, show=1, tolerance=1e-8, robustresidual=True)
    print('ri')
    #solver = Richardson(S, precond=S_p, callback=callback, maxiter=1, tolerance=1e-12, show=1, relativeconv=True)
    res_i = (x).norm()
    try:
        b = solver * x
        res_f = (x - S * b).norm()
        print()
        print(f'1> {res_i}, {res_f}, {res_f/res_i}')
        print()
        for i in range(30):
            d = x - S * b
            w = solver * d
            b = b + w
            res_f = (x - S * b).norm()
            print(f'1> {res_i}, {res_f}, {res_f/res_i}')

        return solver.iterations, solver.converged
    except:
        return float('nan'), False


def _run_demo(args):
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


def _parameter_study_constant(args):
    #list_eps = [1., 1e-2, 1e-4, 1e-6]
    #list_eps = [1., 1e-1, 1e-2, 1e-4]
    list_eps = [1., 1e-1, 1e-2, 1e-4]
    #list_eps = [0.]
    #list_num_levels = [2, 3, 4]
    #list_num_levels = [4, 5, 6, 7, 8]
    list_num_levels = [4, 5, 6, 7, 8]
    #list_mobility = [1, 1e-2, 1e-4, 1e-6]
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

def _parameter_study_jump(args):
    #list_eps = [1., 1e-2, 1e-4, 1e-6]
    list_eps = [1., 1e-1, 1e-2, 1e-4]
    #list_num_levels = [2, 3, 4]
    list_num_levels = [4, 5, 6, 7, 8]
    #list_num_levels = [4, 5, 6]
    #list_mobility = [1, 1e-2, 1e-4, 1e-6]
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


if __name__ == '__main__':
    _run()
    #_run_demo()
    #_parameter_study_jump()
