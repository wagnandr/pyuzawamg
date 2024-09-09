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

from .biharmonic import (
    MyAMG,
    assemble_biharmonic_operator_a,
    assemble_biharmonic_operator_b,
    assemble_laplace_operator_a,
    assemble_laplace_operator_b,
    assemble_system,
    solve_problem
)
from .biharmonic import (
     assembly_operators,
     mobilities,
     applications
)


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
    return assemble_2(M, K, eps)


def _cli():
    # problem properties
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(required=True)

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
    const_parser.set_defaults(func=_cli_run_schur)

    const_parser = subparsers.add_parser('schur:study')
    const_parser.add_argument('--smoothing-steps', type=int, default=3)
    const_parser.add_argument('--w-cycles', type=int, default=1)
    const_parser.add_argument('--maxitermg', type=int, default=1)
    const_parser.add_argument('--tolmg', type=float, default=1e-16)
    const_parser.add_argument('--precond-type', type=str, choices=('S1r', 'S1l', 'S1s', 'Smg', 'Smg_gs', 'Smg_nogs', 'S2r', 'S2l', 'S2s', 'Sd', 'S1lgs'), required=True)
    const_parser.add_argument('--operator', type=str, choices=('real-schur', 'approx-schur'), required=True)
    const_parser.add_argument('--mobility', type=str, choices=mobilities.keys(), default=list(mobilities.keys())[0])
    const_parser.add_argument('--verbose', action='store_true')
    const_parser.set_defaults(func=_cli_schur_parameter_study)

    args = parser.parse_args()
    args.func(args)


def _cli_run_schur(args):
    _precond(
        num_levels=args.num_levels,
        eps=args.eps,
        mobility=args.mobility,
        mobility_parameter=args.mobility_parameter,
        args=args
    )


def _cli_schur_parameter_study(args):
    list_eps = [1., 1e-1, 1e-2, 1e-4]
    list_num_levels = [4, 5, 6, 7, 8]
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
    return Richardson(precond=block_mat([[SOR(A[0,0], parameters={"petsc_sor_symmetric": 1, "ksp_monitor": 1})]]), A=A, maxiter=smoothing_steps, tolerance=1e-16, show=1)


class unblock(block_base):
    def matvec(self, x):
        return x[0].copy()

class block(block_base):
    def matvec(self, x):
        return block_vec([x.copy()])


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
    MepK = assemble_2(M, K, eps)
    S_real = block_mat([[Km + M * LU(df.assemble(2 * phi*psi*df.dx + eps**2 * df.inner(df.grad(phi), df.grad(psi)) * df.dx)) * M ]])
    S_approx = block_mat([[Km + Mdiag * LU(MepK) * Mdiag]])

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
        phi, psi = df.TrialFunction(V[0][0]), df.TestFunction(V[0][0])
        M = df.assemble(phi*psi*df.dx)
        Mdiag = Diag(M)
        Km = df.assemble(m * df.inner(df.grad(phi), df.grad(psi)) * df.dx)
        S_approx2 = block_mat([[Km + Mdiag * unblock() * iMepK * block() * Mdiag]])
        S_approx2.create_vec = lambda dim=1: block_vec([Km.create_vec(dim)])
        return S_approx2

    iBa = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_biharmonic_operator_a)
    iBb = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_biharmonic_operator_b)

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
        print('A ', MepK.size(0), MepK.size(1))
        S1_l_gs.name = f"> {MepK.size(0)} {MepK.size(1)}"
        return S1_l_gs

    def create_smoother_S1_r_gs(V, m, eps, usegs=True):
        phi, psi = df.TrialFunction(V[0][0]), df.TestFunction(V[0][0])
        K = df.assemble(df.inner(df.grad(phi), df.grad(psi)) * df.dx)
        M = df.assemble(phi*psi*df.dx)
        Minvdiag = InvDiag(M)
        if usegs:
            iBb_gs = create_gs_smoother(V, m, eps, args.smoothing_steps, assemble_biharmonic_operator_b)
        else:
            iBb_gs = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_biharmonic_operator_b)
        MepK = assemble_2(M, K, eps)
        S1_r_gs = block_mat([[Minvdiag]]) * block_mat([[MepK]]) * iBb_gs
        return S1_r_gs

    S1_l = iBa * block_mat([[MepK]]) * block_mat([[Minvdiag]])
    S1_l_gs = iBa_gs * block_mat([[MepK]]) * block_mat([[Minvdiag]])
    S1_r = block_mat([[Minvdiag]]) * block_mat([[MepK]]) * iBb

    S2_l = block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]]) - block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]]) * block_mat([[Km]]) * S1_l
    S2_r = block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]]) - S1_r * block_mat([[Km]]) * block_mat([[Minvdiag]]) * block_mat([[MepK]]) * block_mat([[Minvdiag]])

    def create_coarse_grid_solver(V):
        phi, psi = df.TrialFunction(V[0][0]), df.TestFunction(V[0][0])
        Km = df.assemble(m * df.inner(df.grad(phi), df.grad(psi)) * df.dx)
        K = df.assemble(df.inner(df.grad(phi), df.grad(psi)) * df.dx)
        M = df.assemble(phi*psi*df.dx)
        Mdiag = Diag(M)
        Minvdiag = InvDiag(M)
        MepK = assemble_2(M, K, eps)
        S_approx = block_mat([[Km + Mdiag * LU(MepK) * Mdiag]])
        iBa = create_multigrid(V, m, eps, args.smoothing_steps, args.smoothing_steps, args.w_cycles, args.maxitermg, args.tolmg, assemble_biharmonic_operator_a)
        print()
        S1_l = iBa * block_mat([[MepK]]) * block_mat([[Minvdiag]])
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


if __name__ == '__main__':
    _cli()
