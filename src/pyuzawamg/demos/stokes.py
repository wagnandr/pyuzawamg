import argparse
import dolfin as df
from block import block_assemble, block_mat
from block.iterative import MinRes, ConjGrad
from block.algebraic.petsc import LU, LumpedInvDiag

from ..solvers import (
    SmootherLower, 
    SmootherUpper, 
    SmootherSymmetric,
    MGSolver, 
    create_mesh_hierarchy,
    create_prolongation_hierarchy,
    estimate_omega
)


def assemble_system(W, bcs, stab=False, f=None):
    if f is None:
        f = df.Constant((0,0))

    u, p = map(df.TrialFunction, W)
    v, q = map(df.TestFunction, W)

    a = [[0,0],[0,0]]
    a[0][0] = df.inner(df.grad(u), df.grad(v))*df.dx
    a[0][1] = -df.div(v) * p * df.dx
    a[1][0] = -df.div(u) * q * df.dx

    if stab:
        h = df.CellDiameter(W[0].mesh())
        beta  = 0.2
        delta = beta*h*h
        a[1][1] = - delta*df.inner(df.grad(q), df.grad(p))*df.dx

    L = [0,0]
    L[0] = df.inner(f, v) * df.dx
    if stab:
        L[1] = -delta*df.inner(df.grad(q), f) * df.dx

    A, _, b = block_assemble(a, L, bcs, symmetric=True)

    return A, b


def stokes_diagonal_preconditioner(A, W):
    u, p = map(df.TrialFunction, W)
    v, q = map(df.TestFunction, W)

    M = df.assemble(df.inner(p,q)*df.dx)
    Minv = LumpedInvDiag(M)
    Pinv = block_mat([[LU(A[0][0]), 0], [0, Minv]])

    return Pinv


def diagonal_preconditioned_minres(A, W):
    Pinv = stokes_diagonal_preconditioner(A, W) 
    Ainv = MinRes(A, precond=Pinv, show=0,tolerance=1e-15)
    return Ainv


def create_A_hat_inv(A):
    return block_mat(1, 1, [[LU(A[0,0])]])


def create_S_hat_inv_mass(V, A, omega):
    u,v = df.TrialFunction(V), df.TestFunction(V)
    M = df.assemble(u*v*df.dx)
    S_hat_inv = 1/omega*LumpedInvDiag(M)
    return S_hat_inv


def create_S_hat_inv_exact(V, A, omega):
    C = -A[1,1]
    BT = A[0,1]
    B = A[1,0]
    iA0 = LU(A[0,0])

    S = C + B * iA0 * BT
    S_prec = create_S_hat_inv_mass(V, A, omega=1.) 
    iS = ConjGrad(S, precond=S_prec, show=1, tolerance=1e-16)

    S_hat_inv = (1./omega) * iS

    return S_hat_inv


def zero_mean(mesh, p):
    V = df.FunctionSpace(mesh, 'P', 1)
    pp = df.Function(V)
    pp.vector()[:] = p[:]
    p[:] -= df.assemble(pp*df.dx)/df.assemble(1*df.dx(domain=mesh))


def run_demo():
    # problem properties
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--smoothing-steps', type=int, default=3)
    parser.add_argument('--w-cycles', type=int, default=2)
    parser.add_argument('--exact-schur-complement', action='store_true')
    parser.add_argument('--stabilization', action='store_true')
    parser.add_argument('--degree-velocity', type=int, default=2)
    parser.add_argument('--num-levels', type=int, default=6)
    parser.add_argument('--smoother-type', type=str, choices=('rectangular', 'symmetric'), default='rectangular')
    args = parser.parse_args()

    presmoothing_steps = postsmoothing_steps = args.smoothing_steps
    w_cycles= args.w_cycles
    stab = args.stabilization 
    deg_velocity = args.degree_velocity 
    num_levels = args.num_levels 

    # create the coarse mesh:
    N = 2
    mesh_coarse = df.RectangleMesh(df.Point(-4,-1), df.Point(4,1), 4*N, N, 'left')

    # setup a mesh hierarchy
    meshes = create_mesh_hierarchy(mesh_coarse, num_levels)
    # create spaces
    V_u = list(map(lambda m: df.VectorFunctionSpace(m, 'P', deg_velocity), meshes))
    V_p = list(map(lambda m: df.FunctionSpace(m, 'P', 1), meshes))
    W = list(zip(V_u, V_p))

    # inlets 
    noslip_domain = df.CompiledSubDomain('x[1] >= 1-1e-8 || x[1] <= -1+1e-8')
    inflow_domain = df.CompiledSubDomain('x[0] <= -4+1e-8')

    # boundary values
    noslip_value = [df.Constant((0, 0)) for _ in range(num_levels)]
    inflow_value = [df.Expression(('-(x[1]+1)*(x[1]-1)', 0), degree=2)] + [df.Constant((0,0)) for _ in range(1,num_levels)]

    # boundary conditions:
    bc_vel_noslip = list(map(lambda args: df.DirichletBC(args[0], args[1], noslip_domain), list(zip(V_u, noslip_value))))
    bc_vel_inflow = list(map(lambda args: df.DirichletBC(args[0], args[1], inflow_domain), list(zip(V_u, inflow_value))))
    bc_pre = [[] for _ in range(num_levels)]

    bcs = list(zip(list(zip(bc_vel_noslip, bc_vel_inflow)), bc_pre))

    # true solution
    solution = df.Function(V_u[0])
    solution.interpolate(inflow_value[0])

    # prolongation operators
    P = create_prolongation_hierarchy(W) 

    # assembled systems
    assembly_results = [assemble_system(W_, bc, stab=stab) for W_, bc in zip(W, bcs)]
    A = [A for A, b in assembly_results]
    b = [b for A, b in assembly_results]

    # coarse grid solver
    Ainv_coarse = diagonal_preconditioned_minres(A[-1], W[-1])

    # create approximation of A
    A_hat_inv = [create_A_hat_inv(A_) for A_ in A]
    # estimate omega for Uzawa Smoother
    if args.exact_schur_complement:
        factory_S_hat = create_S_hat_inv_exact
    else:
        factory_S_hat = create_S_hat_inv_mass
    S_hat_inv = factory_S_hat(W[0][-1], A[0], omega=1.)
    omega = estimate_omega(A_hat_inv[0], S_hat_inv, A[0])
    # create approximation of S 
    S_hat_inv = [factory_S_hat(W_[-1], A_, omega=omega) for W_, A_ in zip(W,A)]

    # create smoothers 
    if args.smoother_type == 'symmetric':
        presmoother = [SmootherSymmetric(A_, A_hat_inv_, A_hat_inv_, S_hat_inv_) for A_, A_hat_inv_, S_hat_inv_ in zip(A, A_hat_inv, S_hat_inv)]
        postsmoother = [SmootherSymmetric(A_, A_hat_inv_, A_hat_inv_, S_hat_inv_) for A_, A_hat_inv_, S_hat_inv_ in zip(A, A_hat_inv, S_hat_inv)]
    elif args.smoother_type == 'rectangular':
        presmoother = [SmootherLower(A_, A_hat_inv_, S_hat_inv_) for A_, A_hat_inv_, S_hat_inv_ in zip(A, A_hat_inv, S_hat_inv)]
        postsmoother = [SmootherUpper(A_, A_hat_inv_, S_hat_inv_) for A_, A_hat_inv_, S_hat_inv_ in zip(A, A_hat_inv, S_hat_inv)]
    else:
        print('unknown smoother')
        exit(-1)

    # setup full solver
    solver = MGSolver(
        presmoother=presmoother,
        postsmoother=postsmoother,
        coarse_grid_solver=Ainv_coarse,
        A=A,
        P=P
    )
    solver.num_presmoothing_steps = presmoothing_steps
    solver.num_postsmoothing_steps = postsmoothing_steps
    solver.num_w_cycles = w_cycles
    solver.projection_nullspace = lambda x: zero_mean(meshes[0], x[-1])

    x = A[0].create_vec()
    x.randomize()

    x = solver.solve(b[0], x)

    # write vectorial solution into functions
    u = df.Function(V_u[0], name='v')
    u.vector()[:] = x[0][:]
    p = df.Function(V_p[0], name='pressure')
    p.vector()[:] = x[1][:]

    # output to files
    file_u = df.File('output/stokes_u.pvd')
    file_u << u, 0
    file_p = df.File('output/stokes_p.pvd')
    file_p << p, 0


if __name__ == '__main__':
    run_demo()
