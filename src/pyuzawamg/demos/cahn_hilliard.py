import argparse
import random
import dolfin as df
from block import block_assemble, block_mat
from block.iterative import MinRes, ConjGrad
from block.algebraic.petsc import LU, LumpedInvDiag

from ..solvers import (
    SmootherLower, 
    SmootherUpper, 
    MGSolver, 
    create_mesh_hierarchy,
    create_prolongation_hierarchy,
    estimate_omega
)


def assemble_system(W, eps, tau, f=None):
    if f is None:
        f = df.Expression('sin(2 * M_PI * x[0]) * cos(4 * M_PI * x[1])', degree=2)

    u, p = map(df.TrialFunction, W)
    v, q = map(df.TestFunction, W)

    a = [[0,0],[0,0]]
    a[0][0] = df.inner(u, v)*df.dx + eps*df.inner(df.grad(u), df.grad(v))*df.dx
    a[0][1] = -df.inner(v, p) * df.dx
    a[1][0] = -df.inner(u, q) * df.dx
    a[1][1] = - tau * df.inner(u, v)*df.dx

    L = [0,0]
    L[0] = df.inner(f, v) * df.dx
    L[1] = df.inner(f, v) * df.dx

    A, b = block_assemble(a, L, symmetric=True)

    return A, b


def diagonal_preconditioner(A, W, tau):
    u, p = map(df.TrialFunction, W)
    v, q = map(df.TestFunction, W)

    S = df.assemble(df.inner(p,q)*df.dx + tau * df.inner(p, q)*df.dx )

    Pinv = block_mat([[LU(A[0][0]), 0], [0, LU(S)]])

    return Pinv


def diagonal_preconditioned_minres(A, W, tau):
    Pinv = diagonal_preconditioner(A, W, tau) 
    Ainv = MinRes(A, precond=Pinv, show=0,tolerance=1e-15)
    return Ainv


def create_A_hat_inv(A):
    return block_mat(1, 1, [[LU(A[0,0])]])


def create_S_hat_inv_mass(V, A, tau, omega):
    u,v = df.TrialFunction(V), df.TestFunction(V)
    S1 = df.assemble(df.inner(u,v)*df.dx + tau * df.inner(u, v)*df.dx )
    S_hat_inv = 1/omega*LU(S1)
    return S_hat_inv


def create_S_hat_inv_exact(V, A, tau, omega):
    C = -A[1,1]
    BT = A[0,1]
    B = A[1,0]
    iA0 = LU(A[0,0])

    S = C + B * iA0 * BT
    S_prec = create_S_hat_inv_mass(V, A, tau, omega=1.) 
    iS = ConjGrad(S, precond=S_prec, show=1, tolerance=1e-16)

    S_hat_inv = (1./omega) * iS

    return S_hat_inv


class RandomInitialConditions(df.UserExpression):
    """ Random initial conditions """
    def __init__(self, **kwargs):
        random.seed(2 + df.MPI.rank(df.MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 0.5 + 0.5*(0.5 - random.random())


def ExpressionSmoothCircle(midpoint: df.Point, r: float, a: float, degree: int):
    r_str = f'std::sqrt(std::pow(x[0]-({midpoint[0]}),2) + std::pow(x[1]-({midpoint[1]}),2))'
    return df.Expression(f'1. / (exp( a * ({r_str}-r) ) + 1)', r=r, a=a, degree=degree)


def mobility_variable(phi: df.Function):
    mobility = 8 * phi**2 * (1-phi)**2
    return mobility


def mobility_constant(phi: df.Function):
    return df.Constant(1.)


def run_demo():
    # problem properties
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--smoothing-steps', type=int, default=3)
    parser.add_argument('--w-cycles', type=int, default=2)
    parser.add_argument('--exact-schur-complement', action='store_true')
    parser.add_argument('--num-levels', type=int, default=6)
    parser.add_argument('--eps', type=float, default=1.)
    parser.add_argument('--tau', type=float, default=1.)
    args = parser.parse_args()

    tau = args.tau
    eps = args.eps

    presmoothing_steps = postsmoothing_steps = args.smoothing_steps
    w_cycles= args.w_cycles
    num_levels = args.num_levels 

    # create the coarse mesh:
    N = 2
    mesh_coarse = df.RectangleMesh(df.Point(-1,-1), df.Point(1,1), N, N, 'left')

    # setup a mesh hierarchy
    meshes = create_mesh_hierarchy(mesh_coarse, num_levels)
    # create spaces
    V_u = list(map(lambda m: df.FunctionSpace(m, 'P', 1), meshes))
    V_p = list(map(lambda m: df.FunctionSpace(m, 'P', 1), meshes))
    W = list(zip(V_u, V_p))

    # true solution
    solution = df.Function(V_u[0])
    solution.interpolate(RandomInitialConditions())

    # prolongation operators
    P = create_prolongation_hierarchy(W) 

    # assembled systems
    assembly_results = [assemble_system(W_, eps, tau) for W_ in W]
    A = [A for A, b in assembly_results]
    b = [b for A, b in assembly_results]
    print(b[0])

    # coarse grid solver
    Ainv_coarse = diagonal_preconditioned_minres(A[-1], W[-1], tau)

    # create approximation of A
    A_hat_inv = [create_A_hat_inv(A_) for A_ in A]
    # estimate omega for Uzawa Smoother
    if args.exact_schur_complement:
        factory_S_hat = create_S_hat_inv_exact
    else:
        factory_S_hat = create_S_hat_inv_mass
    S_hat_inv = factory_S_hat(W[0][-1], A[0], tau, omega=1.)
    omega = estimate_omega(A_hat_inv[0], S_hat_inv, A[0])
    # create approximation of S 
    S_hat_inv = [factory_S_hat(W_[-1], A_, tau, omega=omega) for W_, A_ in zip(W,A)]

    # create smoothers 
    presmoother = [SmootherLower(A_, A_hat_inv_, S_hat_inv_) for A_, A_hat_inv_, S_hat_inv_ in zip(A, A_hat_inv, S_hat_inv)]
    postsmoother = [SmootherUpper(A_, A_hat_inv_, S_hat_inv_) for A_, A_hat_inv_, S_hat_inv_ in zip(A, A_hat_inv, S_hat_inv)]

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

    x = A[0].create_vec()
    x.randomize()

    x = solver.solve(b[0], x)


if __name__ == '__main__':
    run_demo()
