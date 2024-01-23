import argparse
import dolfin as df
from block import block_assemble, block_mat, block_vec, iterative
from block.block_base import block_base
from block.iterative import MinRes, ConjGrad, Richardson, LGMRES
from block.algebraic.petsc import LU, LumpedInvDiag, SOR, InvDiag, Jacobi
from block.algebraic.petsc.precond import precond 
import petsc4py as p4py

from pyuzawamg.solvers import (
    SmootherLower, 
    SmootherUpper, 
    SmootherSymmetric,
    MGSolver, 
    create_mesh_hierarchy,
    create_prolongation_hierarchy,
    estimate_omega
)
from pyuzawamg.util import (
    CBCBlockWrapper,
    power_iteration
)


def assemble_system(W, bcs, stab=False, f=None):
    if f is None:
        f = df.Constant((0,0))

    u, p = map(df.TrialFunction, W)
    v, q = map(df.TestFunction, W)

    a = [[0,0],[0,0]]
    a[0][0] = df.inner(df.grad(u), df.grad(v))*df.dx
    a[0][1] = -df.div(v) * p * df.dx
    #a[0][1] = +df.inner(df.grad(p), v) * df.dx
    a[1][0] = -df.div(u) * q * df.dx
    #a[1][0] = +df.inner(df.grad(q), u) * df.dx

    if stab:
        h = df.CellDiameter(W[0].mesh())
        beta  = 0.2
        delta = beta*h*h
        a[1][1] = - delta*df.inner(df.grad(q), df.grad(p))*df.dx

    L = [0,0]
    L[0] = df.inner(f, v) * df.dx
    if stab:
        L[1] = -delta*df.inner(df.grad(q), f) * df.dx

    #A, _, b = block_assemble(a, L, bcs, symmetric=True)
    A, b = block_assemble(a, L, bcs, symmetric=False)
    A[0,1] = A[0,1] * MeanZeroProjection()
    A[1,0] = MeanZeroProjection() * A[1,0] 

    return A, b


class MeanZeroProjection(block_base):
    """Base class for diagonal PETSc operators (represented by an PETSc vector)."""
    from block.object_pool import shared_vec_pool

    def __init__(self):
        pass

    def _transpose(self):
        return self

    def matvec(self, b):
        x = b.copy()
        zero_mean(None, x)
        return x

class WrappedMeanZeroProjection(block_base):
    from block.object_pool import shared_vec_pool

    def __init__(self, A):
        self._A = A

    def _transpose(self):
        return self

    def matvec(self, b):
        b = b.copy()
        zero_mean(None, b)
        x = self._A * b 
        zero_mean(None, x)
        return x

    @shared_vec_pool
    def create_vec(self, dim=1):
        from dolfin import PETScVector
        if dim > 1:
            raise ValueError('dim must be <= 1')
        return self._A.create_vec()


def stokes_diagonal_preconditioner(A, W):
    u, p = map(df.TrialFunction, W)
    v, q = map(df.TestFunction, W)

    b = df.assemble(q*df.dx)

    M = df.assemble(df.inner(p,q)*df.dx)
    Minv = LumpedInvDiag(M)
    Pinv = block_mat([[LU(A[0][0]), 0], [0, WrappedMeanZeroProjection(Minv)]])

    return Pinv


def diagonal_preconditioned_minres(A, W):
    Pinv = stokes_diagonal_preconditioner(A, W) 
    Ainv = MinRes(A, precond=Pinv, show=0,tolerance=1e-15, maxiter=1000)
    return Ainv


def create_A_hat_inv(A):
    return block_mat(1, 1, [[LU(A[0,0])]])


def create_S_hat_inv_mass(V, A, omega):
    u,v = df.TrialFunction(V), df.TestFunction(V)
    M = df.assemble(u*v*df.dx)
    S_hat_inv = (1./omega)*LumpedInvDiag(M)
    return WrappedMeanZeroProjection(S_hat_inv)


def create_S_hat_inv_exact(V, A, omega):
    C = -A[1,1]
    BT = A[0,1]
    B = A[1,0]
    iA0 = LU(A[0,0])

    S = C + B * iA0 * BT
    S_prec = create_S_hat_inv_mass(V, A, omega=1.) 
    iS = ConjGrad(S, precond=S_prec, show=1, tolerance=1e-16)

    S_hat_inv = (1./omega) * iS

    return WrappedMeanZeroProjection(S_hat_inv)


def zero_mean(mesh, p):
    p = df.as_backend_type(p).vec()
    p[:] -= p.sum() / p.size


def create_gs_f(A, maxit):
    return Richardson(precond=block_mat([[SOR(A[0,0], parameters={"petsc_sor_forward": 1, "ksp_monitor": 1})], ]), A=block_mat([[A[0,0]]]), maxiter=maxit, tolerance=1e-16, show=0)

def create_gs_b(A, maxit):
    return Richardson(precond=block_mat([[SOR(A[0,0], parameters={"petsc_sor_backward": 1, "ksp_monitor": 1})], ]), A=block_mat([[A[0,0]]]), maxiter=maxit, tolerance=1e-16, show=0)

def create_jacobi(A, maxit):
    #return SGS(A, maxit)
    #return block_mat([[SOR(A[0,0], parameters={"petsc_sor_symmetric": 1, "ksp_monitor": 1})], ])
    #return block_mat([[SOR(A[0,0], parameters={"petsc_sor_forward": 1, "ksp_monitor": 1})], ])
    #return block_mat([[LU(A[0,0])], ])
    #return block_mat([[0.25*InvDiag(A[0,0])], ])
    #return block_mat([[0.00001*InvDiag(A[0,0])], ])
    #return block_mat([[0.5*InvDiag(A[0,0])]])
    #return block_mat([[0.5*InvDiag(A[0,0])]])
    return block_mat([[0.5*InvDiag(A[0,0])]])
    #return Richardson(precond=block_mat([[0.1 * InvDiag(A[0,0])]]), A=block_mat([[A[0,0]]]), maxiter=maxit, tolerance=1e-16, show=1)
    #return block_mat([[0.5*Jacobi(A=A[0,0])]])
    #return block_mat([[Jacobi(A=A[0,0])]])
    #omega = 1
    #AA = block_mat([[ omega*InvDiag(A[0,0]) ]]) 
    #print(AA)
    #return block_mat([[1]])
    #return block_mat([[0]])
    #return Richardson(A=block_mat([[A[0,0]]]), maxiter=maxit, tolerance=1e-16, show=1)
    return block_mat([[Jacobi(A=A[0,0],prefix='j', parameters={"jksp_monitor": 1})]])

class Chebyshev(precond):
    """
    Actually this is only a diagonal scaling preconditioner; no support for relaxation or multiple iterations.
    """
    def __init__(self, A, parameters=None, pdes=1, nullspace=None, prefix=None):
        super().__init__(A, p4py.PETSc.PC.Type.Chebyshev,
                         pdes=pdes, nullspace=nullspace, options=parameters, prefix=prefix,
                         defaults={
                             #"pc_jacobi_rowmax": "",  # Use row maximums for diagonal
                             #"pc_jacobi_rowsum": "",  # Use row sums for diagonal
                             #"pc_jacobi_abs":, "",    # Use absolute values of diagaonal entries
                         })

def create_cheby(A, maxit):
    return Richardson(precond=block_mat([[SOR(A[0,0], parameters={"petsc_sor_backward": 1, "ksp_monitor": 1})], ]), A=block_mat([[A[0,0]]]), maxiter=maxit, tolerance=1e-16, show=0)

    pass


class SGS(block_base):
    def __init__(self, A, maxit):
        self._A = A
        self._maxit = maxit
        self.gs_f = create_gs_f(A, maxit) 
        self.gs_b = create_gs_b(A, maxit) 
    
    def matvec(self, b):
        x = self.gs_f * b
        return self.gs_b(initial_guess=x) * b
    

def create_sgs(A, maxit):
    return SGS(A, maxit)


def run_demo():
    # problem properties
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--smoothing-steps', type=int, default=3)
    parser.add_argument('--gs-steps', type=int, default=1)
    parser.add_argument('--w-cycles', type=int, default=2)
    parser.add_argument('--exact-schur-complement', action='store_true')
    parser.add_argument('--stabilization', action='store_true')
    parser.add_argument('--degree-velocity', type=int, default=2)
    parser.add_argument('--num-levels', type=int, default=4)
    parser.add_argument('--smoother-type', type=str, choices=('rectangular', 'symmetric'), default='rectangular')
    args = parser.parse_args()

    presmoothing_steps = postsmoothing_steps = args.smoothing_steps
    w_cycles= args.w_cycles
    stab = args.stabilization 
    deg_velocity = args.degree_velocity 
    num_levels = args.num_levels 

    maxit_gs = args.gs_steps

    # create the coarse mesh:
    mesh_coarse = df.Mesh()
    file = df.XDMFFile('AnnulusShell.xdmf')
    file.read(mesh_coarse)

    # setup a mesh hierarchy
    meshes = create_mesh_hierarchy(mesh_coarse, num_levels)
    # create spaces
    V_u = list(map(lambda m: df.VectorFunctionSpace(m, 'P', deg_velocity), meshes))
    V_p = list(map(lambda m: df.FunctionSpace(m, 'P', 1), meshes))
    W = list(zip(V_u, V_p))

    print(W[0][0].dim(), W[0][1].dim(), W[0][0].dim()+W[0][1].dim())

    # inlets 
    noslip_domain = df.CompiledSubDomain('false')
    inflow_domain = df.CompiledSubDomain('on_boundary')


    # boundary values
    noslip_value = [df.Constant((0, 0)) for _ in range(num_levels)]
    inflow_value = [df.Expression((
         'x[1]+std::sin( 3*M_PI*x[1]) +std::sin( M_PI*( x[0]+x[1]) ) +std::cos( 8*M_PI*x[1])',
         '-x[0]+std::sin( 4*M_PI*x[0]) -std::sin( M_PI*( x[0]+x[1]) ) +std::cos( 2*M_PI*x[0])'), degree=2)] + [df.Constant((0,0)) for _ in range(1,num_levels)]

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

    f = df.Expression((
        ''' 2*x[0]+9*std::pow( M_PI, 2) *std::sin( 3*M_PI*x[1]) +
            3*M_PI*std::sin( 4*M_PI*x[1]) *std::cos( 3*M_PI*x[0]) +
            2*std::pow( M_PI, 2) *std::sin( M_PI*( x[0]+x[1]) ) +
            64*std::pow( M_PI, 2) *std::cos( 8*M_PI*x[1]) +2
        ''',
        ''' 2*x[1]+4*M_PI*std::sin( 3*M_PI*x[0]) *std::cos( 4*M_PI*x[1]) +
            16*std::pow( M_PI, 2) *std::sin( 4*M_PI*x[0]) -
            2*std::pow( M_PI, 2) *std::sin( M_PI*( x[0]+x[1]) ) +
            4*std::pow( M_PI, 2) *std::cos( 2*M_PI*x[0]) -1
        '''), degree=2)

    # assembled systems
    assembly_results = [assemble_system(W[0], bcs[0], stab=stab, f=f)] + [assemble_system(W_, bc, stab=stab) for idx, (W_, bc) in enumerate(zip(W, bcs)) if idx != 0]
    A = [A for A, b in assembly_results]
    b = [b for A, b in assembly_results]
    zero_mean(meshes[0], b[0][-1])

    '''
    x = block_vec(1, [A[0].create_vec()[0]])
    x.randomize()
    for _ in range(100):
        #x_next = SGS(A[0],maxit=1) * block_mat([[A[0][0,0]]]) * x
        x_next = create_gs_f(A[0], maxit=1) * block_mat([[A[0][0,0]]]) * x
        # rayleigh quotient
        alpha = x_next.inner(x)
        x_next_norm = x_next.norm()
        x = (1./x_next_norm) * x_next 
        print('!!', alpha)
    '''

    # coarse grid solver
    Ainv_coarse = diagonal_preconditioned_minres(A[-1], W[-1])

    # create approximation of A
    #A_hat_inv = [create_gs_b(A_, maxit_gs) for A_ in A]
    #A_hat_inv = [create_sgs(A_, maxit_gs) for A_ in A]
    #A_hat_inv = [create_jacobi(A_, maxit_gs) for A_ in A]
    A_hat_inv = [block_mat([[LU(A_[0,0])]]) for A_ in A]
    # estimate omega for Uzawa Smoother
    if args.exact_schur_complement:
        factory_S_hat = create_S_hat_inv_exact
    else:
        factory_S_hat = create_S_hat_inv_mass
    S_hat_inv = factory_S_hat(W[0][-1], A[0], omega=1.)
    #omega = estimate_omega(A_hat_inv[0], S_hat_inv, A[0])
    omega = 1. 
    #omega = 1./0.3
    #omega = 0.2
    print(f'omega {omega}')
    # create approximation of S 
    S_hat_inv = [factory_S_hat(W_[-1], A_, omega=omega) for W_, A_ in zip(W,A)]

    # create smoothers 
    if args.smoother_type == 'symmetric':
        presmoother = [SmootherSymmetric(A_, A_hat_inv_, A_hat_inv_, S_hat_inv_) for A_, A_hat_inv_, S_hat_inv_ in zip(A, A_hat_inv, S_hat_inv)]
        postsmoother = [SmootherSymmetric(A_, A_hat_inv_, A_hat_inv_, S_hat_inv_) for A_, A_hat_inv_, S_hat_inv_ in zip(A, A_hat_inv, S_hat_inv)]
    elif args.smoother_type == 'rectangular':
        #A_hat_inv_pre = [create_gs_f(A_, maxit_gs) for A_ in A]
        #A_hat_inv_post = [create_gs_f(A_, maxit_gs) for A_ in A]
        #A_hat_inv_pre = [create_sgs(A_, maxit_gs) for A_ in A]
        #A_hat_inv_post = [create_sgs(A_, maxit_gs) for A_ in A]
        A_hat_inv_pre = [create_jacobi(A_, maxit_gs) for A_ in A]
        A_hat_inv_post = [create_jacobi(A_, maxit_gs) for A_ in A]
        presmoother = [SmootherLower(A_, A_hat_inv_, S_hat_inv_) for A_, A_hat_inv_, S_hat_inv_ in zip(A, A_hat_inv_pre, S_hat_inv)]
        #postsmoother = [SmootherUpper(A_, A_hat_inv_, S_hat_inv_) for A_, A_hat_inv_, S_hat_inv_ in zip(A, A_hat_inv_post, S_hat_inv)]
        postsmoother = [SmootherLower(A_, A_hat_inv_, S_hat_inv_) for A_, A_hat_inv_, S_hat_inv_ in zip(A, A_hat_inv_post, S_hat_inv)]
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
    solver.projection_nullspace_coarse = lambda x: zero_mean(meshes[-1], x[-1])
    solver.num_iterations = 100

    x = A[0].create_vec()
    x.randomize()
    df.DirichletBC(V_u[0], df.Constant((0,0)), inflow_domain).apply(x[0])
    df.DirichletBC(V_u[0], inflow_value[0], inflow_domain).apply(x[0])
    print(x[0][:])

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
    #file_u = df.File('output/stokes_u_sol.pvd')
    #file_u << df.project(inflow_value, V_u), 0


if __name__ == '__main__':
    run_demo()
