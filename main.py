import dolfin as df
import numpy as np
from block import block_assemble, block_mat, block_vec
from block.iterative import MinRes
from block.algebraic.petsc import LU, LumpedInvDiag

from prolongation import block_prolongation 


def assemble_system(W, bcs, stab=False, f=None):
    if f is None:
        f = [df.Constant(0), df.Constant(0)]


    ux, uy, p = map(df.TrialFunction, W)
    vx, vy, q = map(df.TestFunction, W)

    a = [[0,0,0],[0,0,0],[0,0,0]]
    a[0][0] = df.inner(df.grad(ux), df.grad(vx))*df.dx
    a[1][1] = df.inner(df.grad(uy), df.grad(vy))*df.dx
    a[0][2] = -vx.dx(0) * p * df.dx
    a[1][2] = -vy.dx(1) * p * df.dx
    a[2][0] = -ux.dx(0) * q * df.dx
    a[2][1] = -uy.dx(1) * q * df.dx


    if stab:
        h = df.CellDiameter(W[0].mesh())
        beta  = 0.2
        delta = beta*h*h
        a[2][2] = - delta*df.inner(df.grad(q), df.grad(p))*df.dx

    L = [0,0,0]
    L[0] = df.inner(f[0], vx) * df.dx
    L[1] = df.inner(f[0], vy) * df.dx

    if stab:
        L[2] = sum([-delta*df.inner(q.dx(0), f[0]) * df.dx for i in range(2)])

    A, _, b = block_assemble(a, L, bcs, symmetric=True)

    return A, b


def stokes_diagonal_preconditioner(A, W, omega=1):
    ux, uy, p = map(df.TrialFunction, W)
    vx, vy, q = map(df.TestFunction, W)

    M = df.assemble(df.inner(p,q)*df.dx)
    Minv = omega*LumpedInvDiag(M)
    Pinv = block_mat([
        [LU(A[0][0]), 0, 0], 
        [0, LU(A[1][1]), 0], 
        [0, 0, Minv]
    ])

    return Pinv


def diagonal_preconditioned_minres(A, W):
    Pinv = stokes_diagonal_preconditioner(A, W) 
    Ainv = MinRes(A, precond=Pinv, show=0, tolerance=1e-16)
    return Ainv


def zero_mean(mesh, p):
    V = df.FunctionSpace(mesh, 'P', 1)
    pp = df.Function(V)
    pp.vector()[:] = p[:]
    p[:] -= df.assemble(pp*df.dx)/df.assemble(1*df.dx(domain=mesh))


class SmootherLower:
    def __init__(self, A,  A_hat_inv, S_hat_inv):
        self.A = A
        self.A_hat_inv = A_hat_inv
        self.S_hat_inv = S_hat_inv
    
    def mult(self, b, x):
        A, A_hat_inv, S_hat_inv = self.A, self.A_hat_inv, self.S_hat_inv
        n,m = A.blocks.shape
        assert n == m, "only symmetric matrices supported"
        # first row:
        r0 = block_vec(b[0:n-1] - A[0:n-1,0:n-1] @ x[0:n-1] - A[0:n-1,n-1:n] @ x[n-1:n])
        x[0:n-1] = x[0:n-1] + (A_hat_inv * r0).blocks
        # second row:
        r1 = b[n-1] - A[n-1,0:n-1] @ x[0:n-1] - A[n-1,n-1] * x[n-1]
        x[n-1] = x[n-1] - S_hat_inv * r1


class SmootherUpper:
    def __init__(self, A,  A_hat_inv, S_hat_inv):
        self.A = A
        self.A_hat_inv = A_hat_inv
        self.S_hat_inv = S_hat_inv
    
    def mult(self, b, x):
        A, A_hat_inv, S_hat_inv = self.A, self.A_hat_inv, self.S_hat_inv
        n,m = A.blocks.shape
        assert n == m, "only symmetric matrices supported"
        # second row:
        r1 = b[n-1] - A[n-1,0:n-1] @ x[0:n-1] - A[n-1,n-1] * x[n-1]
        x[n-1] = x[n-1] - S_hat_inv * r1
        # first row:
        r0 = block_vec(b[0:n-1] - A[0:n-1,0:n-1] @ x[0:n-1] - A[0:n-1,n-1:n] @ x[n-1:n])
        x[0:n-1] = x[0:n-1] + (A_hat_inv * r0).blocks


def create_A_hat_inv(A):
    return block_mat([
        [LU(A[0,0]), 0],
        [0, LU(A[1,1])],
    ])


def create_S_hat_inv(V, omega):
    u,v = df.TrialFunction(V), df.TestFunction(V)
    M = df.assemble(u*v*df.dx)
    S_hat_inv = 1/omega*LumpedInvDiag(M)
    return S_hat_inv


def estimate_omega(A_hat_inv, S_hat_inv, A, num_iterations=10):
    n,m = A.blocks.shape
    assert n == m, "only symmetric matrices supported"
    # extract the operator
    S_hat_inv = block_mat(1, 1, [[S_hat_inv]])
    C = block_mat(1, 1, [[-A[n-1,n-1]]])
    BT = block_mat(A[0:n-1,n-1:n])
    B = block_mat(A[n-1:n,0:n-1])
    # apply our problem
    def op(x):
        y = (B * A_hat_inv * BT * x)
        if not C[0,0] == 0:
            y += C * x
        return S_hat_inv * y
    # We create a block vector for our power iteration. 
    # Note that 
    #   x = C.create_vec()
    # would not work if C = 0.
    x = block_vec(1, [A.create_vec()[-1]])
    x.randomize()
    for _ in range(num_iterations):
        x_next = op(x) 
        # rayleigh quotient
        alpha = x_next.inner(x)
        x_next_norm = x_next.norm()
        x = (1./x_next_norm) * x_next 
    return alpha


class MGSolver:
    def __init__(self, presmoother, postsmoother, coarse_grid_solver, A, P) -> None:
        self.presmoother = presmoother 
        self.postsmoother = postsmoother 
        self.coarse_grid_solver = coarse_grid_solver
        self.A = A 
        self.P = P 
        self.num_iterations = 10
        self.num_presmoothing_steps = 3 
        self.num_postsmoothing_steps = 3 
        self.num_w_cycles = 2
        self.projection_nullspace = lambda x: 1 
        self.show = 1
    
    def solve(self, b, x=None):
        # provide initial guess
        if x is None:
            x = self.A[0].create_vec()
        else:
            x = x.copy()
        # calculate initial residual
        r = b - self.A[0] * x
        res_prev = r.norm()
        # iterate mg solver
        for j in range(self.num_iterations):
            x = self._solve(x, b, 0)
            # calculate residual:
            r = b - self.A[0] * x
            self.projection_nullspace(r)
            res = r.norm()
            if self.show > 0:
                print(f'{j} - rate = {res / res_prev} ({res})')
            res_prev = res
        return x
    
    @property
    def num_levels(self):
        return len(self.A)

    def _solve(self, x, b_fine, k):
        x = x.copy()
        # solve on finest level
        if k + 1 == self.num_levels:
            return self.coarse_grid_solver * b_fine
        # extract local solver components:
        A_fine = self.A[k]
        A_coarse = self.A[k+1]
        presmoother = self.presmoother[k]
        postsmoother = self.postsmoother[k]
        P = self.P[k]
        # presmoothing steps
        for _ in range(self.num_presmoothing_steps):
            presmoother.mult(b_fine, x)
        # coarse grid correction
        for _ in range(self.num_w_cycles):
            r_fine = b_fine - A_fine * x
            r_coarse = A_coarse.create_vec()
            d_coarse = A_coarse.create_vec()
            P.tmult(r_fine, r_coarse) 
            d_coarse = self._solve(d_coarse, r_coarse, k+1)
            d_fine = A_fine.create_vec()
            P.mult(d_coarse, d_fine) 
            x += d_fine
        # postmoothing steps
        for _ in range(self.num_postsmoothing_steps):
            postsmoother.mult(b_fine, x)
        return x


def create_mesh_hierarchy(coarse_mesh, num_levels):
    meshes = [coarse_mesh]
    for i in range(num_levels-1):
        meshes.append(df.refine(meshes[-1]))
    return list(reversed(meshes))


def create_prolongation_hierarchy(W):
    prolongations = []
    for k in range(len(W)-1):
        W_fine = W[k]
        W_coarse = W[k+1]
        P = block_prolongation(W_coarse, W_fine)
        prolongations.append(P)
    return prolongations


def demo2():
    presmoothing_steps = postsmoothing_steps = 3 * 1 
    w_cycles=2
    stab = True 
    deg_velocity = 1
    num_levels = 6

    N = 2
    mesh_coarse = df.RectangleMesh(df.Point(-4,-1), df.Point(4,1), 4*N, N, 'left')

    meshes = create_mesh_hierarchy(mesh_coarse, num_levels)
    V_u = list(map(lambda m: df.FunctionSpace(m, 'P', deg_velocity), meshes))
    V_p = list(map(lambda m: df.FunctionSpace(m, 'P', 1), meshes))
    W = list(zip(V_u, V_u, V_p))

    noslip_domain = df.CompiledSubDomain('x[1] >= 1-1e-8 || x[1] <= -1+1e-8')
    inflow_domain = df.CompiledSubDomain('x[0] <= -4+1e-8')

    noslip_value = [df.Constant(0) for _ in range(num_levels)]
    inflow_valuex = [df.Expression('-(x[1]+1)*(x[1]-1)', degree=2)] + [df.Constant(0) for _ in range(1,num_levels)]
    inflow_valuey = [df.Constant(0) for _ in range(num_levels)] 

    bc0x = list(map(lambda args: df.DirichletBC(args[0], args[1], noslip_domain), list(zip(V_u,noslip_value))))
    bc0y = list(map(lambda args: df.DirichletBC(args[0], args[1], noslip_domain), list(zip(V_u,noslip_value))))

    bc1x = list(map(lambda args: df.DirichletBC(args[0], args[1], inflow_domain), list(zip(V_u,inflow_valuex))))
    bc1y = list(map(lambda args: df.DirichletBC(args[0], args[1], inflow_domain), list(zip(V_u,inflow_valuey))))

    bcp = [[] for _ in range(num_levels)]

    bcs = list(zip(list(zip(bc0x, bc1x)), list(zip(bc0y, bc1y)), bcp))

    solution = df.Function(V_u[0])
    solution.interpolate(inflow_valuex[0])

    P = create_prolongation_hierarchy(W) 
    print(len(W))
    print(len(P))

    assembly_results = [assemble_system(W_, bc, stab=stab) for W_, bc in zip(W, bcs)]
    A = [A for A, b in assembly_results]
    b = [b for A, b in assembly_results]

    Ainv_coarse = diagonal_preconditioned_minres(A[-1], W[-1])

    A_hat_inv = [create_A_hat_inv(A_) for A_ in A]
    S_hat_inv = create_S_hat_inv(W[0][-1], 1.)
    omega = estimate_omega(A_hat_inv[0], S_hat_inv, A[0])
    omega *= 1.
    S_hat_inv = [create_S_hat_inv(W_[-1], omega) for W_ in W]

    presmoother = [SmootherLower(A_, A_hat_inv_, S_hat_inv_) for A_, A_hat_inv_, S_hat_inv_ in zip(A, A_hat_inv, S_hat_inv)]
    postsmoother = [SmootherUpper(A_, A_hat_inv_, S_hat_inv_) for A_, A_hat_inv_, S_hat_inv_ in zip(A, A_hat_inv, S_hat_inv)]

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

    ux = df.Function(V_u[0], name='vx')
    uy = df.Function(V_u[0], name='vy')
    ux.vector()[:] = x[0][:]
    uy.vector()[:] = x[1][:]

    p = df.Function(V_p[0], name='pressure')
    p.vector()[:] = x[2][:]

    file_ux = df.File('output/stokes_ux.pvd')
    file_uy = df.File('output/stokes_uy.pvd')
    file_ux << ux, 0
    file_uy << uy, 0
    file_p = df.File('output/stokes_p.pvd')
    file_p << p, 0


if __name__ == '__main__':
    #demo()
    demo2()
