import dolfin as df
import numpy as np
from block import block_assemble, block_bc, block_mat, block_vec
from block.iterative import MinRes, ConjGrad, Richardson
from block.algebraic.petsc import LU, AMG, LumpedInvDiag, InvDiag

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
    S_hat_inv = omega*LumpedInvDiag(M)
    return S_hat_inv


def estimate_omega(A_hat_inv, S_hat_inv, A):
    n,m = A.blocks.shape
    assert n == m, "only symmetric matrices supported"
    '''
    C = A[n-1,n-1]
    BT = block_mat(A[0:n-1,n-1:n])
    B = block_mat(A[n-1:n,0:n-1])
    def op(x):
        y0 = (B * A_hat_inv * BT * block_vec(1, [x]))[0]
        y1 = C * x
        return S_hat_inv * (y0 + y1)
    x = C.create_vec()
    op(x)
    '''
    S_hat_inv = block_mat(1, 1, [[S_hat_inv]])
    C = block_mat(1, 1, [[-A[n-1,n-1]]])
    BT = block_mat(A[0:n-1,n-1:n])
    B = block_mat(A[n-1:n,0:n-1])
    def op(x):
        y0 = (B * A_hat_inv * BT * x)
        y1 = C * x
        return S_hat_inv * (y0 + y1)
    x = C.create_vec()
    x.randomize()
    num_iterations = 10
    for _ in range(num_iterations):
        x_next = op(x) 
        alpha = x_next.inner(x)
        print(alpha)
        x_next_norm = x_next.norm()
        x = (1./x_next_norm) * x_next 
    return alpha



if __name__ == '__main__':
    #omega=1/0.55849
    omega=0.4
    presmoothing_steps = postsmoothing_steps = 3 * 1 
    w_cycles=2
    stab = True 

    N = 4 * 8 * 1
    mesh_coarse = df.RectangleMesh(df.Point(-4,-1), df.Point(4,1), 4*N, N, 'left')
    V_u_coarse = df.FunctionSpace(mesh_coarse, 'P', 2)
    V_p_coarse = df.FunctionSpace(mesh_coarse, 'P', 1)
    W_coarse = [V_u_coarse, V_u_coarse, V_p_coarse]

    mesh_fine = df.refine(mesh_coarse)
    V_u_fine = df.FunctionSpace(mesh_fine, 'P', 2)
    V_p_fine = df.FunctionSpace(mesh_fine, 'P', 1)
    W_fine = [V_u_fine, V_u_fine, V_p_fine]

    noslip_value = df.Constant(0)
    noslip_domain = df.CompiledSubDomain('x[1] >= 1-1e-8 || x[1] <= -1+1e-8')
    bc0x_fine = df.DirichletBC(V_u_fine, noslip_value, noslip_domain )
    bc0y_fine = df.DirichletBC(V_u_fine, noslip_value, noslip_domain )

    inflow_domain = df.CompiledSubDomain('x[0] <= -4+1e-8')
    inflow_valuex = df.Expression('-(x[1]+1)*(x[1]-1)', degree=2)
    inflow_valuey = df.Constant(0)
    bc1x_fine = df.DirichletBC(V_u_fine, inflow_valuex, inflow_domain)
    bc1y_fine = df.DirichletBC(V_u_fine, inflow_valuey, inflow_domain)

    bc0x_coarse = df.DirichletBC(V_u_coarse, df.Constant(0), noslip_domain )
    bc0y_coarse = df.DirichletBC(V_u_coarse, df.Constant(0), noslip_domain )
    bc1x_coarse = df.DirichletBC(V_u_coarse, df.Constant(0), inflow_domain)
    bc1y_coarse = df.DirichletBC(V_u_coarse, df.Constant(0), inflow_domain)

    bc1x_fine_zero = df.DirichletBC(V_u_fine, df.Constant(0), inflow_domain)
    bc1y_fine_zero = df.DirichletBC(V_u_fine, df.Constant(0), inflow_domain)

    solution = df.Function(V_u_fine)
    solution.interpolate(inflow_valuex)

    P = block_prolongation(W_coarse, W_fine)

    bcs_fine = [
        [bc0x_fine, bc1x_fine],
        [bc0y_fine, bc1y_fine],
        []
    ]

    bcs_fine_zero = [
        [bc0x_fine, bc1x_fine_zero],
        [bc0y_fine, bc1y_fine_zero],
        []
    ]

    bcs_coarse = [
        [bc0x_coarse, bc1x_coarse],
        [bc0y_coarse, bc1y_coarse],
        []
    ]

    A_fine, b_fine = assemble_system(W_fine, bcs_fine, stab=stab)
    #block_bc(bcs_fine, False).apply(A_fine, b_fine)

    A_coarse, b_coarse = assemble_system(W_coarse, bcs_coarse, stab=stab)
    #block_bc(bcs_coarse, False).apply(A_coarse, b_coarse)

    Ainv_fine = diagonal_preconditioned_minres(A_fine, W_fine)
    Ainv_coarse = diagonal_preconditioned_minres(A_coarse, W_coarse)

    A_hat_inv = create_A_hat_inv(A_fine)
    S_hat_inv = create_S_hat_inv(W_fine[-1], 1.)
    omega_inv = estimate_omega(A_hat_inv, S_hat_inv, A_fine)
    S_hat_inv = create_S_hat_inv(W_fine[-1], 1./omega_inv)

    smoother_lower = SmootherLower(A_fine, A_hat_inv, S_hat_inv)
    smoother_upper = SmootherUpper(A_fine, A_hat_inv, S_hat_inv)



    x = Ainv_fine.create_vec()

    r_fine = b_fine - A_fine * x
    res_prev = r_fine.norm()

    for j in range(10):
        for i in range(presmoothing_steps):
            smoother_lower.mult(b_fine, x)
            #r_fine = b_fine - A_fine * x
            #print(f'{j}-{i}: residual = {r_fine.norm()}')
        for k in range(w_cycles):
            r_fine = b_fine - A_fine * x
            r_coarse = A_coarse.create_vec()
            #[x.apply(r_fine[0]) for x in bcs_fine_zero[0]]
            #[x.apply(r_fine[1]) for x in bcs_fine_zero[1]]
            P.tmult(r_fine, r_coarse) 
            #[x.apply(r_coarse[0]) for x in bcs_coarse[0]]
            #[x.apply(r_coarse[1]) for x in bcs_coarse[1]]
            d_coarse = Ainv_coarse * r_coarse
            #print(f'd_coarse {d_coarse.norm()}')
            d_fine = A_fine.create_vec()
            P.mult(d_coarse, d_fine) 
            #[x.apply(d_fine[0]) for x in bcs_fine_zero[0]]
            #[x.apply(d_fine[1]) for x in bcs_fine_zero[1]]
            x += d_fine
            #r_fine = b_fine - A_fine * x
            #print(f'{j}-cgc: residual = {r_fine.norm()}')
        for i in range(postsmoothing_steps):
            smoother_upper.mult(b_fine, x)
            #r_fine = b_fine - A_fine * x
            #print(f'{j}-{i}: residual = {r_fine.norm()}')
        # print residual rate
        r_fine = b_fine - A_fine * x
        zero_mean(mesh_fine, r_fine[-1])
        res = r_fine.norm()
        print(f'{j} - rate = {res / res_prev} ({res})')
        res_prev = res

    #x = Ainv_fine * b_fine

    # Schur evaluation S = 0.5 * S + 0.5 * S = 0.5 * S1 + 0.5 * S2

    ux = df.Function(V_u_fine, name='vx')
    uy = df.Function(V_u_fine, name='vy')
    ux.vector()[:] = x[0][:]
    uy.vector()[:] = x[1][:]

    p = df.Function(V_p_fine, name='pressure')
    p.vector()[:] = x[2][:]

    file_ux = df.File('output/stokes_ux.pvd')
    file_uy = df.File('output/stokes_uy.pvd')
    file_ux << ux, 0
    file_uy << uy, 0
    #file_u = df.XDMFFile('output/stokes_u.xdmf')
    #file_u.write(ux, 0)
    #file_u.write(uy, 0)
    file_p = df.File('output/stokes_p.pvd')
    file_p << p, 0
