import dolfin as df 
from block import block_vec, block_mat

from .prolongation import block_prolongation


class SmootherSymmetric:
    def __init__(self, A,  pAinv, pATinv, pSinv):
        self.A = A
        self.pAinv = pAinv
        self.pATinv = pATinv
        self.pSinv = pSinv
    
    def mult(self, b, x):
        A, pAinv, pATinv, pSinv = self.A, self.pAinv, self.pATinv, self.pSinv
        n,m = A.blocks.shape
        assert n == m, "only symmetric matrices supported"
        # first row:
        r0 = block_vec(b[0:n-1] - A[0:n-1,0:n-1] @ x[0:n-1] - A[0:n-1,n-1:n] @ x[n-1:n])
        x[0:n-1] = x[0:n-1] + (pAinv * r0).blocks
        # second row:
        r1 = b[n-1] - A[n-1,0:n-1] @ x[0:n-1] - A[n-1,n-1] * x[n-1]
        x[n-1] = x[n-1] - pSinv * r1
        # first row (again):
        r0 = block_vec(b[0:n-1] - A[0:n-1,0:n-1] @ x[0:n-1] - A[0:n-1,n-1:n] @ x[n-1:n])
        x[0:n-1] = x[0:n-1] + (pATinv * r0).blocks


class SmootherLower:
    def __init__(self, A,  pAinv, pSinv):
        self.A = A
        self.pAinv = pAinv
        self.pSinv = pSinv
    
    def mult(self, b, x):
        A, pAinv, pSinv = self.A, self.pAinv, self.pSinv
        n,m = A.blocks.shape
        assert n == m, "only symmetric matrices supported"
        # first row:
        r0 = block_vec(b[0:n-1] - A[0:n-1,0:n-1] @ x[0:n-1] - A[0:n-1,n-1:n] @ x[n-1:n])
        x[0:n-1] = x[0:n-1] + (pAinv * r0).blocks
        # second row:
        r1 = b[n-1] - A[n-1,0:n-1] @ x[0:n-1] - A[n-1,n-1] * x[n-1]
        x[n-1] = x[n-1] - pSinv * r1


class SmootherUpper:
    def __init__(self, A,  pAinv, pSinv):
        self.A = A
        self.pAinv = pAinv
        self.pSinv = pSinv
    
    def mult(self, b, x):
        A, pAinv, pSinv = self.A, self.pAinv, self.pSinv
        n,m = A.blocks.shape
        assert n == m, "only symmetric matrices supported"
        # second row:
        r1 = b[n-1] - A[n-1,0:n-1] @ x[0:n-1] - A[n-1,n-1] * x[n-1]
        x[n-1] = x[n-1] - pSinv * r1
        # first row:
        r0 = block_vec(b[0:n-1] - A[0:n-1,0:n-1] @ x[0:n-1] - A[0:n-1,n-1:n] @ x[n-1:n])
        x[0:n-1] = x[0:n-1] + (pAinv * r0).blocks


def estimate_omega(pAinv, pSinv, A, num_iterations=10):
    n,m = A.blocks.shape
    assert n == m, "only symmetric matrices supported"
    # extract the operator
    pSinv = block_mat(1, 1, [[pSinv]])
    C = block_mat(1, 1, [[-A[n-1,n-1]]])
    BT = block_mat(n-1, 1, A[0:n-1,n-1:n])
    B = block_mat(1, n-1, A[n-1:n,0:n-1])
    # apply our problem
    def op(x):
        y = (B * pAinv * BT * x)
        if not C[0,0] == 0:
            y += C * x
        return pSinv * y
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
        self.rtol = 1e-12
    
    def solve(self, b, x=None, residual_rate_list=[]):
        # provide initial guess
        if x is None:
            x = self.A[0].create_vec()
        else:
            x = x.copy()
        # calculate initial residual
        r = b - self.A[0] * x
        res_start = res_prev = r.norm()
        print(f'{-1} - rate = - ({res_start})')
        # iterate mg solver
        for j in range(self.num_iterations):
            x = self._solve(x, b, 0)
            # calculate residual:
            r = b - self.A[0] * x
            self.projection_nullspace(r)
            res = r.norm()
            res_rate = res / res_prev
            if self.show > 0:
                print(f'{j} - rate = {res_rate} ({res})')
            residual_rate_list.append(res_rate)
            if res < res_start * self.rtol:
                return x
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
        # print(f'{k}, pre-before, {(b_fine - A_fine * x).norm()}')
        # presmoothing steps
        for _ in range(self.num_presmoothing_steps):
            presmoother.mult(b_fine, x)
        # print(f'{k}, pre-after, {(b_fine - A_fine * x).norm()}')
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
        # print(f'{k}, coarse-after, {(b_fine - A_fine * x).norm()}')
        # postmoothing steps
        for _ in range(self.num_postsmoothing_steps):
            postsmoother.mult(b_fine, x)
        # print(f'{k}, post-after, {(b_fine - A_fine * x).norm()}')
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
        if W_fine.num_sub_spaces() == 0:
            W_fine = [W_fine]
        if W_coarse.num_sub_spaces() == 0:
            W_coarse = [W_coarse]
        P = block_prolongation(W_coarse, W_fine)
        prolongations.append(P)
    return prolongations