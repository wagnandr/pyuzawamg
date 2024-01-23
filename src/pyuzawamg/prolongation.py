import dolfin as df


class prolongation_operator:
    def __init__(self, coarse_space, fine_space, mat, transpose) -> None:
        self._coarse_space = coarse_space
        self._fine_space = fine_space
        self._mat = mat
        self._transpose = transpose 
    
    @property
    def dst_space(self):
        return self._coarse_space if self._transpose else self._fine_space
        
    @property
    def src_space(self):
        return self._coarse_space if not self._transpose else self._fine_space
    
    @property
    def T(self):
        return prolongation_operator(
            self._coarse_space, 
            self._fine_space, 
            self._mat, 
            not self._transpose)
    
    def matvec(self, src):
        if self._transpose:
            dst_fun = self.dst_fun = df.Function(self.src_space)
            self._mat.transpmult(src, dst_fun.vector())
        else:
            dst_fun = self.dst_fun = df.Function(self.dst_space)
            self._mat.mult(src, dst_fun.vector())
        return dst_fun.vector()

    def transpmult(self, src):
        if not self._transpose:
            dst_fun = self.dst_fun = df.Function(self.src_space)
            self._mat.transpmult(src, dst_fun.vector())
        else:
            dst_fun = self.dst_fun = df.Function(self.dst_space)
            self._mat.mult(src, dst_fun.vector())
        return dst_fun.vector()
    
    def __mul__(self, src):
        return self.matvec(src)

    def __rmul__(self, src):
        return self.matvec(src)

    @staticmethod
    def create(coarse, fine):
        mat = df.PETScDMCollection.create_transfer_matrix(coarse, fine)
        return prolongation_operator(coarse, fine, mat, False)


'''
class block_prolongation:
    def __init__(self, W_coarse, W_fine) -> None:
        self._W_fine = W_fine
        self._W_coarse = W_coarse
        self.blocks = [prolongation_operator.create(c, f) for (c,f) in zip(W_coarse, W_fine)]
    
    def mult(self, src, dst):
        src = src.copy()
        for idx in range(len(self.blocks)):
            dst[idx] = self.blocks[idx].matvec(src[idx])

    def tmult(self, src, dst):
        src = src.copy()
        for idx in range(len(self.blocks)):
            dst[idx] = self.blocks[idx].transpmult(src[idx])
'''

domain = df.CompiledSubDomain('on_boundary')

class block_prolongation:
    def __init__(self, W_coarse, W_fine) -> None:
        self._W_fine = W_fine
        self._W_coarse = W_coarse
        self.blocks = [prolongation_operator.create(c, f) for (c,f) in zip(W_coarse, W_fine)]
        self.post_mult = lambda x: x 
        self.post_tmult = lambda x: x 
    
    def mult(self, src, dst):
        #self.post_tmult(src)
        src = src.copy()
        df.DirichletBC(self._W_coarse[0], df.Constant((0,0)), domain).apply(src[0])
        for idx in range(len(self.blocks)):
            dst[idx] = self.blocks[idx].matvec(src[idx])
        #self.post_mult(dst)
        df.DirichletBC(self._W_fine[0], df.Constant((0,0)), domain).apply(dst[0])
        d = df.as_backend_type(dst[1]).vec()
        d[:] -= d.sum() / d.size

    def tmult(self, src, dst):
        src = src.copy()
        df.DirichletBC(self._W_fine[0], df.Constant((0,0)), domain).apply(src[0])
        #self.post_mult(src)
        for idx in range(len(self.blocks)):
            dst[idx] = self.blocks[idx].transpmult(src[idx])
        df.DirichletBC(self._W_coarse[0], df.Constant((0,0)), domain).apply(dst[0])
        #self.post_tmult(dst)
        d = df.as_backend_type(dst[1]).vec()
        d[:] -= d.sum() / d.size
