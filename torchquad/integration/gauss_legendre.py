import torch
#import numpy
from loguru import logger
from autoray import numpy as anp
from autoray import do

from .base_integrator import BaseIntegrator
from .utils import _setup_integration_domain

class GaussLegendre(BaseIntegrator):
    """Gauss Legendre quadrature rule in torch. See https://en.wikipedia.org/wiki/Gaussian_quadrature#Gauss%E2%80%93Legendre_quadrature."""

    def __init__(self):
        super().__init__()
        
    def _gauss_legendre(self,n):
        """returns Gauss-Legendre points and weights for degree n and dimension self._dim
        Adjusts from interval [-1,1] to integration limits [a,b]"""
        a,b=self._integration_domain.T
        xi, wi = do("polynomial.legendre.leggauss",n,like="numpy")
        #scale from [-1,1] to [a,b] e.g. https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
        
        xm=0.5*(b+a)
        xl=0.5*(b-a)
        if isinstance(xm,torch.Tensor):
            #for now... figure out a better solution later
            aa=torch.zeros(n)
            xi=aa.new(xi)
            wi=aa.new(wi)

        if xm.device !='cpu':
            xi=do("repeat",xm,n,like="numpy").reshape(self._dim,npoints)+anp.outer(xl,xi)
            #now xm has to be on CPU ... but xi needs to be on GPU
        else:
            xi=do("repeat",xm.cpu(),n,like="numpy").reshape(self._dim,npoints).to(torch.cuda.current_device())+anp.outer(xl,xi)
        wi=anp.outer(wi,xl).T
        return xi,wi

    def integrate(self, fn, dim, args=None, N=2, eps_abs=None, eps_rel=1e-3, max_N=12, base=2, integration_domain=None, fixed=False):
        """Integrates the passed function on the passed domain using Gauss-Legendre quadrature.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            args (iterable object, optional): Additional arguments ``t0, ..., tn``, required by `fn`.
            N (int, optional): Total number of sample points to use for the integration. Defaults to 2.
            eps_abs (float, optional): Absolute error condition used to evaluate quadrature. Defaults to None
            eps_rel (float, optional): Relative error condition used to evaluate quadrature. Defaults to 1e-3
            max_N (int, optional): Maximum number of sample points to use for the integration. Defaults to 12.
            base (int, optional): Base number to use for determining npoints. Defaults to 2. This means if N=2, the number of Gauss-Legendre points that the integral starts with will be base**N or 2**2=4. Likewise, the maximum number of points that the integral will evaluate will be base**max_N, or with the defaults 2**12=4096.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.

        Returns:
            float: integral value
            
        Basic example (works):
            def cosfn(x): return np.cos(x)
            gl=GaussLegendre()
            integral=gl.integrate(cosfn,dim=10,eps_rel=1e-10)
        """
        self._integration_domain = _setup_integration_domain(dim, integration_domain)
        self._check_inputs(dim=dim, N=N, integration_domain=self._integration_domain) #might need to check more

        #logger.debug(f"Using Trapezoid for integrating a fn with {npoints} points over {self._integration_domain}")

        self._dim = dim
        self._fn = fn
        #with torch.no_grad(): #should already be like this since no nn

        all_points,all_weights=[],[]
        for ires in range(N, max_N + 1): #if starting at npoints=8
            npoints = base ** ires #is this standard?
            
#            if npoints > base**max_N:
#                raise ValueError(f"Integral did not satisfy the conditions eps_abs={eps_abs} or eps_rel={eps_rel} using the maximum number of points {base**max_N}") #or a different error?
#                break

            # generate positions and weights
            xi, wi = self._gauss_legendre(npoints)
            all_points.append(xi.flatten())
            all_weights.append(wi.flatten())
            
        #logger.debug("Evaluating integrand for {xi}.")
        all_eval=self._eval(xi[i], args=args)*wi[i]
        
        #now reshape based on dim, npoints... does this have to be done in loop? booo.....
    
        if self._nr_of_fevals > 0:
            lastsum = anp.array(integral)
            integral[i] = anp.sum(self._eval(xi[i], args=args)*wi[i],axis=1) #[a[i.tolist()] for a in args]
        else:
            integral = torch.sum(self._eval(xi,args=args)*wi,axis=1) #integral from a to b f(x) ≈ sum (w_i*f(x_i))
            if fixed:
                pass#break #no error evaluation if fixed-point quadrature desired

        #print(npoints,integral)
        # Convergence criterion
        if self._nr_of_fevals//self._dim > 1:
            l1 = anp.abs(integral - lastsum)
            if eps_abs is not None:
                i = anp.where(l1 > eps_abs)[0]
            if eps_rel is not None:
                l2 = eps_rel * anp.abs(integral)
                i = anp.where(l1 > l2)[0]
        else:
            i= anp.arange(self._dim) #indices of integral

        # If all point have reached criterion return value
        if i.size == 0:
            pass#break

        #logger.info(f"Computed integral was {integral}.")
        return integral

