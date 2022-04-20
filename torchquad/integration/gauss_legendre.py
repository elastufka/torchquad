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
        """Returns Gauss-Legendre points and weights for degree n and dimension self._dim
        Adjusts from interval [-1,1] to integration limits [a,b]
        
        Args:
            n (int): Degree to use for computing sample points and weights, which will correctly integrate polynomials of degree 2*deg-1 or less. Generation of Gauss-Legendre points have only been tested up to degree 100.
        
        Returns:
            tuple(points, weights)
            """
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
            xi=do("repeat",xm,n,like="numpy").reshape(self._dim,n)+anp.outer(xl,xi)
        else:
            xi=do("repeat",xm.cpu(),n,like="numpy").reshape(self._dim,n).to(torch.cuda.current_device())+anp.outer(xl,xi) #what if backend isn't torch?
        wi=anp.outer(wi,xl).T
        return xi,wi

    def integrate(self, fn, dim, args=None, N=8, integration_domain=None):
        """Integrates the passed function on the passed domain using fixed-point Gauss-Legendre quadrature.

        Args:
            fn (func): The function to integrate over.
            dim (int): Dimensionality of the function to integrate.
            args (iterable object, optional): Additional arguments ``t0, ..., tn``, required by `fn`.
            N (int, optional): Degree to use for computing sample points and weights, which will correctly integrate polynomials of degree 2*deg-1 or less. Note that this is not the total number of sample points used by the integration, which will be dim**N. Generation of Gauss-Legendre points have only been tested up to degree 100. Defaults to 8.
            integration_domain (list, optional): Integration domain, e.g. [[-1,1],[0,1]]. Defaults to [-1,1]^dim.

        Returns:
            float: integral value
    
        """
        self._integration_domain = _setup_integration_domain(dim, integration_domain)
        self._check_inputs(dim=dim, N=N, integration_domain=self._integration_domain)

        logger.debug(f"Using GaussLegendre for integrating a fn with {N} points over {self._integration_domain}")

        self._dim = dim
        self._fn = fn
        
        xi, wi = self._gauss_legendre(N)
        integral= anp.sum(self._eval(xi,args=args,weights=wi)) #what if there is a sum in the function? then wi*self._eval() will have dimension mismatch
        #if dim >1:
        #    integral=anp.sum(integral)
        logger.info(f"Computed integral was {integral}.")

        return integral

