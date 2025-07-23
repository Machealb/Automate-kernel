# DARTR for learning kernels in operators
Data Adaptive RKHS Regularization for learning kernels in operators

## Problem Statement
We fit to the function data $(u_i,f_i)$ an operator $R_\phi[u_i] = f_i$, where  
    $$ R_\phi[u](x) = \int \phi(|y|)g[u](x,x+y) dy 
                   = \sum_r \phi(r) [ g[u](x,x+r)+ g[u](x,x-r) ] dr.$$

Examples include:

- Linear integral operator with $g[u](x,y) =  u(x+y) $  
- Nonlinear operator with $g[u](x,y) = div(u(x+y))u(x) $
- nonlocal operator with $g[u](x,y) =  u(x+y)-u(x)$


## References

[1] Haibo Li and Fei Lu. *Automatic reproducing kernel and regularization for learning convolution kernels*. arXiv:2507.11944, 2025.  
📄 [Read on arXiv](https://doi.org/10.48550/arXiv.2507.11944)
