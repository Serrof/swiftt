# swiftt

## Name
SoftWare for Intrusive Function's Taylor-series Truncation

## Author
Romain Serra, PhD (serra.romain@gmail.com)

## Dependencies
*swiftt* is a Python 3 library that only requires [NumPy](https://numpy.org/), 
[SciPy](https://scipy.org/) and [Numba](https://numba.pydata.org/) to run.
Optionally, [SymPy](https://www.sympy.org/en/index.html) can be used to easily instantiate objects. 
All these third parties are available via pip and/or conda.

## Documentation
Automatic generation of the documentation is possible with [Sphinx](https://www.sphinx-doc.org/en/master/).

## Features
### In a nutshell
*swiftt* is a pure-Python, object-oriented library implementing the so-called Taylor Differential Algebra (TDA), 
as formalized by Berz [3, Chapter 2] and whose pillars are:

- Multivariate Automatic Differentiation (AD) at arbitrary order.
- Derivation and anti-derivation w.r.t. independent variables.
- Differentiation of inverse functions.

All of the above is achieved by the manipulation of Taylor expansions a.k.a. Truncated Taylor Series, which *swiftt*
does both for complex and real numbers. 

The present library also has a few additional functionalities for the latter case:
- An ordering between expansions, making ">" or "<" comparisons possible. 
- Some integration schemes e.g. Runge-Kutta 4 compatible with AD, which are a must-have to apply it to dynamical systems.
- Interval arithmetic, which is of interest to bound the polynomial part of a Taylor expansion.

### In more details

#### Generalities

The TDA is a computational tool, made possible by the power of modern machines yet also based on solid mathematical grounds, 
at the heart of which is Taylor's theorem.
The polynomial coefficients of a Taylor expansion can be mapped straightforwardly to the partial derivatives of the 
underlying functions, and are even sometimes called the normalized derivatives. So computing a Taylor expansion at a
given order is equivalent to the determination of all its derivatives at this order. 
As a matter of fact, *swiftt* works with the normalized derivatives, as they are more intuitive to handle for some operations.
The univariate case has its own implementation as many simplifications occur.
Note that for *n* variables and order *d*, the number of coefficients is *(n+d)!/n!d!*, which grows exponentially
with both parameters. This limits in practise what values can be used.

#### AD

Automatic Differentiation consists in computing not just the nominal output(s) of a computer routine,
but also its derivatives w.r.t. its inputs (so assuming continuous values) up to a given order, often equal to one.

To the best of the author's knowledge, all dedicated implementations of multivariate, high-order AD
(in the broader frame of the TDA), from Berz's first proposal to the present one, 
use a so-called forward approach.
This means that the computations start from the independent variables and work out the chain rule from there, 
much as engineers learn how to calculate Taylor expansions by hand at university.
Some first-order AD codes use the reverse direction in the chain rule, or a blend of the two, 
but their extension to arbitrary order can only be done by nesting, which is suboptimal.

If available, a natural way of implementing forward AD, used here, is to overload all the algebraic operators
('+','-','*' and '/') as well as the intrinsic a.k.a. usual functions, such as cosine. 
An elegant, double recursion exists to achieve this [10]. 
However, it trades performance for conciseness and as such is not pursued by *swiftt*,
which looks instead at explicit algorithms, also for pedagogical purposes.

The overloaded algebraic operations reflect the properties of function differentiation, such as linearity. 
Multiplication between Taylor expansions consists in forming the truncated product of their polynomial parts 
(equivalent on the partial derivatives to Leibniz's rule). 
It is the bottleneck of the whole TDA, as it is a building brick for many other things 
and unlike addition for example, it does not exhibit linear complexity w.r.t. the number of normalized derivatives. 
For this reason, *swiftt* uses a look-up table and leverages on Just In Time compilation via Numba for performance. 
As for division, it utilizes the reciprocal, which is treated like an intrinsic function, as described thereafter.

Intrinsic, scalar-valued functions use the composition (from the left) rule with univariate expansions. 
Their Maclaurin series (Taylor series around zero) are well known. For other reference points, their coefficients
follow a recursive formula. Note that for functions that are D-finite [9], like most of the intrinsic ones, 
this recursion is actually linear in *n*. When applicable, remarkable identities could be exploited instead, 
but it is less computationally efficient and is not the path followed by *swiftt*.

#### Derivation and anti-derivation operators
The most common definition of the differentiation operator, used here, is the one that coincides with differentiation 
on functions, but other are possible [5], as long as it follows Leibniz's law. 
To be rigorous, the order of the remainder should be lowered, 
but both options are available in *swiftt* for convenience. 
The anti-derivation is the "inverse" operation
(but is really an inverse only in the univariate case, constant part aside).
In *swiftt* it coincides with the classical integration on function. 
Both operators use the same look-up table.

#### Derivatives of inverse functions
Berz coined the term "Taylor maps" for collections of Taylor expansions. If they have the same number of components
than of independent variables and the underlying Jacobian is invertible, then according to the inverse function theorem,
the function itself has an inverse. In the univariate (and non-truncated) case, finding the derivatives of the inverse 
is called series reversion.
Berz proposed a fixed-point, always-converging algorithm to obtain the inverted Taylor map, 
which requires the composition rule between compatible Taylor expansions. 
The latter boils down to composition of the polynomial parts, equivalent on the partial derivatives 
to Faa Di Bruno's formula. 

## Some applications of the TDA
### Multivariate, high-order AD 
It has applications whenever partial derivatives are needed, for example in numerical, local 
optimization (although usually only up to second-order). It is a good compromise between symbolic 
calculations (that are costly) and finite differences methods (that are noisy). 

The TDA in general also finds its use in uncertainty quantification, as the Taylor expansion of a function is its best 
polynomial, local approximation. The classical TDA only keeps track of the remainder symbolically, but it can be 
extended to the so-called Taylor Models [4] which include an interval enclosure of the error.
### Derivation and anti-derivation operators
They have applications in numerical integration schemes, for example with the so-called Picard integrator. 
### Derivatives of inverse functions 
Computing them is useful when solving implicit systems of equations or to swap 
independent and dependent variables.

## Origins of the TDA
Note that the TDA is sometimes referred to simply as Differential Algebra in the literature. However, it 
is not to be confused with the eponym branch of mathematics which is broader.
Some people even call it Jet Transport in the context of dynamical systems [7].

The TDA was introduced in the seminal work of Berz in the late 1980s (see for example [1, 2]), 
for application in particle physics. 
As far as multivariate AD is concerned, it can be seen as a generalization both to multiple variables and high order of the 
so-called dual numbers. In a way, the TDA is the pinnacle of calculus, whose invention is attributed independently to 
Leibniz and Newton (although only rigorously defined much later by Cauchy [8]).

Since the late 2000s, some researchers have been applying the TDA to astrodynamics (often in collaboration with 
Berz himself, cf. [6]), eventually leading the author to come across it.

## Similar libraries
Below is a non-exhaustive list of open-source libraries (all on GitHub) in various programming languages implementing 
the three corner stones of the TDA, except if indicated otherwise. In no way does *swiftt* claim to be more 
computationally efficient than any of them, it is just a pure Python project, showcasing all the internal tricks of 
the TDA.

### DACE (C++)
The first library developed with astrodynamics applications in mind, the 
[Differential Algebra Computational Engine](https://github.com/dacelib/dace) has 
been thoroughly validated against COSY Infinity, the original software of Berz which is not open source.

### AuDi (C++) and pyaudi (Python)
A header-only C++ library also wrapped in Python, [AuDi](https://github.com/darioizzo/audi) 
(for Automatic Differentiation) has a good [documentation](https://darioizzo.github.io/audi/#)
both for the theory and the practise of the TDA.

As of version 1.7 of pyaudi, only the Taylor map inversion is exposed in Python, not the intermediate composition rules.

### Hipparchus (Java, and Python via Orekit)
Named after the Greek scholar, [Hipparchus](https://github.com/Hipparchus-Math/hipparchus) is a generic mathematical library 
available in Java 1.8, originally forked from Apache Commons Math. 
It is the main dependency of Orekit, a low-level astrodynamics library. 
Because Java does not support operator overloading, the syntax for the TDA is a bit tedious, however the developers have
already coded everything in the library with it on top of the original methods. 

Since its version 2.2, Hipparchus features Derivation and anti-derivation (a contribution of the author)
as well as composition and Taylor map inversion, making it a complete TDA emulator.
The Python wrapper of Orekit 11.3 (to appear) is expected to include Hipparchus 2.2.

### TaylorSeries (Julia)
 [TaylorSeries](https://github.com/JuliaDiff/TaylorSeries.jl) is used by its developers as the core dependency for some 
 of their other libraries. It looks like it does not feature map inversion.

### SMART-UQ (C++)
[Strathclyde Mechanical and Aerospace Research Toolboxes - UQ](https://github.com/strath-ace/smart-uq)
is a generic library for Uncertainty Quantification. It only features the AD part of the TDA, but it is mentionned
here as it was the author's first contribution to an open source project.

## References

[1] BERZ, A. Differential algebraic description of beam dynamics to very high orders. *Part. Accel.*, 
1988, vol. 24, no SSC-152, p. 109-124.

[2] BERZ, M. Differential algebra-a new tool. 
*In : Proceedings of the 1989 IEEE Particle Accelerator Conference,.' Accelerator Science and Technology*. 
IEEE, 1989. p. 1419-1423.

[3] BERZ, Martin. *Modern map methods in particle beam physics*. Academic Press, 1999.

[4] MAKINO, Kyoko et BERZ, Martin. Taylor models and other validated functional inclusion methods.
*International Journal of Pure and Applied Mathematics*, 2003, vol. 6, p. 239-316.

[5] VALLI, Monica, ARMELLIN, Roberto, DI LIZIA, Pierluigi, et al. Nonlinear mapping of uncertainties in celestial mechanics. 
*Journal of Guidance, Control, and Dynamics*, 2013, vol. 36, no 1, p. 48-63.

[6] ARMELLIN, Roberto, DI LIZIA, Pierluigi, BERNELLI-ZAZZERA, Franco, et al. Asteroid close encounters characterization 
using differential algebra: the case of Apophis. 
*Celestial Mechanics and Dynamical Astronomy*, 2010, vol. 107, no 4, p. 451-470.

[7] ALESSI, Elisa Maria, FARRES, Ariadna, VIEIRO, Arturo, et al. Jet transport and applications to NEOs. 
*In : Proceedings of the 1st IAA Planetary Defense Conference*, Granada, Spain. 2009. p. 10-11.

[8] BOYER, Carl B. *The history of the calculus and its conceptual development:(The concepts of the calculus).* 
Courier Corporation, 1959.

[9] LIPSHITZ, Leonard. D-finite power series. *Journal of algebra*, 1989, vol. 122, no 2, p. 353-373

[10] KALMAN, Dan. Doubly recursive multivariate automatic differentiation. *Mathematics magazine*, 
2002, vol. 75, no 3, p. 187-202.
