# coding: utf-8
import numpy as np
import numpy.polynomial.polynomial as P
poly = P.polyfromroots([1.0, 2.0])
poly
poly2 = P.fromroots(1.0, 2.0)
import numpy.polynomial.Polynomial as Poly
# remove poly2 line
# remove Poly import line
P.polyval(0.0, poly)
node_val = P.polyval(0.0, poly)
poly = poly / node_val
poly
poly_class = P(poly)
from numpy.polynomial import Polynomial as Poly
Poly(poly)
Poly(poly).integ()
Poly(poly).integ()(0.0)
Poly(poly).integ()(1.0)
%save -r shape_funs_scratch.py 1-99
