/*!
 * 
 *
 * \brief
 * \author      O. Krause 
 * \date        2017
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#define SHARK_COMPILE_DLL
#include <shark/Core/DLLSupport.h>
#include  <shark/Algorithms/GradientDescent/LineSearch.h>

namespace shark{

namespace{
/// \brief backtracking line search statisfying the weak wolfe conditions
template <class VectorT, class Function>
void backtracking(
	VectorT &point,
	const VectorT &searchDirection,
	double &value,
	Function const& func,
	VectorT &gradient,
	double t = 1.0
) {
	SIZE_CHECK(point.size() == searchDirection.size());
	SIZE_CHECK(point.size() == gradient.size());

	// Constants
	const std::size_t maxIter = 100; //maximum number of iterations to try
	const double shrinking = 0.5;//shrinking factor when condition is not yet fulfilled
	const double c1 = 1e-4;//constant for weak wolfe condition
	
	double gtd = inner_prod(gradient, searchDirection);
	// Initial step values
	VectorT g_new(point.size());
	double f_new = value;
	
	std::size_t iter = 0;
	while(iter < maxIter) {
		f_new  = func.evalDerivative(point + t * searchDirection, g_new);
		if (f_new < value + c1 * t * gtd) {
			break;
		}else{
			t *= shrinking;
			++iter;
		}
	}
	//~ std::cout<<f_new<<" "<<value<<std::endl;
	if (iter < maxIter){
		noalias(point) += t * searchDirection;
		value = f_new;
		gradient = g_new;
	}
}

/*!
 *  \brief Minimizes a function of "N" variables by using
 *         derivative information.
 *
 *  Performs a minimization of a function of \f$ N \f$ variables, i.e.
 *  given as input the vectors \f$ P \f$ and \f$ n \f$ and the function
 *  \f$ f \f$, the function finds the scalar \f$ \lambda \f$
 *  that minimizes \f$ f(P + \lambda n) \f$.
 *  \f$ P \f$ is replaced by \f$ P + \lambda n \f$ and \f$ n \f$
 *  by \f$ \lambda n \f$.
 *
 *  \param  p     N-dimensional initial starting point for the search, is set to the point where the function takes on a minimum.
 *  \param  searchDirection    N-dimensional search direction, is replaced by the actual vector displacement that \em p was moved.
 *  \param  value  The function value at the new point \em p.
 *  \param  func  The function that will be minimized.
 *  \param  ax    guess for the lower bracket
 *  \param  bx    guess for the upper bracket
 *  \return       none.
 *  \throw SharkException the type of the exception will be "size mismatch" and indicates that \em p is not one-dimensional or that \em p and \em searchDirection don't have the same length
 */
template<class VectorT,class VectorU,class DifferentiableFunction>
void dlinmin
(
	VectorT& p,
	const VectorU& searchDirection,
	double& value,
	DifferentiableFunction& func,
	double ax,
	double bx
)
{
	SIZE_CHECK(p.size()==searchDirection.size());

	const double   GOLD   = 1.618034;
	const double   GLIMIT = 100.;
	const double   TINY   = 1.0e-20;
	const std::size_t ITMAX  = 100;
	const double   ZEPS   = 1.0e-10;
	const double   TOL    = 2.0e-4;

	bool     ok1, ok2;
	std::size_t iter;
	double   fa, fb, fc, fp, cx;
	double   ulim, dum;
	double   a, b, d(0.), e, fu, fv, fw, fx, q, r, tol1, tol2, u, v, w, x, xm;
	double   dv, dw, dx, d1, d2, u1, u2, olde;

	std::size_t        n = p.size();
	VectorT xt(n);
	VectorT gradient(n);

	//===================================================================

	fa = fp = func.eval(p);

	noalias(xt) = p + searchDirection;
	fb = func.eval(xt);

	//ensure that fb <= fa
	if (fb > fa) {
		dum = ax;
		ax = bx;
		bx = dum;
		dum = fb;
		fb = fa;
		fa = dum;
	}

	//evaluate function a golden selection
	cx = bx + GOLD * (bx - ax);
	noalias(xt) = p + cx*searchDirection;
	fc = func.eval(xt);

	while (fb > fc) {//find interval containing the minimum
		r = (bx - ax) * (fb - fc);
		q = (bx - cx) * (fb - fa);
		u = bx - ((bx - cx) * q - (bx - ax) * r) /
			(2. * copySign(std::max(std::abs(q - r), TINY), q - r));
		ulim = bx + GLIMIT * (cx - bx);
		if ((bx - u) *(u - cx) > 0.) {//u is in the interval (cx,bx)
			noalias(xt) = p + u*searchDirection;
			fu = func.eval(xt);
			if (fu < fc) {//continue in the interval [bx,u]
				ax = bx;
				bx = u;
				fa = fb;
				fb = fu;
				break;
			}
			else if (fu > fb) {
				cx = u;
				fc = fu;
				break;
			}
			//evaluate new golden selection point
			u = cx + GOLD * (cx - bx);
			noalias(xt) = p + u*searchDirection;
			fu = func.eval(xt);
		}
		else if ((cx - u) *(u - ulim) > 0.) {//u is in the interval [ulim,cx]
			noalias(xt) = p + u*searchDirection;
			fu = func.eval(xt);
			if (fu < fc) {//continue in the interval [ax,cx]
				bx = cx;
				cx = u;
				u  = cx + GOLD * (cx - bx);
				fb = fc;
				fc = fu;
				noalias(xt) = p + u*searchDirection;
				fu = func.eval(xt);
			}
		}
		else if ((u - ulim) *(ulim - cx) >= 0.) {//ulim is in the interval [cx,u]
			u = ulim;
			noalias(xt) = p + u*searchDirection;
			fu = func.eval(xt);
		}
		else {
			u = cx + GOLD * (cx - bx);
			noalias(xt) = p + u*searchDirection;
			fu = func.eval(xt);
		}
		//exchange interval limits and corresponding function values
		ax = bx;
		bx = cx;
		cx = u;
		fa = fb;
		fb = fc;
		fc = fu;
	}

	//=======================================================================

	e = 0.;
	if (ax < cx) {
		a = ax;
		b = cx;
	}
	else {
		a = cx;
		b = ax;
	}

	x = w = v = bx;
	noalias(xt) = p + x*searchDirection;
	fw = fv = fx = func.evalDerivative(xt, gradient);
	dx = inner_prod(searchDirection,gradient);
	dw = dv = dx;

	for (iter = 0; iter < ITMAX; iter++) {
		xm = 0.5 * (a + b);//interval centrum
		tol2 = 2. * (tol1 = TOL * std::abs(x) + ZEPS);
		if (std::abs(x - xm) <= (tol2 - 0.5 *(b - a))) {
			break;
		}
		// calculate delta x
		if (std::abs(e) > tol1) {
			d1 = 2. * (b - a);
			d2 = d1;
			if (dw != dx)
				d1 = (w - x) * dx / (dx - dw);
			if (dv != dx)
				d2 = (v - x) * dx / (dx - dv);
			u1   = x + d1;
			u2   = x + d2;
			ok1  = (a - u1) * (u1 - b) > 0. && dx * d1 <= 0.;
			ok2  = (a - u2) * (u2 - b) > 0. && dx * d2 <= 0.;
			olde = e;
			e    = d;
			if (ok1 || ok2) {
				if (ok1 && ok2)
					d = (std::abs(d1) < std::abs(d2) ? d1 : d2);
				else if (ok1)
					d = d1;
				else
					d = d2;
				if (std::abs(d) <= std::abs(0.5 * olde)) {
					u = x + d;
					if (u - a < tol2 || b - u < tol2)
						d = copySign(tol1, xm - x);
				}
				else {
					d = 0.5 * (e = (dx >= 0. ? a - x : b - x));
				}
			}
			else {
				d = 0.5 * (e = (dx >= 0. ? a - x : b - x));
			}
		}
		else {
			d = 0.5 * (e = (dx >= 0. ? a - x : b - x));
		}
		//update u
		if (std::abs(d) >= tol1) {
			u = x + d;
			noalias(xt) = p + u*searchDirection;
			fu = func.eval(xt);
		}
		else {
			u = x + copySign(tol1, d);
			noalias(xt) = p + u*searchDirection;
			fu = func.eval(xt);
			if (fu > fx)
				break;
		}
		func.evalDerivative(xt, gradient);
		double du =  inner_prod(searchDirection, gradient);
		//reduce interval length
		if (fu <= fx) {
			if (u >= x)
				a = x;
			else
				b = x;
			v  = w;
			w  = x;
			x  = u;
			fv = fw;
			fw = fx;
			fx = fu;
			dv = dw;
			dw = dx;
			dx = du;
		}
		else {
			if (u < x)
				a = u;
			else
				b = u;
			if (fu <= fw || w == x) {
				v  = w;
				w  = u;
				fv = fw;
				fw = fu;
				dv = dw;
				dw = du;
			}
			else if (fu < fv || v == x || v == w) {
				v  = u;
				fv = fu;
				dv = du;
			}
		}
	}

	if (fx < fp) {
		value = fx;
		noalias(p) += searchDirection * x;
	}
	else
		value = fp;
}


///  \brief Cubic interpolation from Numerical Optimization p. 59.
inline double wlsCubicInterp(double t1, double t2,
        double f1, double f2,
        double gtd1, double gtd2
) {
	if (t1 == t2)
		return t1;
	if (t2 < t1) {
		std::swap(t1, t2);
		std::swap(f1, f2);
		std::swap(gtd1, gtd2);
	}

	double d1 = gtd1 + gtd2 - 3 * (f1 - f2)/(t1 - t2);
	if (d1 * d1 - gtd1 * gtd2 < 0)
		return (t1 + t2) / 2.0;
	double d2 = std::sqrt(d1 * d1 - gtd1 * gtd2);
	double t = t2 - (t2 - t1) * ((gtd2 + d2 - d1)/(gtd2 - gtd1 + 2 * d2));

	// New step length should be in interval [t1, t2]
	return std::min(std::max(t, t1), t2);
}
/// \brief Line search, using cubic interpolation, satisfying the strong Wolfe conditions.
template <class VectorT, class Function>
void wolfecubic(
	VectorT &point,
	const VectorT &searchDirection,
	double &value,
	Function const& func,
	VectorT &gradient,
	double t = 1.0
) {
	SIZE_CHECK(point.size() == searchDirection.size());
	SIZE_CHECK(point.size() == gradient.size());

	typedef typename VectorT::value_type Float;
	// Constants
	const double tol = 1e-9;
	const size_t maxIter = 25;
	const double c1 = 1e-4;
	const double c2 = 0.9;
	double maxD = norm_1(searchDirection);

	// Previous step
	double f_prev = value;
	double t_prev = 0.0;
	VectorT g_prev = gradient;
	double gtd = inner_prod(gradient, searchDirection);

	// Initial step values
	VectorT g_new(point.size());
	double f_new  = func.evalDerivative(point + Float(t) * searchDirection, g_new);
	double gtd_new = inner_prod(g_new, searchDirection);

	// Bracket vars
	double bracket[2];//bracket [t_lo,t_hi]
	double bracketf[2];//function values at the borders of the bracket
	VectorT bracketg[2];//gradient values at the borders of the bracket
	bool single = false;

	unsigned int iter = 0;
	bool done = false;

	// Bracketing phase.
	// Create a bracke [t_lo, t_hi] within which there lies a point satisfying
	// the strong Wolfe conditions with certainty.
	while(iter++ < maxIter) {
		// If the new point doesn't decrease enough there must be one that
		// does so before it.
		if (f_new > value + c1 * t * gtd || (iter > 1 && f_new >= f_prev)) {
			bracket[0] = t_prev;
			bracket[1] = t;
			bracketf[0] = f_prev;
			bracketf[1] = f_new;
			bracketg[0] = g_prev;
			bracketg[1] = g_new;
			break;
		}
		// If t satisfies both wolfe conditions we are done.
		if (std::abs(gtd_new) <= -c2 * gtd) {
			bracket[0] = t;
			bracketf[0] = f_new;
			bracketg[0] = g_new;
			done = single = true;
			break;
		}
		// If the directional searchDirection at the new point is positive, there
		// must be a good point before it.
		if (gtd_new >= 0) {
			bracket[0] = t_prev;
			bracket[1] = t;
			bracketf[0] = f_prev;
			bracketf[1] = f_new;
			bracketg[0] = g_prev;
			bracketg[1] = g_new;
			break;
		}

		t_prev = t;
		t *= 10; // Expand the bracket.

		f_prev = f_new;
		g_prev = g_new;
		f_new = func.evalDerivative(point + Float(t) * searchDirection, g_new);
		gtd_new = inner_prod(g_new, searchDirection);
	}

	// Convention: prev = lo and new = hi.
	bool insuf = false;
	// Zooming phase
	// We know a good point lies in [t_lo, t_hi]. Now find it!
	while (!done && iter++ < maxIter) {
		// Make sure lo and hi are correct
		size_t lo = bracketf[1] < bracketf[0] ? 1 : 0;
		size_t hi = 1 - lo;

		// Cubic interpolation for new step length.
		t = wlsCubicInterp(bracket[0], bracket[1], bracketf[0], bracketf[1],
		        inner_prod(bracketg[0], searchDirection),
		        inner_prod(bracketg[1], searchDirection));

		// Sufficient progress?
		double mint = std::min(bracket[0], bracket[1]);
		double maxt = std::max(bracket[0], bracket[1]);
		if (std::min(maxt - t, t - mint)/(maxt - mint) < 0.1) {
			if (insuf || t >= maxt || t <= mint) {
				if (std::abs(t - maxt) < std::abs(t - mint))
					t = maxt - 0.1 * (maxt - mint);
				else
					t = mint + 0.1 * (maxt - mint);
				insuf = false;
			} else
				insuf = true;
		} else
			insuf = false;

		f_new = func.evalDerivative(point + Float(t) * searchDirection, g_new);
		gtd_new = inner_prod(g_new, searchDirection);

		// If the new point doesn't decrease enough, make it new t_hi
		if (f_new > value + c1 * t * gtd || f_new > bracketf[lo]) {
			bracket[hi] = t;
			bracketf[hi] = f_new;
			bracketg[hi] = g_new;
		} else {
			// If step has sufficient length, use it.
			if (std::abs(gtd_new) <= -c2 * gtd)
				done = true;
			else if (gtd_new * (bracket[hi] - bracket[lo]) >= 0) {
				bracket[hi] = bracket[lo];
				bracketf[hi] = bracketf[lo];
				bracketg[hi] = bracketg[lo];
			}
			bracket[lo] = t;
			bracketf[lo] = f_new;
			bracketg[lo] = g_new;
		}

		// tolerance checking. If the steps become too small avoid them.
		if (!done && std::abs(bracket[0] - bracket[1])*maxD < tol) {
			break;
		}
	}

	if (iter < maxIter || value > bracketf[0] || value > bracketf[1]) {
		if (bracketf[0] < bracketf[1] || single) {
			noalias(point) += bracket[0] * searchDirection;
			value = bracketf[0];
			gradient = bracketg[0];
		} else {
			noalias(point) += bracket[1] * searchDirection;
			value = bracketf[1];
			gradient = bracketg[1];
		}
	}
}
	
}


template<class SearchPointType>
void LineSearch<SearchPointType>::operator()(SearchPointType &searchPoint,double &pointValue,SearchPointType const& newtonDirection, SearchPointType &derivative, double stepLength)const{
	switch (m_lineSearchType) {
	case LineSearchType::Dlinmin:
		dlinmin(searchPoint, newtonDirection, pointValue, *m_function, m_minInterval, m_maxInterval);
		m_function->evalDerivative(searchPoint, derivative);
		break;
	case LineSearchType::WolfeCubic:
		wolfecubic(searchPoint, newtonDirection, pointValue, *m_function, derivative, stepLength);
		break;
	case LineSearchType::Backtracking:
		backtracking(searchPoint, newtonDirection, pointValue, *m_function, derivative, stepLength);
		break;
	}
}


template class SHARK_EXPORT_SYMBOL LineSearch<RealVector>;
template class SHARK_EXPORT_SYMBOL LineSearch<FloatVector>;
#ifdef SHARK_USE_OPENCL
template class SHARK_EXPORT_SYMBOL LineSearch<RealGPUVector>;
template class SHARK_EXPORT_SYMBOL LineSearch<FloatGPUVector>;
#endif
}