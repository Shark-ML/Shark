//===========================================================================
/*!
 *  \file BFGS.h
 *
 *  \brief Offers the Broyden-Fletcher-Goldfarb-Shanno algorithm
 *         for the optimization of models.
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Copyright (c) 1999
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      ReClaM
 *
 *
 *  This file is part of ReClaM. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ---------------------------------------------------------------------
 *
 * This code is a modification of the original code from the
 * library:
 *
 * ============================================================
 * COOOL           version 1.1           ---     Nov,  1995
 * Center for Wave Phenomena, Colorado School of Mines
 * ============================================================
 *
 * This code is part of a preliminary release of COOOL (CWP
 * Object-Oriented Optimization Library) and associated class
 * libraries.
 *
 * The COOOL library is a free software. You can do anything you want
 * with it including make a fortune.  However, neither the authors,
 * the Center for Wave Phenomena, nor anyone else you can think of
 * makes any guarantees about anything in this package or any aspect
 * of its functionality.
 *
 * Since you've got the source code you can also modify the
 * library to suit your own purposes. We would appreciate it
 * if the headers that identify the authors are kept in the
 * source code.
 *
 * ==================================
 * author:  H. Lydia Deng, 06/17/96
 * ==================================
 *
 */
//===========================================================================


#ifndef BFGS_H
#define BFGS_H

#include "ReClaM/Optimizer.h"

//===========================================================================
/*!
 *  \brief Offers methods to use the Broyden-Fletcher-Goldfarb-Shanno
 *         algorithm for the optimization of models.
 *
 *  Given the error function \f$E\f$, that shall be optimized,
 *  this function will be square approximated around the last
 *  given search point \f$w^{(t)}\f$ by
 *  \f[
 *  E(w) = E(w^{(t)}) + (w - w^{(t)})^{T} \frac{\partial E}{\partial w}
 *  \arrowvert_{w^{(t)}} + \frac{1}{2}(w - w^{(t)})^{T} H \arrowvert_{w^{(t)}}
 *  (w - w^{(t)})
 *  \f]
 *  where \f$H \arrowvert_{w^{(t)}}\f$ is the Hessian matrix evaluated
 *  at \f$w^{(t)}\f$ with the entries
 *  \f[
 *  h_{ij} \arrowvert_{w^{(t)}} = \frac{\partial^2 E}
 *  {\partial w_i \partial w_j} \arrowvert_{w^{(t)}}
 *  \f]
 *  The derivation is then
 *  \f[
 *  \nabla E \arrowvert_w \approx \nabla E \arrowvert_{w^{(t)}}
 *  + H \arrowvert_{w^{(t)}} (w - w^{(t)})
 *  \f]
 *  At the optimum point \f$w^{opt}\f$ holds \f$ \nabla E
 *  \arrowvert_w^{(t)} = 0\f$, so the Newton step (the BFGS algorithm
 *  is a quasi-Newton method) is
 *  \f[
 *  w^{opt} - w^{(t)} \approx - H^{-1} \arrowvert_{w^{(t)}}
 *  \nabla E \arrowvert_{w^{(t)}}
 *  \f]
 *  The new search point is then directly determined from this step with this
 *  method used iteratively, because of the approximation. <br>
 *  The inverse Hessian matrix is iteratively estimated with possibility
 *  to restrict the number of iterations. <br> <br>
 *  For a more detailed description of the BFGS algorithm refer to
 *  "Neural Networks for Pattern Recognition" by Christopher M. Bishop
 *  or "Numerical Recipes in C" by Press, Teukolsky, Vetterling and
 *  Flannery.
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 */

class BFGS : public Optimizer
{
public:

	//! Used to specify the type of line search algorithm
	//! that is used by the BFGS algorithm: <br>
	//! "Linmin" is the original line search algorithm,
	//! "DLinmin" is a line search algorithm that additionally uses
	//! derivative information and "Cblnsrch" is a cubic line
	//! search.
	enum LineSearch { Dlinmin, Linmin, Cblnsrch };

	//! Updates the weights of the network by using
	//! the BFGS optimization algorithm.

	void bfgs(Model& model,
			  ErrorFunction& error,
			  const Array<double> &i,
			  const Array<double> &t,
			  double    gtol,
			  unsigned& iter,
			  double&   fret,
			  unsigned  iterMax);
//===========================================================================
	/*!
	 *  \brief Initializes some internal variables used by the BFGS algorithm.
	 *
	 *  The given parameters are used to initialize some variables
	 *  used by the BFGS and the line search algorithms and to allocate
	 *  memory for internal structures.
	 *
	 *      \param model	  Type of model that uses BFGS.
	 *      \param ls         Type of LineSearch algorithm, that is used
	 *                        by the BFGS algorithm. Possible values:
	 *                        "Dlinmin" (default value), "Linmin", "Cblnsrch"
	 *      \param ax         Initial left bracket for the line search
	 *                        algorithms, the default bracket is "0".
	 *      \param bx         Initial right bracket for the line search
	 *                        algorithms, the default bracket is "1".
	 *      \param lambda     The initial value for the searched \f$\lambda\f$,
	 *                        that will minimize the function used by the
	 *                        line search algorithm. The default value is "0.25".
	 *      \return none
	 *
	 *  \author  C. Igel
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */

	void init(Model& model);

	void initBfgs(Model& model, LineSearch ls = Dlinmin,
				  double ax = 0., double bx = 1., double lambda = 0.25);

	//! Updates the weights of the network by using
	//! the BFGS optimization algorithm, that was previously initialized
	//! and returns the minimum value by reference.

//===========================================================================
	/*!
	 *  \brief Updates the weights of the network by using
	 *         the BFGS optimization algorithm, that was previously initialized.
	 *
	 *  The given input patterns \em in and their corresponding target
	 *  values \em target are used to calculate the error of the currently
	 *  used model. The resulting error function is then minimized by the
	 *  BFGS algorithm, the weights of the network are updated. <br>
	 *  This method needs far less parameters than the other BFGS method,
	 *  because it is guessed, that one of the #initBfgs methods was
	 *  called before.
	 *
	 *      \param in         Input patterns for the currently used model.
	 *      \param target     Target values corresponding to the input patterns.
	 *      \return none
	 *
	 *  \author  C. Igel
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */

	double optimize(
		Model& model,
		ErrorFunction& errorfunction,
		const Array<double>& input,
		const Array<double>& target);
	double optimize(
		Model& model,
		ErrorFunction& errorfunction,
		const Array<double>& input,
		const Array<double>& target,
		double gtol,
		int iterMax);

	void bfgs(Model& model,
			  ErrorFunction& errorfunction,
			  const Array<double> &in,
			  const Array<double> &target,
			  double& fret);

	void bfgs(Model& model,
			  ErrorFunction& errorfunction,
			  const Array<double> &in,
			  const Array<double> &target)
	{
		double fret;
		bfgs(model, errorfunction, in, target, fret);
	}
//===========================================================================
	/*!
	 *  \brief Sets the initial value for the left bracket used by
	 *         the line search algorithms.
	 *
	 *  The line search algorithms used by the BFGS algorithm searches for
	 *  a minimum value of the error function \f$E\f$ along a search
	 *  direction \f$d\f$, i.e. it searches for a scalar \f$\lambda\f$,
	 *  that minimizes
	 *
	 *  \f$
	 *  E(\lambda) = E(w + \lambda d)
	 *  \f$
	 *
	 *  The minimum value searched is iteratively bracketed by two points.
	 *  The left point (i.e. the value \f$a\f$ that is less than the minimum)
	 *  can be set by this method.
	 *
	 *      \param x The initial value for the left bracket.
	 *      \return none
	 *
	 *  \author  C. Igel
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void setAx(double x)
	{
		LS_ax = x;
	}

//===========================================================================
	/*!
	 *  \brief Sets the initial value for the right bracket used by
	 *         the line search algorithms.
	 *
	 *  The line search algorithms used by the BFGS algorithm searches for
	 *  a minimum value of the error function \f$E\f$ along a search
	 *  direction \f$d\f$, i.e. it searches for a scalar \f$\lambda\f$,
	 *  that minimizes
	 *
	 *  \f$
	 *  E(\lambda) = E(w + \lambda d)
	 *  \f$
	 *
	 *  The minimum value searched is iteratively bracketed by two points.
	 *  The right point (i.e. the value \f$a\f$ that is greater than the minimum)
	 *  can be set by this method.
	 *
	 *      \param x The initial value for the right bracket.
	 *      \return none
	 *
	 *  \author  C. Igel
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void setBx(double x)
	{
		LS_bx = x;
	}


//===========================================================================
	/*!
	 *  \brief Sets the initial value for \f$\lambda\f$ value used by
	 *         the line search algorithms.
	 *
	 *  The line search algorithms used by the BFGS algorithm searches for
	 *  a minimum value of the error function \f$E\f$ along a search
	 *  direction \f$d\f$, i.e. it searches for a scalar \f$\lambda\f$,
	 *  that minimizes
	 *
	 *  \f$
	 *  E(\lambda) = E(w + \lambda d)
	 *  \f$
	 *
	 *
	 *  \f$\lambda\f$ can be set by this method.
	 *
	 *      \param x The initial value for \f$\lambda\f$.
	 *      \return none
	 *
	 *  \author  C. Igel
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void setLambda(double x)
	{
		LS_lambda = x;
	}

	void reset();
private:
//  baustelle

	void dlinmin(Model& model,

				 ErrorFunction& errorfunction,

				 Array<double>& dedw,

				 Array< double >& p,

				 const Array< double >& xi,

				 double&                fret,

				 const Array< double >& in,

				 const Array< double >& target);

	void linmin(Model& model,

				ErrorFunction& errorfunction,

				Array<double>& dedw,

				Array< double >& p,

				const Array< double >& xi,

				double&                fret,

				const Array< double >& in,

				const Array< double >& target);

	void cblnsrch(Model& model,

				  ErrorFunction& errorfunction,

				  double fold,           // evaluation at current solution

				  Array< double >& g,    // current gradient

				  Array< double >& p,    // search direction

				  double& fret,

				  const Array< double >&in,

				  const Array< double >&target);

	LineSearch    lineSearchType;

	bool		 firstIteration;

	double	 FRET;

	double        LS_lambda;

	double        LS_ax;

	double        LS_bx;

	unsigned      BFGS_n;

	double        BFGS_d;

	double        BFGS_scale;

	Array<double> BFGS_s;

	Array<double> DEDW;

	Array<double> BFGS_g0;

	Array<double> BFGS_g1;

	Array<double> BFGS_p1;

	Array<double> BFGS_gamma;

	Array<double> BFGS_delta;

	Array<double> BFGS_p;

	// the approximated Hessian matrix
	Array<double> H;

};

#endif

