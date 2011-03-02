//===========================================================================
/*!
 *  \file Rprop.h
 *
 *  \brief This file offers classes to use the
 *         Resilient-Backpropagation-algorithm for the optimization of the
 *         adaptive parameters of a network.
 *
 *  Four classes with four versions of the algorithm are included in this
 *  file: <br>
 *
 *  <ul>
 *      <li>RpropPlus:   The Rprop algorithm with weight-backtracking.</li>
 *      <li>RpropMinus:  The Rprop algorithm without weight-backtracking.</li>
 *      <li>IRpropPlus:  An improved Rprop algorithm with
 *                       weight-backtracking.</li>
 *      <li>IRpropMinus: An improved Rprop algorithm without
 *                       weight-backtracking.</li>
 *  </ul>
 *
 *  \author  C. Igel, M. H&uuml;sken
 *  \date    1999
 *
 *  \par Copyright (c) 1999-2000:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  \par Project:
 *      ReClaM
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
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
 */
//===========================================================================

#define NOMINMAX

#ifndef RPROP_H
#define RPROP_H

#include <SharkDefs.h>
#include <Array/Array.h>
#include <Array/ArrayOp.h>
#include <ReClaM/Optimizer.h>


//===========================================================================
/*!
 *  \brief This class offers methods for the usage of the
 *         Resilient-Backpropagation-algorithm with weight-backtracking.
 *
 *  The Rprop algorithm is an improvement of the algorithms with adaptive
 *  learning rates (as the Adaptive Backpropagation algorithm by Silva
 *  and Ameida, please see AdpBP.h for a description of the
 *  working of such an algorithm), that uses increments for the update
 *  of the weights, that are independant from the absolute partial
 *  derivatives. This makes sense, because large flat regions
 *  in the search space (plateaus) lead to small absolute partial
 *  derivatives and so the increments are chosen small, but the increments
 *  should be large to skip the plateau. In contrast, the absolute partial
 *  derivatives are very large at the "slopes" of very "narrow canyons",
 *  which leads to large increments that will skip the minimum lying
 *  at the bottom of the canyon, but it would make more sense to
 *  chose small increments to hit the minimum. <br>
 *  So, the Rprop algorithm only uses the signs of the partial derivatives
 *  and not the absolute values to adapt the parameters. <br>
 *  Instead of individual learning rates, it uses the parameter
 *  \f$\Delta_i^{(t)}\f$ for weight \f$w_i,\ i = 1, \dots, n\f$ in
 *  iteration \f$t\f$, where the parameter will be adapted before the
 *  change of the weights: <br>
 *
 *  \f$
 *  \Delta_i^{(t)} = \Bigg\{
 *  \begin{array}{ll}
 *  min( \eta^+ \cdot \Delta_i^{(t-1)}, \Delta_{max} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} > 0 \\
 *  max( \eta^- \cdot \Delta_i^{(t-1)}, \Delta_{min} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0 \\
 *  \Delta_i^{(t-1)}, & \mbox{otherwise}
 *  \end{array}
 *  \f$
 *
 *  The parameters \f$\eta^+ > 1\f$ and \f$0 < \eta^- < 1\f$ control
 *  the speed of the adaptation. To stabilize the increments, they are
 *  restricted to the interval \f$[\Delta_{min}, \Delta_{max}]\f$. <br>
 *  After the adaptation of the \f$\Delta_i\f$ the update for the
 *  weights will be calculated as
 *
 *  \f$
 *  \Delta w_i^{(t)} := - \mbox{sign}
 *  \left( \frac{\partial E^{(t)}}{\partial w_i}\right) \cdot \Delta_i^{(t)}
 *  \f$
 *
 *  Furthermore, weight-backtracking will take place to increase the
 *  stability of the method, i.e.
 *  if \f$\frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0\f$ then
 *  \f$\Delta w_i^{(t)} := - \Delta w_i^{(t-1)};
 *  \frac{\partial E^{(t)}}{\partial w_i} := 0\f$, where
 *  the assignment of zero to the partial derivative of the error
 *  leads to a freezing of the increment in the next iteration.
 *
 *  For further information about the algorithm, please refer to: <br>
 *
 *  Martin Riedmiller and Heinrich Braun, <br>
 *  "A Direct Adaptive Method for Faster Backpropagation Learning: The
 *  RPROP Algorithm". <br>
 *  In "Proceedings of the IEEE International Conference on Neural Networks",
 *  pp. 586-591, <br>
 *  Published by IEEE Press in 1993
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 *
 */
class RpropPlus : public Optimizer
{
public:

	//===========================================================================
	/*!
	*  \brief Initializes the optimizer with default parameters.
	*
	*  Initializes the optimizer's parameters with the following defaults:
	*  <ul>
	*    <li>Increase factor \f$\eta^+ = 1.2\f$
	*    <li>Decrease factor \f$\eta^- = 0.5\f$
	*    <li>Upper increment limit = 1e100
	*    <li>Lower increment limit = 0.0
	*    <li>Initial value for \f$\Delta = 0.01\f$
	*  </ul> 
	*
	*      \param  model Model to be optimized.
	*      \return None.
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
	void init(Model& model)
	{
		initUserDefined(model);
	}

	//===========================================================================
	/*!
	*  \brief Prepares the Rprop algorithm for the given network.
	*
	*  Internal variables of the class instance are initialized and memory
	*  for used structures adapted to the used network. An initial value
	*  for the parameter \f$\Delta\f$ is assigned to all weights of the network.
	*
	*   \param  model   Model to be optimized.
	*   \param  npP     The increase factor \f$\eta^+\f$, set to 1.2 by default.
	*   \param  nmP     The decrease factor \f$\eta^-\f$, set to 0.5 by default.
	*   \param  dMaxP   The upper limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 1e100 by default.
	*   \param  dMinP   The lower limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 0.0 by default.
	*   \param  delta0P Initial value for the parameter \f$\Delta\f$, set to 0.01 by default.
	*   \return none
	*
	*   \author  C. Igel
	*   \date    1999
	*
	*   \par Changes
	*       none
	*
	*   \par Status
	*       stable
	*
	*
	*/
	void initUserDefined(Model& model, double npP = 1.2, double nmP = 0.5, double dMaxP = 1e100, double dMinP = 0.0, double delta0P = 0.01)
	{
		np      = npP;
		nm      = nmP;
		dMax    = dMaxP;
		dMin    = dMinP;

		deltaw.resize(model.getParameterDimension());
		delta.resize(model.getParameterDimension());
		dedwOld.resize(model.getParameterDimension());

		delta   = delta0P;
		deltaw  = 0;
		dedwOld = 0;
	}

	//===========================================================================
	/*!
	*  \brief Rprop algorithm initialization
	*
	*	\par
	*	This initialization allows for the definition of
	*	user defined initial step widths for each coordinate
	*	individually.
	*
	*   \param  model   Model to be optimized.
	*   \param  delta0P Initial values for the parameters \f$\Delta\f$
	*   \param  npP     The increase factor \f$\eta^+\f$, set to 1.2 by default.
	*   \param  nmP     The decrease factor \f$\eta^-\f$, set to 0.5 by default.
	*   \param  dMaxP   The upper limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 1e100 by default.
	*   \param  dMinP   The lower limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 0.0 by default.
	*   \return none
	*
	*   \author  T. Glasmachers
	*   \date    2006
	*/
	void initUserDefined(Model& model, const Array<double>& delta0P, double npP = 1.2, double nmP = 0.5, double dMaxP = 1e100, double dMinP = 0.0)
	{
		np      = npP;
		nm      = nmP;
		dMax    = dMaxP;
		dMin    = dMinP;

		deltaw.resize(model.getParameterDimension());
		delta.resize(model.getParameterDimension());
		dedwOld.resize(model.getParameterDimension());

		delta   = delta0P;
		deltaw  = 0.0;
		dedwOld = 0.0;
	}

	//===========================================================================
	/*!
	*  \brief Performs a run of the Rprop algorithm.
	*
	*  The error \f$E^{(t)}\f$ of the used network for the current iteration
	*  \f$t\f$ is calculated and the values of the weights \f$w_i,\ i = 1, 
	*  \dots, n\f$ and the parameters \f$\Delta_i\f$ are 
	*  adapted depending on this error.
	*
	*      \param  model            Model to be optimized.
	*      \param  errorfunction    Error function to be used for the optimization.
	*      \param  input            Input vector for the model. 
	*      \param  target           Corresponding targets to the given inputs.
	*      \return The current error.
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
	*
	*/
	double optimize(
		Model& model,
		ErrorFunction& errorfunction,
		const Array<double>& input,
		const Array<double>& target)
	{
		Array<double> dedw(model.getParameterDimension());
		double currentError = errorfunction.errorDerivative(model, input, target, dedw);

		for (unsigned i = 0; i < model.getParameterDimension(); i++)
		{
			double p = model.getParameter(i);
			if (dedw(i) * dedwOld(i) > 0)
			{
				delta(i) = Shark::min(dMax, np * delta(i));
				deltaw(i) = delta(i) * -sgn(dedw(i));
				model.setParameter(i, p + deltaw(i));
				dedwOld(i) = dedw(i);
			}
			else if (dedw(i) * dedwOld(i) < 0)
			{
				delta(i) = Shark::max(dMin, nm * delta(i));
				model.setParameter(i, p - deltaw(i));
				dedwOld(i) = 0;
			}
			else
			{
				deltaw(i) = delta(i) * -sgn(dedw(i));
				model.setParameter(i, p + deltaw(i));
				dedwOld(i) = dedw(i);
			}
			if (! model.isFeasible())
			{
				model.setParameter(i, p);
				delta(i) *= nm;
				dedwOld(i) = 0.0;
			}
		}
		return currentError;
	}

	//! return the maximal step size component
	double maxDelta() const
	{
		return maxElement(delta);
	}

protected:

	//===========================================================================
	/*!
	*  \brief Determines the sign of "x".
	*
	*  \param x The value of which the sign shall be determined.
	*  \return "-1", if the sign of \em x is negative, "0" otherwise.
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
	int sgn(double x)
	{
		if (x > 0) return 1; if (x < 0) return -1; return 0;
	}

	//! The increase factor \f$ \eta^+ \f$, set to 1.2 by default.
	double np;

	//! The decrease factor \f$ \eta^- \f$, set to 0.5 by default.
	double nm;

	//! The upper limit of the increments \f$ \Delta w_i^{(t)} \f$, set to 1e100 by default.
	double dMax;

	//! The lower limit of the increments \f$ \Delta w_i^{(t)} \f$, set to 0.0 by default.
	double dMin;

	//! The final update values for all weights.
	Array<double> deltaw;

	//! The last error gradient.
	Array<double> dedwOld;

	//! The absolute update values (increment) for all weights.
	Array<double> delta;

};

//===========================================================================
/*!
 *  \brief This class offers methods for the usage of the
 *         Resilient-Backpropagation-algorithm without weight-backtracking.
 *
 *  The Rprop algorithm is an improvement of the algorithms with adaptive
 *  learning rates (as the Adaptive Backpropagation algorithm by Silva
 *  and Ameida, please see AdpBP.h for a description of the
 *  working of such an algorithm), that uses increments for the update
 *  of the weights, that are independant from the absolute partial
 *  derivatives. This makes sense, because large flat regions
 *  in the search space (plateaus) lead to small absolute partial
 *  derivatives and so the increments are chosen small, but the increments
 *  should be large to skip the plateau. In contrast, the absolute partial
 *  derivatives are very large at the "slopes" of very "narrow canyons",
 *  which leads to large increments that will skip the minimum lying
 *  at the bottom of the canyon, but it would make more sense to
 *  chose small increments to hit the minimum. <br>
 *  So, the Rprop algorithm only uses the signs of the partial derivatives
 *  and not the absolute values to adapt the parameters. <br>
 *  Instead of individual learning rates, it uses the parameter
 *  \f$\Delta_i^{(t)}\f$ for weight \f$w_i,\ i = 1, \dots, n\f$ in
 *  iteration \f$t\f$, where the parameter will be adapted before the
 *  change of the weights: <br>
 *
 *  \f$
 *  \Delta_i^{(t)} = \Bigg\{
 *  \begin{array}{ll}
 *  min( \eta^+ \cdot \Delta_i^{(t-1)}, \Delta_{max} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} > 0 \\
 *  max( \eta^- \cdot \Delta_i^{(t-1)}, \Delta_{min} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0 \\
 *  \Delta_i^{(t-1)}, & \mbox{otherwise}
 *  \end{array}
 *  \f$
 *
 *  The parameters \f$\eta^+ > 1\f$ and \f$0 < \eta^- < 1\f$ control
 *  the speed of the adaptation. To stabilize the increments, they are
 *  restricted to the interval \f$[\Delta_{min}, \Delta_{max}]\f$. <br>
 *  After the adaptation of the \f$\Delta_i\f$ the update for the
 *  weights will be calculated as
 *
 *  \f$
 *  \Delta w_i^{(t)} := - \mbox{sign}
 *  \left( \frac{\partial E^{(t)}}{\partial w_i}\right) \cdot \Delta_i^{(t)}
 *  \f$
 *
 *  For further information about the algorithm, please refer to: <br>
 *
 *  Martin Riedmiller, <br>
 *  "Advanced Supervised Learning in Multi-layer Perceptrons -
 *  From Backpropagation to Adaptive Learning Algorithms". <br>
 *  In "International Journal of Computer Standards and Interfaces", volume 16,
 *  no. 5, 1994, pp. 265-278 <br>
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 *
 */
class RpropMinus : public Optimizer
{
public:

	//===========================================================================
	/*!
	*  \brief Initializes the optimizer with default parameters.
	*
	*  Initializes the optimizer's parameters with the following defaults:
	   *  <ul>
	   *    <li>Increase factor \f$\eta^+ = 1.2\f$
	   *    <li>Decrease factor \f$\eta^- = 0.5\f$
	   *    <li>Upper increment limit = 1e100
	   *    <li>Lower increment limit = 0.0
	   *    <li>Initial value for \f$\Delta = 0.01\f$
	   *  </ul> 
	   *
	   *      \param  model Model to be optimized.
	*      \return None.
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
	void init(Model& model)
	{
		initUserDefined(model);
	}

	//===========================================================================
	/*!
	*  \brief Prepares the Rprop algorithm for the given network.
	*
	*  Internal variables of the class instance are initialized and memory
	*  for used structures adapted to the used network. An initial value
	*  for the parameter \f$\Delta\f$ is assigned to all weights of the network.
	*
	*   \param  model   Model to be optimized.
	*   \param  npP     The increase factor \f$\eta^+\f$, set to 1.2 by default.
	*   \param  nmP     The decrease factor \f$\eta^-\f$, set to 0.5 by default.
	*   \param  dMaxP   The upper limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 1e100 by default.
	*   \param  dMinP   The lower limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 0.0 by default.
	*   \param  delta0P Initial value for the parameter \f$\Delta\f$, set to 0.01 by default.
	*   \return none
	*
	*   \author  C. Igel
	*   \date    1999
	*
	*   \par Changes
	*       none
	*
	*   \par Status
	*       stable
	*
	*
	*/
	void initUserDefined(Model& model, double npP = 1.2, double nmP = 0.5, double dMaxP = 1e100, double dMinP = 0.0, double delta0P = 0.01)
	{
		np      = npP;
		nm      = nmP;
		dMax    = dMaxP;
		dMin    = dMinP;

		deltaw.resize(model.getParameterDimension());
		delta.resize(model.getParameterDimension());
		dedwOld.resize(model.getParameterDimension());

		delta   = delta0P;
		deltaw  = 0;
		dedwOld = 0;
	}

	//===========================================================================
	/*!
	*  \brief Rprop algorithm initialization
	*
	*	\par
	*	This initialization allows for the definition of
	*	user defined initial step widths for each coordinate
	*	individually.
	*
	*   \param  model   Model to be optimized.
	*   \param  delta0P Initial values for the parameters \f$\Delta\f$
	*   \param  npP     The increase factor \f$\eta^+\f$, set to 1.2 by default.
	*   \param  nmP     The decrease factor \f$\eta^-\f$, set to 0.5 by default.
	*   \param  dMaxP   The upper limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 1e100 by default.
	*   \param  dMinP   The lower limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 0.0 by default.
	*   \return none
	*
	*   \author  T. Glasmachers
	*   \date    2006
	*/
	void initUserDefined(Model& model, const Array<double>& delta0P, double npP = 1.2, double nmP = 0.5, double dMaxP = 1e100, double dMinP = 0.0)
	{
		np      = npP;
		nm      = nmP;
		dMax    = dMaxP;
		dMin    = dMinP;

		deltaw.resize(model.getParameterDimension());
		delta.resize(model.getParameterDimension());
		dedwOld.resize(model.getParameterDimension());

		delta   = delta0P;
		deltaw  = 0.0;
		dedwOld = 0.0;
	}

	//===========================================================================
	/*!
	*  \brief Performs a run of the Rprop algorithm.
	*
	*  The error \f$E^{(t)}\f$ of the used network for the current iteration
	*  \f$t\f$ is calculated and the values of the weights \f$w_i,\ i = 1, 
	*  \dots, n\f$ and the parameters \f$\Delta_i\f$ are 
	*  adapted depending on this error.
	*
	   *      \param  model            Model to be optimized.
	*      \param  errorfunction    Error function to be used for the optimization.
	*      \param  input            Input vector for the model. 
	*      \param  target           Corresponding targets to the given inputs.
	*      \return The current error.
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
	*
	*/
	double optimize(
		Model& model,
		ErrorFunction& errorfunction,
		const Array<double>& input,
		const Array<double>& target)
	{
		Array<double> dedw(model.getParameterDimension());
		double currentError = errorfunction.errorDerivative(model, input, target, dedw); //derror(in, out,false);
		for (unsigned i = 0; i < model.getParameterDimension(); i++)
		{
			double p = model.getParameter(i);
			if (dedw(i) * dedwOld(i) > 0)
			{
				delta(i) = Shark::min(dMax, np * delta(i));
			}
			else if (dedw(i) * dedwOld(i) < 0)
			{
				delta(i) = Shark::max(dMin, nm * delta(i));
			}
			else
			{
				; // void
			}
			model.setParameter(i, p + delta(i) * -sgn(dedw(i))); //w(i) += delta(i) * -sgn(dedw(i));
			if (! model.isFeasible())
			{
				model.setParameter(i, p);
				delta(i) *= nm;
				dedwOld(i) = 0.0;
			}
			else
			{
				dedwOld(i) = dedw(i);
			}
		}
		return currentError;
	}

	//! return the maximal step size component
	double maxDelta() const
	{
		return maxElement(delta);
	}

protected:

	//===========================================================================
	/*!
	*  \brief Determines the sign of "x".
	*
	*  \param x The value of which the sign shall be determined.
	*  \return "-1", if the sign of \em x is negative, "0" otherwise.
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
	int sgn(double x)
	{
		if (x > 0) return 1; if (x < 0) return -1; return 0;
	}

	//! The increase factor \f$ \eta^+ \f$, set to 1.2 by default.
	double np;

	//! The decrease factor \f$ \eta^- \f$, set to 0.5 by default.
	double nm;

	//! The upper limit of the increments \f$ \Delta w_i^{(t)} \f$, set to 1e100 by default.
	double dMax;

	//! The lower limit of the increments \f$ \Delta w_i^{(t)} \f$, set to 0.0 by default.
	double dMin;

	//! The final update values for all weights.
	Array<double> deltaw;

	//! The last error gradient.
	Array<double> dedwOld;

	//! The absolute update values (increment) for all weights.
	Array<double> delta;

};


//===========================================================================
/*!
 *  \brief This class offers methods for the usage of the improved
 *         Resilient-Backpropagation-algorithm with weight-backtracking.
 *
 *  The Rprop algorithm is an improvement of the algorithms with adaptive
 *  learning rates (as the Adaptive Backpropagation algorithm by Silva
 *  and Ameida, please see AdpBP.h for a description of the
 *  working of such an algorithm), that uses increments for the update
 *  of the weights, that are independant from the absolute partial
 *  derivatives. This makes sense, because large flat regions
 *  in the search space (plateaus) lead to small absolute partial
 *  derivatives and so the increments are chosen small, but the increments
 *  should be large to skip the plateau. In contrast, the absolute partial
 *  derivatives are very large at the "slopes" of very "narrow canyons",
 *  which leads to large increments that will skip the minimum lying
 *  at the bottom of the canyon, but it would make more sense to
 *  chose small increments to hit the minimum. <br>
 *  So, the Rprop algorithm only uses the signs of the partial derivatives
 *  and not the absolute values to adapt the parameters. <br>
 *  Instead of individual learning rates, it uses the parameter
 *  \f$\Delta_i^{(t)}\f$ for weight \f$w_i,\ i = 1, \dots, n\f$ in
 *  iteration \f$t\f$, where the parameter will be adapted before the
 *  change of the weights: <br>
 *
 *  \f$
 *  \Delta_i^{(t)} = \Bigg\{
 *  \begin{array}{ll}
 *  min( \eta^+ \cdot \Delta_i^{(t-1)}, \Delta_{max} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} > 0 \\
 *  max( \eta^- \cdot \Delta_i^{(t-1)}, \Delta_{min} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0 \\
 *  \Delta_i^{(t-1)}, & \mbox{otherwise}
 *  \end{array}
 *  \f$
 *
 *  The parameters \f$\eta^+ > 1\f$ and \f$0 < \eta^- < 1\f$ control
 *  the speed of the adaptation. To stabilize the increments, they are
 *  restricted to the interval \f$[\Delta_{min}, \Delta_{max}]\f$. <br>
 *  After the adaptation of the \f$\Delta_i\f$ the update for the
 *  weights will be calculated as
 *
 *  \f$
 *  \Delta w_i^{(t)} := - \mbox{sign}
 *  \left( \frac{\partial E^{(t)}}{\partial w_i}\right) \cdot \Delta_i^{(t)}
 *  \f$
 *
 *  Furthermore, weight-backtracking will take place to increase the
 *  stability of the method. In contrast to the original Rprop algorithm
 *  with weight-backtracking (see RpropPlus) this weight-backtracking
 *  is improved by additionally taken the error of the last iteration
 *  \f$t - 1\f$ into account. <br>
 *  The idea of this modification is, that a change of the sign of the
 *  partial derivation \f$\frac{\partial E}{\partial w_i}\f$
 *  only states, that a minimum was skipped and not, whether this step
 *  lead to an approach to the minimum or not. <br>
 *  By using the old error value the improved weight-backtracking only
 *  undoes changes, when the error has increased and only the parameters
 *  \f$w_i\f$ are reset to the old values, where a sign change of
 *  \f$\frac{\partial E}{\partial w_i}\f$ has taken place. <br>
 *  So the new weight-backtracking rule is: <br>
 *
 *  \f$
 *  \mbox{if\ } \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0 \mbox{\ then} \{
 *  \f$
 *
 *  \f$
 *  \begin{array}{lll}
 *   \Delta w_i^{(t)} = \bigg\{ &
 *   - \Delta w_i^{(t-1)}, & \mbox{if\ } E^{(t)} > E^{(t - 1)} \\
 *   & 0, & otherwise \\
 *  \frac{\partial E^{(t)}}{\partial w_i} := 0
 *  \end{array}
 *  \f$
 *
 *  \f$\}\f$
 *
 *  , where the assignment of zero to the partial derivative of the error
 *  leads to a freezing of the increment in the next iteration. <br>
 *
 *  This modification of the weight backtracking leads to a better
 *  optimization on artifical, paraboloidal error surfaces. <br>
 *
 *  For further information about the algorithm, please refer to: <br>
 *
 *  Christian Igel and Michael H&uuml;sken, <br>
 *  "Empirical Evaluation of the Improved Rprop Learning Algorithm". <br>
 *  In Neurocomputing Journal, 2002, in press <br>
 *
 *  \author  C. Igel, M. H&uuml;sken
 *  \date    1999
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 *
 */
class IRpropPlus : public Optimizer
{
public:

	//===========================================================================
	/*!
	*  \brief Initializes the optimizer with default parameters.
	*
	*  Initializes the optimizer's parameters with the following defaults:
	   *  <ul>
	   *    <li>Increase factor \f$\eta^+ = 1.2\f$
	   *    <li>Decrease factor \f$\eta^- = 0.5\f$
	   *    <li>Upper increment limit = 1e100
	   *    <li>Lower increment limit = 0.0
	   *    <li>Initial value for \f$\Delta = 0.01\f$
	   *  </ul> 
	   *
	   *      \param  model Model to be optimized.
	*      \return None.
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
	void init(Model& model)
	{
		initUserDefined(model);
	}

	//===========================================================================
	/*!
	*  \brief Prepares the Rprop algorithm for the given network.
	*
	*  Internal variables of the class instance are initialized and memory
	*  for used structures adapted to the used network. An initial value
	*  for the parameter \f$\Delta\f$ is assigned to all weights of the network.
	*
	*   \param  model   Model to be optimized.
	*   \param  npP     The increase factor \f$\eta^+\f$, set to 1.2 by default.
	*   \param  nmP     The decrease factor \f$\eta^-\f$, set to 0.5 by default.
	*   \param  dMaxP   The upper limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 1e100 by default.
	*   \param  dMinP   The lower limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 0.0 by default.
	*   \param  delta0P Initial value for the parameter \f$\Delta\f$, set to 0.01 by default.
	*   \return none
	*
	*   \author  C. Igel
	*   \date    1999
	*
	*   \par Changes
	*       none
	*
	*   \par Status
	*       stable
	*
	*
	*/
	void initUserDefined(Model& model, double npP = 1.2, double nmP = 0.5, double dMaxP = 1e100, double dMinP = 0.0, double delta0P = 0.01)
	{
		np      = npP;
		nm      = nmP;
		dMax    = dMaxP;
		dMin    = dMinP;

		deltaw.resize(model.getParameterDimension());
		delta.resize(model.getParameterDimension());
		dedwOld.resize(model.getParameterDimension());

		delta   = delta0P;
		deltaw  = 0;
		dedwOld = 0;

		oldError   = MAXDOUBLE;
	}

	//===========================================================================
	/*!
	*  \brief Rprop algorithm initialization
	*
	*	\par
	*	This initialization allows for the definition of
	*	user defined initial step widths for each coordinate
	*	individually.
	*
	*   \param  model   Model to be optimized.
	*   \param  delta0P Initial values for the parameters \f$\Delta\f$
	*   \param  npP     The increase factor \f$\eta^+\f$, set to 1.2 by default.
	*   \param  nmP     The decrease factor \f$\eta^-\f$, set to 0.5 by default.
	*   \param  dMaxP   The upper limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 1e100 by default.
	*   \param  dMinP   The lower limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 0.0 by default.
	*   \return none
	*
	*   \author  T. Glasmachers
	*   \date    2006
	*/
	void initUserDefined(Model& model, const Array<double>& delta0P, double npP = 1.2, double nmP = 0.5, double dMaxP = 1e100, double dMinP = 0.0)
	{
		np      = npP;
		nm      = nmP;
		dMax    = dMaxP;
		dMin    = dMinP;

		deltaw.resize(model.getParameterDimension());
		delta.resize(model.getParameterDimension());
		dedwOld.resize(model.getParameterDimension());

		delta   = delta0P;
		deltaw  = 0.0;
		dedwOld = 0.0;

		oldError   = MAXDOUBLE;
	}

	//===========================================================================
	/*!
	*  \brief Performs a run of the Rprop algorithm.
	*
	*  The error \f$E^{(t)}\f$ of the used network for the current iteration
	*  \f$t\f$ is calculated and the values of the weights \f$w_i,\ i = 1, 
	*  \dots, n\f$ and the parameters \f$\Delta_i\f$ are 
	*  adapted depending on this error.
	*
	*      \param  model            Model to be optimized.
	*      \param  errorfunction    Error function to be used for the optimization.
	*      \param  input            Input vector for the model. 
	*      \param  target           Corresponding targets to the given inputs.
	*      \return The current error.
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
	*
	*/
	double optimize(
		Model& model,
		ErrorFunction& errorfunction,
		const Array<double>& input,
		const Array<double>& target)
	{
		Array<double> dedw(model.getParameterDimension());
		double currentError = errorfunction.errorDerivative(model, input, target, dedw);

		for (unsigned i = 0; i < model.getParameterDimension(); i++)
		{
			double p = model.getParameter(i);
			if (dedw(i) * dedwOld(i) > 0)
			{
				delta(i) = Shark::min(dMax, np * delta(i));
				deltaw(i) = delta(i) * -sgn(dedw(i));
				model.setParameter(i, p + deltaw(i)); //w(i) += deltaw(i);
				dedwOld(i) = dedw(i);
			}
			else if (dedw(i) * dedwOld(i) < 0)
			{
				delta(i) = Shark::max(dMin, nm * delta(i));
				if (oldError < currentError)
				{
					model.setParameter(i, p - deltaw(i)); //w(i) -= deltaw(i);
				}
				dedwOld(i) = 0;
			}
			else
			{
				deltaw(i) = delta(i) * -sgn(dedw(i));
				model.setParameter(i, p + deltaw(i)); //w(i) += deltaw(i);
				dedwOld(i) = dedw(i);
			}
			if (! model.isFeasible())
			{
				model.setParameter(i, p);
				delta(i) *= nm;
				dedwOld(i) = 0.0;
			}
		}
		oldError = currentError;
		return currentError;
	}

	//! return the maximal step size component
	double maxDelta() const
	{
		return maxElement(delta);
	}

protected:

	//===========================================================================
	/*!
	*  \brief Determines the sign of "x".
	*
	*  \param x The value of which the sign shall be determined.
	*  \return "-1", if the sign of \em x is negative, "0" otherwise.
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
	int sgn(double x)
	{
		if (x > 0) return 1; if (x < 0) return -1; return 0;
	}

	//! The increase factor \f$ \eta^+ \f$, set to 1.2 by default.
	double np;

	//! The decrease factor \f$ \eta^- \f$, set to 0.5 by default.
	double nm;

	//! The upper limit of the increments \f$ \Delta w_i^{(t)} \f$, set to 1e100 by default.
	double dMax;

	//! The lower limit of the increments \f$ \Delta w_i^{(t)} \f$, set to 0.0 by default.
	double dMin;

	//! The error of the last iteration.
	double oldError;

	//! The final update values for all weights.
	Array<double> deltaw;

	//! The last error gradient.
	Array<double> dedwOld;

	//! The absolute update values (increment) for all weights.
	Array<double> delta;

};

//===========================================================================
/*!
 *  \brief This class offers methods for the usage of the improved
 *         Resilient-Backpropagation-algorithm without weight-backtracking.
 *
 *  The Rprop algorithm is an improvement of the algorithms with adaptive
 *  learning rates (as the Adaptive Backpropagation algorithm by Silva
 *  and Ameida, please see AdpBP.h for a description of the
 *  working of such an algorithm), that uses increments for the update
 *  of the weights, that are independant from the absolute partial
 *  derivatives. This makes sense, because large flat regions
 *  in the search space (plateaus) lead to small absolute partial
 *  derivatives and so the increments are chosen small, but the increments
 *  should be large to skip the plateau. In contrast, the absolute partial
 *  derivatives are very large at the "slopes" of very "narrow canyons",
 *  which leads to large increments that will skip the minimum lying
 *  at the bottom of the canyon, but it would make more sense to
 *  chose small increments to hit the minimum. <br>
 *  So, the Rprop algorithm only uses the signs of the partial derivatives
 *  and not the absolute values to adapt the parameters. <br>
 *  Instead of individual learning rates, it uses the parameter
 *  \f$\Delta_i^{(t)}\f$ for weight \f$w_i,\ i = 1, \dots, n\f$ in
 *  iteration \f$t\f$, where the parameter will be adapted before the
 *  change of the weights. <br>
 *  As an improving modification, this algorithm
 *  adapts the "freezing" of the increment in the next iteration as
 *  usually only practiced by the Rprop algorithm with weight-backtracking
 *  (see RpropPlus), i.e. \f$\frac{\partial E^{(t)}}{\partial w_i} := 0\f$.
 *  Tests have shown a far more better optimization when using this
 *  modification. So the new adaptation rule of \f$\Delta\f$ is given
 *  as: <br>
 *
 *  \f$
 *  \Delta_i^{(t)} = \Bigg\{
 *  \begin{array}{ll}
 *  min( \eta^+ \cdot \Delta_i^{(t-1)}, \Delta_{max} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} > 0 \\
 *  max( \eta^- \cdot \Delta_i^{(t-1)}, \Delta_{min} );
 *  \frac{\partial E^{(t)}}{\partial w_i} := 0, & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0 \\
 *  \Delta_i^{(t-1)}, & \mbox{otherwise}
 *  \end{array}
 *  \f$
 *
 *  The parameters \f$\eta^+ > 1\f$ and \f$0 < \eta^- < 1\f$ control
 *  the speed of the adaptation. To stabilize the increments, they are
 *  restricted to the interval \f$[\Delta_{min}, \Delta_{max}]\f$. <br>
 *  After the adaptation of the \f$\Delta_i\f$ the update for the
 *  weights will be calculated as
 *
 *  \f$
 *  \Delta w_i^{(t)} := - \mbox{sign}
 *  \left( \frac{\partial E^{(t)}}{\partial w_i}\right) \cdot \Delta_i^{(t)}
 *  \f$
 *
 *  For further information about the algorithm, please refer to: <br>
 *
 *  Christian Igel and Michael H&uuml;sken, <br>
 *  "Empirical Evaluation of the Improved Rprop Learning Algorithm". <br>
 *  In Neurocomputing Journal, 2002, in press <br>
 *
 *
 *  \author  C. Igel, M. H&uuml;sken
 *  \date    1999
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 *
 */
class IRpropMinus : public Optimizer
{
public:

	//===========================================================================
	/*!
	*  \brief Initializes the optimizer with default parameters.
	*
	*  Initializes the optimizer's parameters with the following defaults:
	*  <ul>
	*    <li>Increase factor \f$\eta^+ = 1.2\f$
	*    <li>Decrease factor \f$\eta^- = 0.5\f$
	*    <li>Upper increment limit = 1e100
	*    <li>Lower increment limit = 0.0
	*    <li>Initial value for \f$\Delta = 0.01\f$
	*  </ul> 
	*
	*      \param  model Model to be optimized.
	*      \return None.
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
	void init(Model& model)
	{
		initUserDefined(model);
	}

	//===========================================================================
	/*!
	*  \brief Prepares the Rprop algorithm for the given network.
	*
	*  Internal variables of the class instance are initialized and memory
	*  for used structures adapted to the used network. An initial value
	*  for the parameter \f$\Delta\f$ is assigned to all weights of the network.
	*
	*   \param  model   Model to be optimized.
	*   \param  npP     The increase factor \f$\eta^+\f$, set to 1.2 by default.
	*   \param  nmP     The decrease factor \f$\eta^-\f$, set to 0.5 by default.
	*   \param  dMaxP   The upper limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 1e100 by default.
	*   \param  dMinP   The lower limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 0.0 by default.
	*   \param  delta0P Initial value for the parameter \f$\Delta\f$, set to 0.01 by default.
	*   \return none
	*
	*   \author  C. Igel
	*   \date    1999
	*
	*   \par Changes
	*       none
	*
	*   \par Status
	*       stable
	*
	*
	*/
	void initUserDefined(Model& model, double npP = 1.2, double nmP = 0.5, double dMaxP = 1e100, double dMinP = 0.0, double delta0P = 0.01)
	{
		np      = npP;
		nm      = nmP;
		dMax    = dMaxP;
		dMin    = dMinP;

		deltaw.resize(model.getParameterDimension());
		delta.resize(model.getParameterDimension());
		dedwOld.resize(model.getParameterDimension());

		delta   = delta0P;
		deltaw  = 0;
		dedwOld = 0;
	}

	//===========================================================================
	/*!
	*  \brief Rprop algorithm initialization
	*
	*	\par
	*	This initialization allows for the definition of
	*	user defined initial step widths for each coordinate
	*	individually.
	*
	*   \param  model   Model to be optimized.
	*   \param  delta0P Initial values for the parameters \f$\Delta\f$
	*   \param  npP     The increase factor \f$\eta^+\f$, set to 1.2 by default.
	*   \param  nmP     The decrease factor \f$\eta^-\f$, set to 0.5 by default.
	*   \param  dMaxP   The upper limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 1e100 by default.
	*   \param  dMinP   The lower limit of the increments \f$\Delta w_i^{(t)}\f$,
	*                   set to 0.0 by default.
	*   \return none
	*
	*   \author  T. Glasmachers
	*   \date    2006
	*/
	void initUserDefined(Model& model, const Array<double>& delta0P, double npP = 1.2, double nmP = 0.5, double dMaxP = 1e100, double dMinP = 0.0)
	{
		np      = npP;
		nm      = nmP;
		dMax    = dMaxP;
		dMin    = dMinP;

		deltaw.resize(model.getParameterDimension());
		delta.resize(model.getParameterDimension());
		dedwOld.resize(model.getParameterDimension());

		delta   = delta0P;
		deltaw  = 0.0;
		dedwOld = 0.0;
	}

	//===========================================================================
	/*!
	*  \brief Performs a run of the Rprop algorithm.
	*
	*  The error \f$E^{(t)}\f$ of the used network for the current iteration
	*  \f$t\f$ is calculated and the values of the weights \f$w_i,\ i = 1, 
	*  \dots, n\f$ and the parameters \f$\Delta_i\f$ are 
	*  adapted depending on this error.
	*
	*      \param  model            Model to be optimized.
	*      \param  errorfunction    Error function to be used for the optimization.
	*      \param  input            Input vector for the model. 
	*      \param  target           Corresponding targets to the given inputs.
	*      \return The current error.
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
	*
	*/
	double optimize(
		Model& model,
		ErrorFunction& errorfunction,
		const Array<double>& input,
		const Array<double>& target)
	{
		Array<double> dedw(model.getParameterDimension());
		double currentError = errorfunction.errorDerivative(model, input, target, dedw); //derror(in, out, false);

		for (unsigned i = 0; i < model.getParameterDimension(); i++)
		{
			double p = model.getParameter(i);
			if (dedw(i) * dedwOld(i) > 0)
			{
				delta(i) = Shark::min(dMax, np * delta(i));
				deltaw(i) = delta(i) * -sgn(dedw(i));
				model.setParameter(i, p + deltaw(i)); //w(i) += deltaw(i);
				dedwOld(i) = dedw(i);
			}
			else if (dedw(i) * dedwOld(i) < 0)
			{
				delta(i) = Shark::max(dMin, nm * delta(i));
				deltaw(i) = delta(i) * -sgn(dedw(i));
				model.setParameter(i, p + deltaw(i)); //w(i) += deltaw(i);
				dedwOld(i) = 0;
			}
			else
			{
				deltaw(i) = delta(i) * -sgn(dedw(i));
				model.setParameter(i, p + deltaw(i)); //w(i) += deltaw(i);
				dedwOld(i) = dedw(i);
			}
			if (! model.isFeasible())
			{
				model.setParameter(i, p);
				delta(i) *= nm;
				dedwOld(i) = 0.0;
			}
		}
		return currentError;
	}

	//! return the maximal step size component
	double maxDelta() const
	{
		return maxElement(delta);
	}

protected:

	//===========================================================================
	/*!
	*  \brief Determines the sign of "x".
	*
	*  \param x The value of which the sign shall be determined.
	*  \return "-1", if the sign of \em x is negative, "0" otherwise.
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
	int sgn(double x)
	{
		if (x > 0) return 1; if (x < 0) return -1; return 0;
	}

	//! The increase factor \f$ \eta^+ \f$, set to 1.2 by default.
	double np;

	//! The decrease factor \f$ \eta^- \f$, set to 0.5 by default.
	double nm;

	//! The upper limit of the increments \f$ \Delta w_i^{(t)} \f$, set to 1e100 by default.
	double dMax;

	//! The lower limit of the increments \f$ \Delta w_i^{(t)} \f$, set to 0.0 by default.
	double dMin;

	//! The final update values for all weights.
	Array<double> deltaw;

	//! The last error gradient.
	Array<double> dedwOld;

	//! The absolute update values (increment) for all weights.
	Array<double> delta;

};

//! Used to connect the class names with the year of
//! publication of the paper in which the algorithm was introduced.
typedef IRpropPlus Rprop99;

//! Used to connect the class names with the year of
//! publication of the paper in which the algorithm was introduced.
typedef IRpropMinus Rprop99d;

//! Used to connect the class names with the year of
//! publication of the paper in which the algorithm was introduced.
typedef RpropPlus Rprop93;

//! Used to connect the class names with the year of
//! publication of the paper in which the algorithm was introduced.
typedef RpropMinus Rprop94;

#endif

