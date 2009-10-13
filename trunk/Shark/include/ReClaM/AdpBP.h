//===========================================================================
/*!
 *  \file AdpBP.h
 *
 *  \brief Offers the two versions of the gradient descent based
 *         optimization algorithm with individual adaptive learning
 *         rates by Silva and Almeida (Adaptive BackPropagation).
 *
 *  \author  C. Igel
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
 **  <BR>
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
 *
 *
 */
//===========================================================================


#ifndef ADP_BP_H
#define ADP_BP_H

#include <SharkDefs.h>
#include <Array/ArrayOp.h>
#include <ReClaM/Optimizer.h>


//===========================================================================
/*!
 *  \brief Offers the gradient-based optimization algorithm with
 *         individual adaptive learning rates by Silva and Almeida
 *         (Adaptive BackPropagation).
 *
 *  This optimization algorithm introduced by Silva and Almeida adds
 *  individual adaptive learning rates and weight-backtracking to
 *  standard steepest descent.
 *  \par
 *  To avoid bad choices of the learning rate, the algorithm by
 *  Silva and Almeida uses an individual adaptive learning rate
 *  \f$\eta_i\f$ for each weight \f$w_i\f$.  The adaptation of
 *  \f$\eta_i\f$ is determined by the sign of
 *  the partial derivation \f$\frac{\partial E}{\partial w_i}\f$. If
 *  the sign of the derivation changes from iteration \f$t - 1\f$ to
 *  iteration \f$t\f$, then at least one minimum with regard to
 *  \f$w_i\f$ was skipped and hence the learning rate \f$\eta_i\f$ is
 *  decreased by the multiplication with a predefined constant.
 *  If there is no change of the sign, it a plateau of the objective function
 *  is assumed and the learning rate \f$\eta_i\f$ is increased by the
 *  multiplication with a second constant.  Furthermore,
 *  weight-backtracking is used, i.e.. if the error increases from
 *  iteration \f$t - 1\f$ to \f$t\f$, then the last weight adaptation
 *  is undone. For further information about this algorithm,
 *  please refer to:
 *
 \verbatim
 @InCollection{silva:90,
    author =       {Fernando M. Silva and Luis B. Almeida},
    editor =       {Rolf Eckmiller},
    booktitle =    {Advanced Neural Computers},
    title =        {Speeding Up Backpropagation},
    publisher =    {North-Holland},
    year =         {1990},
    pages =        {151-158},
  }
 \endverbatim
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable, documentation checked
 *
 *  \sa AdpBP90b
 *
 */
class AdpBP90a : public Optimizer
{
public:

	//===========================================================================
	/*!
	*  \brief Initializes the optimizer with default parameters.
	*
	*  Initializes the optimizer's parameters with the following defaults:
	   *  <ul>
	   *    <li>Increase factor for adaptive learning rate = 1.2
	   *    <li>Decrease factor for adaptive learning rate = 0.7
	   *    <li>Momentum factor \f$ \alpha = 0.5 \f$
	   *    <li>Initial value for adaptive learning rate = 0.1
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
	*  \brief Initializes the optimizer with user defined parameters.
	   *
	   *      \param  model            Model to be optimized.
	   *      \param  increaseFactorP  Increase factor for the adaptive learning rate \f$ \eta \f$,
	   *                               set to 1.2 by default.
	   *      \param  decreaseFactorP  Decrease factor for the adaptive learning rate \f$ \eta \f$,
	   *                               set to 0.7 by default.
	   *      \param  alphaP           Momentum factor \f$ \alpha \f$ that controls the effect
	   *                               of the momentum term, set to 0.5 by default.
	   *      \param  eta0P            Initial value for the adapted learning rate \f$ \eta \f$, set to 0.1 by default.
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
	void initUserDefined(Model& model, double increaseFactorP = 1.2, double decreaseFactorP = 0.7, double alphaP = 0.5, double eta0P = 0.1)
	{
		increaseFactor  = increaseFactorP;
		decreaseFactor  = decreaseFactorP;
		alpha           = alphaP;
		eta0            = eta0P;

		deltaw.resize(model.getParameterDimension());
		eta.resize(model.getParameterDimension());
		dedwOld.resize(model.getParameterDimension());
		v.resize(model.getParameterDimension());

		v       = 0.;
		eta     = eta0P;
		deltaw  = 0;
		dedwOld = 0;

		oldError   = MAXDOUBLE;
	}

	//===========================================================================
	/*!
	*  \brief Performs a run of the optimization algorithm.
	*
	*  The error \f$E^{(t)}\f$ and its derivatives with respect to the model 
	*  parameters for the current iteration 
	*  \f$t\f$ are calculated and the values of the weights \f$w_i,\ i = 1, 
	*  \dots, n\f$ and the  individual learning rates \f$\eta_i\f$ are 
	*  adapted.
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
		double currentError = errorfunction.errorDerivative(model, input, target, dedw); //derror(in, out);

		if (oldError < currentError)
		{
			for (unsigned i = 0; i < model.getParameterDimension(); i++)
			{
				model.setParameter(i, model.getParameter(i) - deltaw(i)); //w -= deltaw;
			}
		}

		for (unsigned  i = 0; i < model.getParameterDimension(); i++)
		{
			if (dedw(i) * dedwOld(i) > 0) eta(i) *= increaseFactor;
			if (dedw(i) * dedwOld(i) < 0) eta(i) *= decreaseFactor;
		}

		v = dedw + alpha * v;
		deltaw = -eta * v;

		for (unsigned i = 0; i < model.getParameterDimension(); i++)
		{
			model.setParameter(i, model.getParameter(i) + deltaw(i)); //w += deltaw;
		}

		dedwOld = dedw;
		oldError = currentError;
		return currentError;
	}

	//===========================================================================
	/*!
	*  \brief Resets internal variables of the class instance.
	*
	*      \return 
	*          none
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
	void clearMemory()
	{
		oldError   = MAXDOUBLE;
		eta = eta0; dedwOld = 0; deltaw = 0;
	};

protected:

	//! The increase factor for the adaptive learning rate.
	//! Set to 1.2 by default.
	double increaseFactor;

	//! The decrease factor for the adaptive learning rate.
	//! Set to 0.7 by default.
	double decreaseFactor;

	//! The momentum factor \f$\alpha\f$ that controls the effect of the momentum term.
	//! Set to 0.5 by default.
	double alpha;

	//! The initial value for the adaptive learning rate \f$\eta\f$
	//! for all weights of the network.
	double eta0;

	//! The old error value.
	double oldError;

	//! The final update values (including the individual learning rates)
	//! for all weights.
	Array<double> deltaw;

	//! The update values (without the individual learning rates) for all
	//! weights.
	Array<double> v;

	//! The last error gradient.
	Array<double> dedwOld;

	//! The adaptive learning rate \f$\eta\f$
	//! (for each weight individually).
	Array<double> eta;

};

//===========================================================================
/*!
 *  \brief Offers the second version of the gradient descent based
 *         optimization algorithm with individual adaptive learning rates
 *         by Silva and Almeida (Adaptive BackPropagation).
 *  This optimization algorithm introduced by Silva and Almeida adds
 *  individual adaptive learning rates and weight-backtracking to
 *  standard steepest descent.
 *  \par
 *  To avoid bad choices of the learning rate, the algorithm by
 *  Silva and Almeida uses an individual adaptive learning rate
 *  \f$\eta_i\f$ for each weight \f$w_i\f$.  The adaptation of
 *  \f$\eta_i\f$ is determined by the sign of
 *  the partial derivation \f$\frac{\partial E}{\partial w_i}\f$. If
 *  the sign of the derivation changes from iteration \f$t - 1\f$ to
 *  iteration \f$t\f$, then at least one minimum with regard to
 *  \f$w_i\f$ was skipped and hence the learning rate \f$\eta_i\f$ is
 *  decreased by the multiplication with a predefined constant.
 *  If there is no change of the sign, it a plateau of the objective function
 *  is assumed and the learning rate \f$\eta_i\f$ is increased by the
 *  multiplication with a second constant.  Furthermore,
 *  weight-backtracking is used, i.e.. if the error increases from
 *  iteration \f$t - 1\f$ to \f$t\f$, then the last weight adaptation
 *  is undone. For further information about this algorithm,
 *  please refer to:
 *
 \verbatim
 @InCollection{silva:90b,
  author =       {Fernando M. Silva and Luis B. Almeida},
  title =        {Acceleration Techniques for the Backpropagation Algorithm},
  booktitle =    {Neural Networks -- EURASIP Workshop 1990},
  pages =        {110-119},
  year =         {1990},
  editor =       {Luis B. Almeida and C. J. Wellekens},
  number =       {412},
  series =       {LNCS},
  publisher = {Springer-Verlag},
 }
 \endverbatim
 *  This second version of the algorithm has some minor modifications
 *  that improve the performance.
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
class AdpBP90b : public Optimizer
{
public:

	//===========================================================================
	/*!
	*  \brief Initializes the optimizer with default parameters.
	*
	*  Initializes the optimizer's parameters with the following defaults:
	   *  <ul>
	   *    <li>Increase factor for adaptive learning rate = 1.2
	   *    <li>Decrease factor for adaptive learning rate = 0.7
	   *    <li>Momentum factor \f$ \alpha = 0.5 \f$
	   *    <li>Initial value for adaptive learning rate = 0.1
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
	*  \brief Initializes the optimizer with user defined parameters.
	   *
	   *      \param  model            Model to be optimized.
	   *      \param  increaseFactorP  Increase factor for the adaptive learning rate \f$ \eta \f$,
	   *                               set to 1.2 by default.
	   *      \param  decreaseFactorP  Decrease factor for the adaptive learning rate \f$ \eta \f$,
	   *                               set to 0.7 by default.
	   *      \param  alphaP           Momentum factor \f$ \alpha \f$ that controls the effect
	   *                               of the momentum term, set to 0.5 by default.
	   *      \param  eta0P            Initial value for the adapted learning rate \f$ \eta \f$, set to 0.1 by default.
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
	void initUserDefined(Model& model, double increaseFactorP = 1.2, double decreaseFactorP = 0.7, double alphaP = 0.5, double eta0P = 0.1)
	{
		increaseFactor  = increaseFactorP;
		decreaseFactor  = decreaseFactorP;
		alpha           = alphaP;
		eta0            = eta0P;

		deltaw.resize(model.getParameterDimension());
		eta.resize(model.getParameterDimension());
		dedwOld.resize(model.getParameterDimension());
		v.resize(model.getParameterDimension());

		v       = 0.;
		eta     = eta0P;
		deltaw  = 0;
		dedwOld = 0;

		oldError   = MAXDOUBLE;
	}

	//===========================================================================
	/*!
	*  \brief Performs a run of the optimization algorithm.
	*
	*  The error \f$E^{(t)}\f$ and its derivatives with respect to the model 
	*  parameters for the current iteration 
	*  \f$t\f$ are calculated and the values of the weights \f$w_i,\ i = 1, 
	*  \dots, n\f$ and the  individual learning rates \f$\eta_i\f$ are 
	*  adapted.
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
	*      2002-01-18, ra: <br>
	*      Removed some forgotten "couts" used for monitoring.
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
		double currentError = errorfunction.errorDerivative(model, input, target, dedw); //derror(in, out);

		for (unsigned i = 0; i < model.getParameterDimension(); i++)
		{
			if (dedw(i) * dedwOld(i) > 0) eta(i) *= increaseFactor;
			if (dedw(i) * dedwOld(i) < 0) eta(i) *= decreaseFactor;
		}

		if ((1E-4 + oldError) < currentError)
		{
			for (unsigned i = 0; i < model.getParameterDimension(); i++)
			{
				model.setParameter(i, model.getParameter(i) - deltaw(i)); //w -= deltaw;
			}
			dedwOld  = dedw;
		}
		else
		{
			v = dedw + alpha * v;
			deltaw = -eta * v;
			dedwOld = dedw;
			oldError = currentError;
			for (unsigned i = 0; i < model.getParameterDimension(); i++)
			{
				model.setParameter(i, model.getParameter(i) + deltaw(i)); //w += deltaw;
			}
		}
		return currentError;
	}

	//===========================================================================
	/*!
	*  \brief Resets internal variables of the class instance.
	*
	*      \return 
	*          none
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
	void clearMemory()
	{
		eta = eta0; dedwOld = 0; deltaw = 0;
		oldError   = MAXDOUBLE;
	};

protected:

	//! The increase factor for the adaptive learning rate.
	//! Set to 1.2 by default.
	double increaseFactor;

	//! The decrease factor for the adaptive learning rate.
	//! Set to 0.7 by default.
	double decreaseFactor;

	//! The momentum factor \f$\alpha\f$ that controls the effect of the momentum term.
	//! Set to 0.5 by default.
	double alpha;

	//! The initial value for the adaptive learning rate \f$\eta\f$
	//! for all weights of the network.
	double eta0;

	//! The old error value.
	double oldError;

	//! The final update values (including the individual learning rates)
	//! for all weights.
	Array<double> deltaw;

	//! The update values (without the individual learning rates) for all
	//! weights.
	Array<double> v;

	//! The last error gradient.
	Array<double> dedwOld;

	//! The adaptive learning rate \f$\eta\f$
	//! (for each weight individually).
	Array<double> eta;

};

#endif

