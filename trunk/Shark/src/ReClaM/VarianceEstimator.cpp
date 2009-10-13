//===========================================================================
/*!
 *  \file VarianceEstimator.cpp
 *
 *  \brief Offers the methods to deal with active learning of a neural
 *         network.
 *
 *  Most neural networks are used as passive learners, i.e. they
 *  will work on a fixed training set. In natural learning problems,
 *  however, the learner can gather new information from its environment
 *  to improve the learning process.
 *  Using active learning in connection with neural networks,
 *  the network is allowed to select a new training input \f$\tilde{x}\f$
 *  at each time step. <br>
 *  To achieve a real improvement of the learning process, the selected
 *  new input is not chosen by random. The goal is to chose an input, that
 *  minimizes the expectation of the learner's mean squared error.
 *  So you will find methods here, that will estimate the output
 *  variance of the network, when adding a new example to the training
 *  set. <br>
 *  For more details about active learning in neural networks
 *  and the formulas used here for the variance estimations, please
 *  refer to David A. Cohn: "Neural Network Exploration Using Optimal
 *  Experimental Design."
 *
 *
 *  \author  M. Kreutz
 *  \date    2001-05-04
 *
 *  \par Copyright (c) 2001:
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


#include <SharkDefs.h>
#include <Array/ArrayOp.h>
#include <LinAlg/LinAlg.h>
#include <ReClaM/VarianceEstimator.h>


//===========================================================================
/*!
 *  \brief The Fisher information matrix is estimated.
 *
 *  Given a training set with input patterns
 *  \f$x_i\f$ with the corresponding target values \f$y_i \mbox{,\ }
 *  i = 1, \dots, m\f$ the current value of the used negative log
 *  likelihood measure of the weights (also called a score) is proportional to
 *
 *  \f$
 *  S^2 = \frac{1}{m} \sum_{i = 1}^m (y_i - \hat{y}(x_i))^2
 *  \f$
 *
 *  where \f$\hat{y}\f$ is the output of the used network for the input
 *  \f$x_i\f$. After calculating the value of the measure, the
 *  Fisher information matrix can be estimated by
 *
 *  \f$
 *  A = \frac{1}{S^2} \sum_{i = 1}^m \frac{\partial \hat{y}}{\partial w}
 *      \frac{\partial \hat{y}}{\partial w}^T
 *  \f$
 *
 *  where \f$\hat{y} = f_{\hat{w}}(x)\f$ is the learners best guess for
 *  mapping inputs \f$x\f$ to outputs \f$y\f$ with \f$\hat{w}\f$ as the
 *  weight vector of the network that minimizes the negative log likelihood
 *  measure \f$S^2\f$, i.e. \f$\hat{y}\f$ is an estimate of the
 *  corresponding \f$y\f$.<br>
 *  The Fisher information matrix measures the information contained
 *  in the training data set about each parameter (i.e. each weight).<br>
 *  The inverse of the matrix then provides an approximation to
 *  the variance of the maximum-likelihood estimator which becomes
 *  increasingly accurate as the sample size increases.
 *
 *      \param  inputA  The input patterns \f$x_i\f$ of the training set.
 *      \param  outputA The target values \f$y_i\f$ for the used input
 *                      patterns.
 *      \param  infMatA The calculated Fisher information matrix \f$A\f$.
 *      \return none
 *      \throw check_exception the type of the exception will be
 *             "size mismatch" and indicates that \em inputA is
 *             not 2-dimensional
 *
 *  \author  M. Kreutz
 *  \date    2001-05-04
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
void VarianceEstimator::estimateFisherInformation
(
	Model& model,
	const Array< double >& inputA,
	const Array< double >& outputA,
	Array< double >&       infMatA
)
{
	SIZE_CHECK(inputA.ndim() == 2)

	double          mseL = 0;
	Array< double > dwL;
	Array< double > modelOutputL(outputA.dim(1));

	for (unsigned i = 0; i < inputA.dim(0); ++i)
	{
		model.modelDerivative(inputA[ i ], modelOutputL, dwL);
		mseL += sqrDistance(outputA[ i ], modelOutputL);

		dwL.resize(dwL.nelem(), true);

		if (i == 0)
		{
			infMatA = outerProduct(dwL, dwL);
		}
		else
		{
			infMatA += outerProduct(dwL, dwL);
		}
	}

	mseL    /= inputA.dim(0);
	infMatA /= mseL;
}


//===========================================================================
/*!
 *  \brief Estimates the inverse Fisher information matrix and other
 *         values used for quantifying changes in variance due to
 *         active learning.
 *
 *  Given all input patterns \f$x_i \mbox{,\ } i = 1, \dots, m\f$
 *  of the training set and the corresponding target values
 *  \f$y_i\f$ three values/matrices are calculated:
 *
 *  <ul>
 *  <li>the inverse Fisher information matrix (\f$A^{-1}\f$)</li>
 *  <li>the negative log likelihood measure \f$S^2\f$ </li>
 *  <li>\f$A^{-1} \cdot {\langle \frac{\partial \hat{y}}{\partial w}\frac{\partial \hat{y}}{\partial w}^T \rangle}_X \cdot A^{-1}\f$, where \f${\langle \cdot \rangle}_X\f$ represents the expected value over \f$X\f$.
 *  </ul>
 *
 *  For the definition of \f$A\f$, \f$S^2\f$ and \f$\hat{y}\f$ please refer
 *  to method #estimateFisherInformation. <br>
 *  The three evaluated values are used for the calculation of the
 *  expected change in the output variance (calculated by method #estimateVarianceChange),
 *  after a new example \f$(\tilde{x}, \tilde{y})\f$ is added to the
 *  training set.
 *
 *      \param  inputA          The input patterns \f$x_i\f$ of the training
 *                              set, including the new input pattern
 *                              \f$\tilde{x}\f$.
 *      \param  outputA         The target values \f$y_i\f$ for the used input
 *                              patterns.
 *      \param  invInfMatA      The calculated inverse Fisher information
 *                              matrix \f$A^{-1}\f$.
 *      \param  transInvInfMatA The third calculated value as listed above.
 *      \param  s2A             The current value of the negative log
 *                              likelihood measure \f$S^2\f$.
 *
 *      \return none
 *      \throw check_exception the type of the exception will be
 *             "size mismatch" and indicates that \em inputA is
 *             not 2-dimensional
 *
 *  \author  M. Kreutz
 *  \date    2001-05-04
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 *  \sa #estimateInvFisher
 *
 */
void VarianceEstimator::estimateInvFisher
(
	Model& model,
	const Array< double >& inputA,
	const Array< double >& outputA,
	Array< double >&       invInfMatA,
	Array< double >&       transInvInfMatA,
	double&                s2A
)
{
	SIZE_CHECK(inputA.ndim() == 2)

	Array< double > infMatL;
	Array< double > dwL;
	Array< double > dwMeanL;
	Array< double > modelOutputL(outputA.dim(1));

	s2A = 0;
	for (unsigned i = 0; i < inputA.dim(0); ++i)
	{
		model.modelDerivative(inputA[ i ], modelOutputL, dwL);
		s2A += sqrDistance(outputA[ i ], modelOutputL);

		dwL.resize(dwL.nelem(), true);

		if (i == 0)
		{
			infMatL = outerProduct(dwL, dwL);
			dwMeanL = dwL;
		}
		else
		{
			infMatL += outerProduct(dwL, dwL);
			dwMeanL += dwL;
		}
	}

	s2A     /= inputA.dim(0);
	infMatL /= s2A;

	invInfMatA = invert(infMatL);

	//
	// innerProduct yields the same results as matrixProduct (not
	// implemented yet) since all matrices are symmetric
	//
	transInvInfMatA =
		innerProduct(invInfMatA,
					 innerProduct(outerProduct(dwMeanL, dwMeanL),
								  invInfMatA));
}


//===========================================================================
/*!
 *  \brief The estimated output variance of the network for all
 *         input patterns is returned.
 *
 *  The average variance over all input patterns \f$x_i \mbox{,\ }
 *  i = 1, \dots, m\f$ is calculated by a stochastic estimate
 *  based on an average of \f$\sigma^2_{\hat{y}|x_r}\f$ with a
 *  reference point \f$x_r\f$ drawn according to the known
 *  input distribution (for a calculation of \f$\sigma^2_{\hat{y}|x_r}\f$
 *  for a single reference point \f$x_r\f$ please refer to method
 *  #estimateVariance). <br>
 *  The overall variance is then estimated by
 *
 *  \f$
 *  {\langle \sigma^2_{\hat{y}} \rangle}_X = {\langle
 *  \frac{\partial \hat{y}}{\partial w}\rangle}_X^T A^{-1}
 *  {\langle \frac{\partial \hat{y}}{\partial w}\rangle}_X +
 *  \mbox{trace}(A^{-1} {\langle \frac{\partial \hat{y}}{\partial w}
 *  \rangle}_X^T {\langle \frac{\partial \hat{y}}{\partial w}\rangle}_X)
 *  \f$
 *
 *  where \f$\hat{y} = f_{\hat{w}}(x)\f$ is the learners best guess for
 *  mapping inputs \f$x\f$ to outputs \f$y\f$ with \f$\hat{w}\f$ as the
 *  weight vector of the network that minimizes the negative log likelihood
 *  measure \f$S^2\f$, \f$A^{-1}\f$ is the inverse Fisher information
 *  matrix (see method #estimateFisherInformation for details) and
 *  \f${\langle \cdot \rangle}_X\f$ represents the expected value over
 *  \f$X\f$.
 *
 *
 *      \param  inputA  The input patterns \f$x_i\f$ of the
 *                      training set.
 *      \param  outputA The corresponding target values \f$y_i\f$
 *                      of the input patterns.
 *      \return The estimated output variance \f${\langle \sigma^2_{\hat{y}}
 *              \rangle}_X\f$.
 *      \throw check_exception the type of the exception will be
 *             "size mismatch" and indicates that \em inputA is
 *             not 2-dimensional
 *
 *  \author  M. Kreutz
 *  \date    2001-05-04
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
double VarianceEstimator::overallVariance
(
	Model& model,
	const Array< double >& inputA,
	const Array< double >& outputA
)
{
	SIZE_CHECK(inputA.ndim() == 2)

	double          mseL = 0;
	Array< double > infMatL;
	Array< double > invInfMatL;
	Array< double > invAdwMeanL;
	Array< double > dwL;
	Array< double > dwMeanL;
	Array< double > modelOutputL(outputA.dim(1));

	for (unsigned i = 0; i < inputA.dim(0); ++i)
	{
		model.modelDerivative(inputA[ i ], modelOutputL, dwL);
		mseL += sqrDistance(outputA[ i ], modelOutputL);

		dwL.resize(dwL.nelem(), true);

		if (i == 0)
		{
			infMatL = outerProduct(dwL, dwL);
			dwMeanL = dwL;
		}
		else
		{
			infMatL += outerProduct(dwL, dwL);
			dwMeanL += dwL;
		}
	}

	mseL       /= inputA.dim(0);
	infMatL    /= mseL;
	invInfMatL  = invert(infMatL);
	invAdwMeanL = innerProduct(invInfMatL, outerProduct(dwMeanL, dwMeanL));

	return scalarProduct(dwMeanL, innerProduct(invInfMatL, dwMeanL))
		   + trace(invAdwMeanL);
}


//===========================================================================
/*!
 *  \brief The estimated output variance of the network for one
 *         input pattern is returned.
 *
 *  Given one input pattern, the output variance \f$\sigma^2_{\hat{y}|x_r}\f$
 *  at this reference input \f$x_r\f$ can be approximated by
 *  \f$
 *  \sigma^2_{\hat{y}|x_r} \approx \frac{\partial \hat{y}}{\partial w}^T
 *  A^{-1} \frac{\partial \hat{y}}{\partial w}
 *  \f$
 *
 *  where \f$\hat{y} = f_{\hat{w}}(x)\f$ is the learners best guess for
 *  mapping inputs \f$x\f$ to outputs \f$y\f$ with \f$\hat{w}\f$ as the
 *  weight vector of the network that minimizes the negative log likelihood
 *  measure \f$S^2\f$ and \f$A^{-1}\f$ is the inverse Fisher information
 *  matrix (see method #estimateFisherInformation for details).
 *
 *      \param  inputA     The reference input pattern \f$x_r\f$ of the
 *                         training set.
 *      \param  invInfMatA The calculated inverse Fisher information matrix
 *                         \f$A^{-1}\f$.
 *      \return The estimated output variance \f$\sigma^2_{\hat{y}|x_r}\f$.
 *
 *  \author  M. Kreutz
 *  \date    2001-05-04
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 */
double VarianceEstimator::estimateVariance
(
	Model& model,
	const Array< double >& inputA,
	const Array< double >& invInfMatA
)
{
	Array< double > dwL;
	model.modelDerivative(inputA, dwL);
	dwL.resize(dwL.nelem(), true);
	return scalarProduct(dwL, innerProduct(invInfMatA, dwL));
}


//===========================================================================
/*!
 *  \brief Estimates the change in the output variance after an active
 *         learning step.
 *
 *  When active learning takes place, i.e. a chosen input \f$\tilde{x}\f$
 *  and its corresponding \f$\tilde{y}\f$ are added to the training set,
 *  the variance of the model will change. Then the variance change
 *  has to be computed to select \f$\tilde{x}\f$ optimally. <br>
 *  The expected change in output variance can be estimated by
 *  \f[
 *  {\langle \Delta \sigma^2_{\hat{y}|\tilde{x}}\rangle}_{X,\tilde{Y}}
 *  = \frac{\frac{\partial \hat{y}}{\partial w}^T \cdot {\langle \frac{\partial \hat{y}}{\partial w}\frac{\partial \hat{y}}{\partial w}^T \rangle}_X \cdot A^{-1} \cdot \frac{\partial \hat{y}}{\partial w}}{S^2 + \frac{\partial \hat{y}}{\partial w}^T \cdot A^{-1} \cdot \frac{\partial \hat{y}}{\partial w}},
 *  \f]
 *  where \f${\langle \cdot \rangle}_{X, \tilde{Y}}\f$ represents the
 *  expected values over \f$X\f$ and \f$\tilde{Y}\f$. <br>
 *  For a definition of \f$S^2\f$, \f$A\f$ and \f$\hat{y}\f$
 *  please refer to method #estimateFisherInformation. <br>
 *  The values used for the calculation of the variance change can be
 *  calculated by using method #estimateInvFisher.
 *
 *      \param  inputA          The new input pattern \f$\tilde{x}\f$
 *                              that should be added to the training set.
 *      \param  invInfMatA      The inverse Fisher information matrix.
 *      \param  transInvInfMatA A term that is used for the calculation of the
 *                              variance change. For a definition of this term
 *                              please refer to method #estimateInvFisher.
 *      \param  s2A             The current value of the negative log
 *                              likelihood measure \f$S^2\f$.
 *      \return
 *
 *  \author  M. Kreutz
 *  \date    2001-05-04
 *
 *  \par Changes
 *      none
 *
 *  \par Status
 *      stable
 *
 *  \sa #estimateInvFisher
 */
double VarianceEstimator::estimateVarianceChange
(
	Model& model,
	const Array< double >& inputA,
	const Array< double >& invInfMatA,
	const Array< double >& transInvInfMatA,
	double                 s2A
)
{
	Array< double > dwL;
	model.modelDerivative(inputA, dwL);
	dwL.resize(dwL.nelem(), true);
	return scalarProduct(dwL, innerProduct(transInvInfMatA, dwL))
		   / (s2A + scalarProduct(dwL, innerProduct(invInfMatA, dwL)));
}

