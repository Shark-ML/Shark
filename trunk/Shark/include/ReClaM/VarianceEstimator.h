//===========================================================================
/*!
 *  \file VarianceEstimator.h
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


#ifndef VARIANCEESTIMATOR_H
#define VARIANCEESTIMATOR_H

#include <ReClaM/Model.h>


//===========================================================================
/*!
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
 *  \par Example
 *  \code
 *  // Create own network class using Variance Estimator:
 *  class MyNet : public FFNet,
 *                public MeanSquaredError,
 *                ...,
 *                public VarianceEstimator
 *
 *  {
 *      ...
 *  }
 *
 *  void main()
 *  {
 *
 *      // Create net:
 *      MyNet net;
 *
 *      // Fill net:
 *      ...
 *
 *      // Training the net:
 *      for ( unsigned epoch = 1; epoch < max_epochs; epoch++ )
 *      {
 *          // Train net with prefered optimization algorithm:
 *          ...
 *
 *          // Estimate current variance of the model:
 *          currVar = net.overallVariance( inputTrain, targetTrain );
 *
 *          // Active learning step - choose one of the patterns
 *          // of the additional set, that will decrease the
 *          // variance most and add it to the training set:
 *          //
 *          // Calculates values "invInfMat", "transInvInfMat" and "s2",
 *          // that are necessary for the estimation
 *          // of the variance change:
 *          net.estimateInvFisher( inputTrain, targetTrain,
 *                                 invInfMat, transInvInfMat, s2 );
 *
 *          // Estimate variance change of the model for all patterns
 *          // of the additional set:
 *          for ( i = 0; i < no_additionals; i++ )
 *          {
 *              varChange =
 *                  net.estimateVarianceChange( inputAdditional[ i ],
 *                                              invInfMat,
 *                                              transInvInfMat, s2    );
 *
 *              ...
 *          }
 *
 *          // Choose pattern which minimizes the model variance:
 *          ...
 *
 *          // Monitor variance at the chosen reference point:
 *          refVar = net.estimateVariance( inputAdditional[ bestIndex ],
 *				    	   invInfMat                     );
 *
 *          // Add chosen pattern to the training set:
 *          ...
 *
 *      } // next training epoch
 *  }
 *  \endcode
 *
 *  This example shows, how to use the variance estimation. <br>
 *  In addition to your "normal" training set, you also use
 *  a set of additional patterns (predefined or random), that can be
 *  added one by one to the original training set. <br>
 *  For each active learning step add one pattern to the training
 *  set and use the methods of class VarianceEstimator to choose
 *  the pattern, that will decrease the variance of the current
 *  model at most. <br>
 *
 *  \author  M. Kreutz
 *  \date    2001-05-04
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 */
class VarianceEstimator
{
public:
	//! The Fisher information matrix is estimated.
	static void estimateFisherInformation
	(
		Model& model,
		const Array< double >& inputA,
		const Array< double >& outputA,
		Array< double >&       infMatA
	);


	//! Estimates the inverse Fisher information matrix and other
	//! values used for quantifying changes in variance due to
	//! active learning.
	static void estimateInvFisher
	(
		Model& model,
		const Array< double >& inputA,           // input
		const Array< double >& outputA,          // input
		Array< double >&       invInfMatA,       // output
		Array< double >&       transInvInfMatA,  // output
		double&                s2A               // output
	);


	//! The estimated output variance of the network for one
	//! input pattern is returned.
	static double estimateVariance
	(
		Model& model,
		const Array< double >& inputA,     // input
		const Array< double >& invInfMatA  // input
	);


	//! Estimates the change in the output variance after an active
	//! learning step.
	static double estimateVarianceChange
	(
		Model& model,
		const Array< double >& inputA,          // input
		const Array< double >& invInfMatA,      // input
		const Array< double >& transInvInfMatA, // input
		double                 s2A              // input
	);

	//! The estimated output variance of the network for all
	//! input patterns is returned.
	static double overallVariance
	(
		Model& model,
		const Array< double >& inputA,   // input
		const Array< double >& outputA   // input
	);
};


#endif // VARIANCEESTIMATOR_H

