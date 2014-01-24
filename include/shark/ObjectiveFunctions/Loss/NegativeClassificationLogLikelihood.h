/*!
 * 
 * \file        NegativeClassificationLogLikelihood.h
 *
 * \brief       Negative logarithm of the likelihood of a probabilistic binary classification model.
 * 
 * 
 *
 * \author      T. Glasmachers, O.Krause, M. Tuma
 * \date        2011
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_LOSS_NEGATIVE_CLASSIFICATION_LOG_LIKELIHOOD_H
#define SHARK_OBJECTIVEFUNCTIONS_LOSS_NEGATIVE_CLASSIFICATION_LOG_LIKELIHOOD_H

#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>

namespace shark{


//!
//! \brief Negative logarithm of the likelihood of a classification model given labeled data.
//!
//! \par
//! This loss class returns the negative logarithm of the likelihood of a classification
//! model given a training example or a set of examples.
//!
//! \par
//! It is most strongly recommended that the model outputs (i.e., the predictions) can be interpreted
//! as probabilities, i.e., that they sum to one over all classes. If this can not be guaranteed,
//! the CrossEntropy loss, which applies a soft-max normalization, or concatenating the model 
//! with a normalizing model should be considered.
//! 
//! \par
//! Consider a model that for each input \f$ x \f$ produces a probability 
//! \f$ P(y=c|x) \f$ of that input belonging to a 
//! certain class \f$ c \f$. Given a set of input-output pairs the predictions made from the inputs can be 
//! compared to the actual class labels. A common measure of agreement between prediction
//! and true label is the negative log-likelihood, which for a set of \f$ m \f$ samples
//! is given by \f[ - \sum_{i=1}^m \ln P(y=y_i|x_i). \f] That is, the logarithmic correct class
//! membership probabilities of all samples are added up.
//! 
//! \par
//! In the case of binary classification, it suffices to pass only one single value,
//! which is interpreted as \f$ P(y=+1|x) \f$ and implies \f$ P(y=-1|x) = 1-P(y=+1|x)\f$
//! (N.B. also in general it would suffice to only specify \f$ d-1 \f$ probabilities, which
//!  is however disregarded here if \f$ d>2 \f$, where \f$ d \f$ is the number of classes,
//!  because of the computational overhead that this would entail).
//! 
//! \par
//! The derivative is that of the output (the loss, or NCLL) w.r.t. the input/prediction:
//! \f[ - \sum_{i=1}^m \frac{1}{P(y=y_i|x_i)} \f]

class NegativeClassificationLogLikelihood : public AbstractLoss<unsigned int,RealVector>
{
private:
	double evalErrorBinary(unsigned int target, double prediction)const;
public:
	NegativeClassificationLogLikelihood();


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NegativeClassificationLogLikelihood"; }

	//! evaluates the loss, see class description
	//! \param target the true label of the example for which the prediction or prediction vector is passed
	//! \param prediction either a single value corresponding to the probability of the example being positive class (label 0) in a binary problem, or a vector with a class membership probability for each class in a multi-class problem
	double eval(UIntVector const& target, RealMatrix const& prediction) const;
	using AbstractLoss<unsigned int,RealVector>::eval;

	//! loss computation with derivatives, see class description
	//! \param target as in #eval
	//! \param prediction as in #eval
	//! \param gradient will hold the gradient of the loss function with respect to the prediction
	double evalDerivative(UIntVector const& target, RealMatrix const& prediction, RealMatrix& gradient) const;
	
protected:

	double m_minArgToLog;
	double m_minLogReturnVal;
	double m_minArgToLog1p;
	double m_minLog1pReturnVal;
	
};

}
#endif
