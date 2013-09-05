/*!
 *  \brief Implements the Squared Error Loss function for regression.
 *
 *
 *  \author  Oswin Krause, Christian Igel
 *  \date    2011
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef SHARK_OBJECTIVEFUNCTIONS_LOSS_SQUAREDLOSS_H
#define SHARK_OBJECTIVEFUNCTIONS_LOSS_SQUAREDLOSS_H


#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>

namespace shark{
/// \brief squared loss for regression and classification
///
/// The SquaredLoss computes the squared distance
/// between target and prediction. It is defined for both
/// vectorial as well as integral labels. In the case of integral labels,
/// the label c is interpreted as unit-vector having the c-th component activated.
///
template<class OutputType = RealVector, class LabelType = OutputType >
class SquaredLoss : public AbstractLoss<LabelType,OutputType>
{
public:
	typedef AbstractLoss<LabelType,OutputType> base_type;
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::BatchLabelType BatchLabelType;

	/// Constructor.
	SquaredLoss()
	{
		this->m_features|=base_type::HAS_FIRST_DERIVATIVE;
	}


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SquaredLoss"; }

	using base_type::eval;

	/// Evaluate the squared loss \f$ (label - prediction)^2 \f$.
	double eval(BatchLabelType const& labels, BatchOutputType const& predictions) const {
		SIZE_CHECK(labels.size1()==predictions.size1());
		SIZE_CHECK(labels.size2()==predictions.size2());

		double error = 0;
		for(std::size_t i = 0; i != labels.size1(); ++i){
			error+=distanceSqr(row(predictions,i),row(labels,i));
		}
		return error;
	}

	/// Evaluate the squared loss \f$ (label - prediction)^2 \f$
	/// and its deriative \f$ \frac{\partial}{\partial prediction} 1/2 (label - prediction)^2 = prediction - label \f$.
	double evalDerivative(BatchLabelType const& label, BatchOutputType const& prediction, BatchOutputType& gradient) const {
		gradient.resize(prediction.size1(),prediction.size2());
		noalias(gradient) = 2.0*(prediction - label);
		return SquaredLoss::eval(label,prediction);
	}
};

//specialisation for classification case.

template<class OutputType>
class SquaredLoss<OutputType,unsigned int> : public AbstractLoss<unsigned int,OutputType>
{
public:
	typedef AbstractLoss<unsigned int,OutputType> base_type;
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::BatchLabelType BatchLabelType;

	/// Constructor.
	SquaredLoss()
	{
		this->m_features|=base_type::HAS_FIRST_DERIVATIVE;
	}


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SquaredLoss"; }

	using base_type::eval;

	/// Evaluate the squared loss \f$ (label - prediction)^2 \f$.
	double eval(BatchLabelType const& labels, BatchOutputType const& predictions) const {
		SIZE_CHECK(labels.size()==predictions.size1());

		double error = 0;
		for(std::size_t i = 0; i != labels.size(); ++i){
			unsigned int c = labels(i);
			SIZE_CHECK(c < predictions.size2());
			error+=norm_sqr(row(predictions,i))+1.0-2.0*predictions(i,c);
		}
		return error;
	}

	/// Evaluate the squared loss \f$ (label - prediction)^2 \f$
	/// and its deriative \f$ \frac{\partial}{\partial prediction} 1/2 (label - prediction)^2 = prediction - label \f$.
	double evalDerivative(BatchLabelType const& labels, BatchOutputType const& predictions, BatchOutputType& gradient) const {
		gradient.resize(predictions.size1(),predictions.size2());
		noalias(gradient) = 2.0*predictions;
		for(std::size_t i = 0; i != labels.size(); ++i){
			unsigned int c = labels(i);
			SIZE_CHECK(c < predictions.size2());
			gradient(i,c)-=2.0;
		}
		return SquaredLoss::eval(labels,predictions);
	}
};

}
#endif
