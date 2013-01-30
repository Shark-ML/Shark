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
///
/// \brief squared loss for regression and classification
///
/// The SquaredLoss computes the squared distance
/// between target and prediction. It is defined for the
/// label type RealVector.
///
template<class VectorType = RealVector>
class SquaredLoss : public AbstractLoss<VectorType, VectorType>
{
private:
	typedef AbstractLoss<VectorType, VectorType> base_type;
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::BatchLabelType BatchLabelType;

	struct DistanceSqr{
		template<class V1, class V2>
	    double operator()(V1 const& v1, V2 const& v2){
	    	return distanceSqr(v1,v2);
	    }
	};
public:
	/// Constructor.
	SquaredLoss()
	{
		this->m_name = "SquaredLoss";
		this->m_features|=base_type::HAS_FIRST_DERIVATIVE;
		//~ this->m_features|=base_type::HAS_SECOND_DERIVATIVE;
	}

	using base_type::eval;

	/// Evaluate the squared loss \f$ (label - prediction)^2 \f$.
	double eval(BatchLabelType const& labels, BatchOutputType const& predictions) const {
		SIZE_CHECK(labels.size1()==predictions.size1());
		SIZE_CHECK(labels.size2()==predictions.size2());

		return accumulateError(labels,predictions,DistanceSqr()); // see Core/utility/functional.h
	}

	/// Evaluate the squared loss \f$ (label - prediction)^2 \f$
	/// and its deriative \f$ \frac{\partial}{\partial prediction} 1/2 (label - prediction)^2 = prediction - label \f$.
	double evalDerivative(BatchLabelType const& label, BatchOutputType const& prediction, BatchOutputType& gradient) const {
		gradient.resize(prediction.size1(),prediction.size2());
		noalias(gradient) = 2.0*(prediction - label);
		return SquaredLoss::eval(label,prediction);
	}

	//~ /// Evaluate the squared loss \f$ (target - prediction)^2 \f$
	//~ /// and its deriatives
	//~ /// \f$ \frac{\partial}{\partial prediction} (target - prediction)^2 = 2*(prediction - target) \f$
	//~ /// and
	//~ /// \f$ \frac{\partial^2}{\partial prediction^2} (target - prediction)^2 = 2I \f$
	//~ /// (where I is the identity matrix).
	//~ double evalDerivative(VectorType const& label, VectorType const& prediction, VectorType& gradient, typename base_type::MatrixType& hessian) const {
		//~ gradient.resize(prediction.size());
		//~ noalias(gradient) = 2.0*(prediction - label);
		//~ hessian = 2.0*RealIdentity(gradient.size());
		//~ return distanceSqr(prediction , label);
	//~ }
};

}
#endif
