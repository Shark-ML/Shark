//===========================================================================
/*!
 * 
 *
 * \brief       Error measure for classication tasks, typically used for evaluation of results
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2010-2011
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_LOSS_ZEROONELOSS_H
#define SHARK_OBJECTIVEFUNCTIONS_LOSS_ZEROONELOSS_H


#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>

namespace shark {

///
/// \brief 0-1-loss for classification.
///
/// The ZeroOneLoss requires the existence of the comparison
/// operator == for its LabelType template parameter. The
/// loss function returns zero of the predictions exactly
/// matches the label, and one otherwise.
///
template<class LabelType = unsigned int, class OutputType = LabelType>
class ZeroOneLoss : public AbstractLoss<LabelType, LabelType>
{
public:
	typedef AbstractLoss<LabelType, LabelType> base_type;
	typedef typename base_type::BatchLabelType BatchLabelType;
	typedef typename base_type::BatchOutputType BatchOutputType;

	/// constructor
	ZeroOneLoss()
	{ }


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ZeroOneLoss"; }

	using base_type::eval;

	///\brief Return zero if labels == predictions and one otherwise.
	double eval(BatchLabelType const& labels, BatchOutputType const& predictions) const{
		std::size_t numInputs = size(labels);
		SIZE_CHECK(numInputs == size(predictions));

		double error = 0;
		for(std::size_t i = 0; i != numInputs; ++i){
			error += (predictions(i) != labels(i))?1.0:0.0;
		}
		return error;
	}
};


/// \brief 0-1-loss for classification.
template <>
class ZeroOneLoss<unsigned int, RealVector> : public AbstractLoss<unsigned int, RealVector>
{
public:
	typedef AbstractLoss<unsigned int, RealVector> base_type;
	typedef base_type::BatchLabelType BatchLabelType;
	typedef base_type::BatchOutputType BatchOutputType;

	/// constructor
	///
    /// \param threshold: in the case dim(predictions) == 1, predictions strictly larger than this parameter are regarded as belonging to the positive class
	ZeroOneLoss(double threshold = 0.0)
	{
		m_threshold = threshold;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ZeroOneLoss"; }


	// annoyingness of C++ templates
	using base_type::eval;

	/// Return zero if labels == arg max { predictions_i } and one otherwise,
	/// where the index i runs over the components of the predictions vector.
	/// A special version of dim(predictions) == 1 computes the predicted
	/// labels by thresholding at zero. Shark's label convention is used,
	/// saying that a positive value encodes class 0, a negative value
	/// encodes class 1.
	double eval(BatchLabelType const& labels, BatchOutputType const& predictions) const{
		std::size_t numInputs = size(labels);
		SIZE_CHECK(numInputs == (std::size_t)size(predictions));

		double error = 0;
		for(std::size_t i = 0; i != numInputs; ++i){
			error+=evalSingle(labels(i),get(predictions,i));
		}
		return error;
	}
private:
	template<class VectorType>
	double evalSingle(unsigned int label, VectorType const& predictions) const{
		std::size_t size = predictions.size();
		if (size == 1){
			// binary case, single real-valued predictions
			unsigned int t = (predictions(0) > m_threshold);
			if (t == label) return 0.0;
			else return 1.0;
		}
		else{
			// multi-class case, one prediction component per class
			RANGE_CHECK(label < size);
			double p = predictions(label);
			for (std::size_t i = 0; i<size; i++)
			{
				if (i == label) continue;
				if (predictions(i) >= p) return 1.0;
			}
			return 0.0;
		}
	}

	double m_threshold; ///< in the case dim(predictions) == 1, predictions strictly larger tha this parameter are regarded as belonging to the positive class
};


}
#endif
