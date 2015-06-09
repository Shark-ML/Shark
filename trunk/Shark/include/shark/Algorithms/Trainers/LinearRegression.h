//===========================================================================
/*!
 * 
 *
 * \brief       Linear Regression
 * 
 * \par
 * This implementation is based on a class removed from the
 * LinAlg package, written by M. Kreutz in 1998.
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2007-2011
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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
//===========================================================================


#ifndef SHARK_ALGORITHMS_TRAINERS_LINEARREGRESSION_H
#define SHARK_ALGORITHMS_TRAINERS_LINEARREGRESSION_H

#include <shark/Core/DLLSupport.h>
#include <shark/Models/LinearModel.h>
#include <shark/Core/IParameterizable.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>

namespace shark {


/*!
 *  \brief Linear Regression
 *
 *  Linear Regression builds an affine linear model
 *  \f$ f(x) = A x + b \f$ minimizing the squared
 *  error from a dataset of pairs of vectors (x, y).
 *  That is, the error
 *  \f$ \sum_i (f(x_i) - y_i)^2 \f$ is minimized.
 *  The solution to this problem is found analytically.
 */
class LinearRegression : public AbstractTrainer<LinearModel<> >, public IParameterizable
{
public:
	SHARK_EXPORT_SYMBOL LinearRegression(double regularization = 0.0);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearRegression"; }

	double regularization() const{ 
		return m_regularization; 
	}
	void setRegularization(double regularization) {
		RANGE_CHECK(regularization >= 0.0);
		m_regularization = regularization;
	}

	RealVector parameterVector() const {
		RealVector param(1);
		param(0) = m_regularization;
		return param;
	}
	void setParameterVector(const RealVector& param) {
		SIZE_CHECK(param.size() == 1);
		m_regularization = param(0);
	}
	size_t numberOfParameters() const{ 
		return 1; 
	}

	SHARK_EXPORT_SYMBOL void train(LinearModel<>& model, LabeledData<RealVector, RealVector> const& dataset);
protected:
	double m_regularization;
};


}
#endif

