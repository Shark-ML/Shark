/*!
 * \brief       Implements the Linear Classifier class of the shark library
 * 
 * \author      O. Krause
 * \date        2013
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
#ifndef SHARK_ML_MODEL_LINEARCLASSIFIER_H
#define SHARK_ML_MODEL_LINEARCLASSIFIER_H

#include <shark/Models/LinearModel.h>
#include <shark/Models/Converter.h>
namespace shark {

/*! \brief Basic linear classifier.
 *
 *  The LinearClassifier class is a multi class classifier model
 *  suited for linear discriminant analysis. For c classes
 *  \f$ 0, \dots, c-1 \f$  the model computes
 *   
 *  \f$ \arg \max_i w_i^T x + b_i \f$
 *  
 *  Thus is it a linear model with arg max computation.
 *  The internal linear model can be queried using decisionFunction().
 */ 
template<class VectorType = RealVector>
class LinearClassifier : public ArgMaxConverter<LinearModel<VectorType> >
{
public:
	LinearClassifier(){}

	std::string name() const
	{ return "LinearClassifier"; }
};
}
#endif
