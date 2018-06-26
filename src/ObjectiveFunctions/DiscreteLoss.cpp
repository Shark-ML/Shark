//===========================================================================
/*!
 * 
 *
 * \brief       Flexible error measure for classication tasks
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2011
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#define SHARK_COMPILE_DLL
#include <shark/ObjectiveFunctions/Loss/DiscreteLoss.h>
#include <boost/lambda/lambda.hpp>
using namespace shark;


DiscreteLoss::DiscreteLoss(RealMatrix const& cost){
	this->m_cost = cost;
	defineCostMatrix(cost);
}


double DiscreteLoss::eval(BatchLabelType const& target, BatchOutputType const& prediction) const{
	SIZE_CHECK(target.size() == prediction.size());
	
//	return accumulateError(target,prediction,boost::bind<double>(boost::ref(m_cost),boost::lambda::_1,boost::lambda::_2));
	double error = 0;
	for(std::size_t i = 0; i != prediction.size(); ++i){
		error += m_cost(target(i), prediction(i));
	}
	return error;
}

void DiscreteLoss::defineCostMatrix(RealMatrix const& cost){
	// check validity
	std::size_t size = cost.size1();
	SHARK_ASSERT(cost.size2() == size);
	for (std::size_t i = 0; i != size; i++){
		for (std::size_t j = 0; j != size; j++){
			SHARK_ASSERT(cost(i, j) >= 0.0);
		}
		SHARK_ASSERT(cost(i, i) == 0.0);
	}
	m_cost = cost;
}

void DiscreteLoss::defineBalancedCost(Data<unsigned int> const& labels){
	std::size_t classes = numberOfClasses(labels);
	std::size_t ic = labels.numberOfElements();
	
	std::vector<std::size_t> freq = classSizes(labels);

	m_cost.resize(classes, classes);
	for (std::size_t i = 0; i!= classes; i++){
		double c = (freq[i] == 0) ? 1.0 : ic / (double)(classes * freq[i]);
		for ( std::size_t j = 0; j != classes; j++) 
			m_cost(i, j) = c;
		m_cost(i, i) = 0.0;
	}
}
