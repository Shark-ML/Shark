/*!
 * 
 *
 * \brief       CG<SearchPointType>
 * 
 * Conjugate-gradient method for unconstraint optimization.
 * 
 * 
 *
 * \author      O. Krause
 * \date        2013
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
#include <shark/Core/DLLSupport.h>
#include <shark/Algorithms/GradientDescent/CG.h>

namespace shark{

template<class SearchPointType>
void CG<SearchPointType>::initModel(){
	m_count = 0;
}
template<class SearchPointType>
void CG<SearchPointType>::computeSearchDirection(ObjectiveFunctionType const&){
	//after numReset conjugent gradient steps, we reset automatically to the original gradient.
	//this ensure numerical stability near the optimum.
	m_count++;
	if (m_count == this->m_dimension)
	{
		m_count = 0;
		noalias(this->m_searchDirection) = -this->m_derivative;
		return;
	}
	
	//compute beta - see class documentation for the formula
	double gg = norm_sqr(this->m_derivative);
	double divisor = inner_prod(this->m_searchDirection, this->m_derivative - this->m_lastDerivative);
	//double divisor = norm_sqr(m_lastDerivative);
	if(gg == 0.0 || std::abs(divisor) <= 1.e-10*gg){
		m_count = 0;
		noalias(this->m_searchDirection) -= this->m_derivative;
		return;
	}
	double beta = gg/divisor;
	
	//Update search direction
	this->m_searchDirection *= beta;
	noalias(this->m_searchDirection) -= this->m_derivative;
}

//from ISerializable
template<class SearchPointType>
void CG<SearchPointType>::read( InArchive & archive ){
	AbstractLineSearchOptimizer<SearchPointType>::read(archive);
	archive>>m_count;
}
template<class SearchPointType>
void CG<SearchPointType>::write( OutArchive & archive ) const{
	AbstractLineSearchOptimizer<SearchPointType>::write(archive);
	archive <<m_count;
}

template class SHARK_EXPORT_SYMBOL CG<RealVector>;
template class SHARK_EXPORT_SYMBOL CG<FloatVector>;
#ifdef SHARK_USE_OPENCL
template class SHARK_EXPORT_SYMBOL CG<RealGPUVector>;
template class SHARK_EXPORT_SYMBOL CG<FloatGPUVector>;
#endif
}
