/*!
 * 
 *
 * \brief       BFGS<SearchPointType>
 * 
 * The Broyden, Fletcher, Goldfarb, Shannon (BFGS<SearchPointType>) algorithm is a
 * quasi-Newton method for unconstrained real-valued optimization.
 * 
 * 
 *
 * \author      O. Krause 
 * \date        2010
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
#include <shark/Algorithms/GradientDescent/BFGS.h>

namespace shark{

template<class SearchPointType>
void BFGS<SearchPointType>::initModel(){
	m_hessian.resize(this->m_dimension, this->m_dimension);
	noalias(m_hessian) = blas::identity_matrix<double>(this->m_dimension);
}
template<class SearchPointType>
void BFGS<SearchPointType>::computeSearchDirection(ObjectiveFunctionType const&){
	SearchPointType gamma = this->m_derivative - this->m_lastDerivative;
	SearchPointType delta = this->m_best.point - this->m_lastPoint;
	double d = inner_prod(gamma,delta);
	
	SearchPointType Hg = prod(m_hessian,gamma);
	
	//update hessian
	if (d < 1e-20){
		noalias(m_hessian) = blas::identity_matrix<double>(this->m_dimension);
	}else{
		double scale=inner_prod(gamma,Hg);
		scale = (scale / d + 1) / d;
		
		m_hessian += scale * outer_prod(delta,delta) 
			  - (outer_prod(Hg,delta)+outer_prod(delta,Hg))/d;

	}
	
	//compute search direction
	noalias(this->m_searchDirection) = -m_hessian % this->m_derivative;
}

//from ISerializable
template<class SearchPointType>
void BFGS<SearchPointType>::read( InArchive & archive ){
	AbstractLineSearchOptimizer<SearchPointType>::read(archive);
	archive>>m_hessian;
}

template<class SearchPointType>
void BFGS<SearchPointType>::write( OutArchive & archive ) const{
	AbstractLineSearchOptimizer<SearchPointType>::write(archive);
	archive<<m_hessian;
}

template class SHARK_EXPORT_SYMBOL BFGS<RealVector>;
template class SHARK_EXPORT_SYMBOL BFGS<FloatVector>;
}