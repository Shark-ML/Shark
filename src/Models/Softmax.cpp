//===========================================================================
/*!
 * 
 *
 * \brief       Soft-max transformation.
 * 
 * 
 *
 * \author      O. Krause, T. Glasmachers
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
//===========================================================================
#include <shark/Models/Softmax.h>

using namespace shark;
using namespace std;

Softmax::Softmax(size_t dim){
	m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
	m_features|=HAS_FIRST_INPUT_DERIVATIVE;
	setStructure(dim);
}
Softmax::Softmax(){
	m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
	m_features|=HAS_FIRST_INPUT_DERIVATIVE;
}

void Softmax::eval(BatchInputType const& patterns,BatchOutputType& outputs)const{

	SIZE_CHECK(patterns.size2() == inputSize());
	if(inputSize() == 1){
		outputs.resize(patterns.size1(),2);
		for(std::size_t i = 0; i != patterns.size1();++i){
			outputs(i,0) = exp(patterns(i,0));
			outputs(i,1) = 1/outputs(i,0);
		}
	}else{
		outputs.resize(patterns.size1(),inputSize());
		noalias(outputs) = exp(patterns);
	}
	
	for(size_t i = 0; i != patterns.size1(); ++i){
		row(outputs,i) /= sum(row(outputs,i));
	}
	
}

void Softmax::eval(BatchInputType const& patterns,BatchOutputType& outputs, State& state)const{
	eval(patterns,outputs);
	InternalState& s = state.toState<InternalState>();
	s.resize(patterns.size1(),inputSize());
	noalias(s.results) = outputs;
}

void Softmax::weightedParameterDerivative(
	BatchInputType const& patterns, BatchOutputType const& coefficients, State const& state, RealVector& gradient
)const{
	SIZE_CHECK(patterns.size2() == inputSize());
	SIZE_CHECK(coefficients.size2()==patterns.size2());
	SIZE_CHECK(coefficients.size1()==patterns.size1());

	gradient.resize(0);
}
void Softmax::weightedInputDerivative(
	BatchInputType const& patterns, BatchOutputType const& coefficients, State const& state, BatchOutputType& gradient
)const{
	SIZE_CHECK(patterns.size2() == inputSize());
	SIZE_CHECK(coefficients.size2()==patterns.size2());
	SIZE_CHECK(coefficients.size1()==patterns.size1());
	InternalState const& s = state.toState<InternalState>();
	gradient.resize(patterns.size1(),inputSize());
	gradient.clear();
	if(inputSize() ==1){
		for(size_t i = 0; i != patterns.size1(); ++i){
			double sdx= s.results(i,0)*(1-s.results(i,0));
			gradient(i,0) = coefficients(i,1)+(coefficients(i,0)-coefficients(i,1))*sdx;
		}
	}
	else{
		for(size_t i = 0; i != patterns.size1(); ++i){
			double mass=inner_prod(row(coefficients,i),row(s.results,i));
			//(c_k-m)*f_k
			noalias(row(gradient,i))=element_prod(
				row(coefficients,i)-blas::repeat(mass,inputSize()),
				row(s.results,i)
			);
		}
	}
}

/// From ISerializable, reads a model from an archive
void Softmax::read( InArchive & archive ){
	archive >> m_inputSize;
}

/// From ISerializable, writes a model to an archive
void Softmax::write( OutArchive & archive ) const{
	archive << m_inputSize;
}
