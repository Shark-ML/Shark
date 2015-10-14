/*!
 * 
 *
 * \brief       Implements a Model using a linear function.
 * 
 * 
 *
 * \author      T. Glasmachers, O. Krause
 * \date        2010-2011
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
#ifndef SHARK_MODELS_ConvexCombination_H
#define SHARK_MODELS_ConvexCombination_H

#include <shark/Models/AbstractModel.h>
namespace shark {


///
/// \brief Models a convex combination of inputs
///
/// For a given input vector x, the convex combination returns \f$ f_i(x) = sum_j w_{ij} x_j \f$,
/// where \f$ w_i > 0 \f$ and \f$ sum_j w_{ij} = 1\f$, that is the outputs of
/// the model are a convex combination of the inputs.
///
/// To ensure that the constraints are fulfilled, the model uses a different
/// set of weights q_i and \f$ w_{ij} = exp(q_{ij})/sum_j exp(q_{ik}) \f$. As usual, this
/// encoding is only used for the derivatives and the parameter vectors, not
/// when the weights are explicitely set. In the latter case, the user must provide
/// a set of suitable \f$ w_{ij} \f$.
class ConvexCombination : public AbstractModel<RealVector,RealVector>
{
private:
	RealMatrix m_w; ///< the convex comination weights. it holds sum(row(w_i)) = 1
public:

	/// CDefault Constructor; use setStructure later
	ConvexCombination(){
		m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
		m_features |= HAS_FIRST_INPUT_DERIVATIVE;
	}
	
	/// Constructor creating a model with given dimnsionalities and optional offset term.
	ConvexCombination(unsigned int inputs, unsigned int outputs = 1)
	: m_w(outputs,inputs,0.0){
		m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
		m_features |= HAS_FIRST_INPUT_DERIVATIVE;
	}
	
	/// Construction from matrix
	ConvexCombination(RealMatrix const& matrix):m_w(matrix){
		m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
		m_features |= HAS_FIRST_INPUT_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ConvexCombination"; }

	///swap
	friend void swap(ConvexCombination& model1,ConvexCombination& model2){
		swap(model1.m_w,model2.m_w);
	}

	///operator =
	ConvexCombination& operator=(ConvexCombination const& model){
		ConvexCombination tempModel(model);
		swap(*this,tempModel);
		return *this;
	}

	/// obtain the input dimension
	size_t inputSize() const{
		return m_w.size2();
	}

	/// obtain the output dimension
	size_t outputSize() const{
		return m_w.size1();
	}

	/// obtain the parameter vector
	RealVector parameterVector() const{
		RealVector ret(numberOfParameters());
		init(ret) << toVector(log(m_w));
		return ret;
	}

	/// overwrite the parameter vector
	void setParameterVector(RealVector const& newParameters)
	{
		init(newParameters) >> toVector(m_w);
		noalias(m_w) = exp(m_w);
		for(std::size_t i = 0; i != outputSize(); ++i){
			row(m_w,i) /= sum(row(m_w,i));
		}
	}

	/// return the number of parameter
	size_t numberOfParameters() const{
		return m_w.size1()*m_w.size2();
	}

	/// overwrite structure and parameters
	void setStructure(unsigned int inputs, unsigned int outputs = 1){
		ConvexCombination model(inputs,outputs);
		swap(*this,model);
	}
	
	RealMatrix const& weights() const{
		return m_w;
	}
	
	RealMatrix& weights(){
		return m_w;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	/// Evaluate the model: output = w * input
	void eval(BatchInputType const& inputs, BatchOutputType& outputs)const{
		outputs.resize(inputs.size1(),m_w.size1());
		axpy_prod(inputs,trans(m_w),outputs);
	}
	/// Evaluate the model: output = w *input
	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State& state)const{
		eval(inputs,outputs);
	}
	
	///\brief Calculates the first derivative w.r.t the parameters and summing them up over all patterns of the last computed batch 
	void weightedParameterDerivative(
		BatchInputType const& patterns, RealMatrix const& coefficients, State const& state, RealVector& gradient
	)const{
		SIZE_CHECK(coefficients.size2()==outputSize());
		SIZE_CHECK(coefficients.size1()==patterns.size1());

		gradient.resize(numberOfParameters());
		blas::dense_matrix_adaptor<double> weightGradient = blas::adapt_matrix(outputSize(),inputSize(),gradient.storage());

		//derivative is
		//sum_i sum_j c_ij sum_k x_ik grad_q w_jk= sum_k sum_j grad_q w_jk (sum_i c_ij x_ik)
		//and we set d_jk=sum_i c_ij x_ik => d = C^TX
		RealMatrix d(outputSize(),inputSize());
		axpy_prod(trans(coefficients), patterns,d);
		
		//use the same drivative as in the softmax model
		for(size_t i = 0; i != outputSize(); ++i){
			double mass=inner_prod(row(d,i),row(m_w,i));
			noalias(row(weightGradient,i)) = element_prod(
				row(d,i)-blas::repeat(mass,inputSize()),
				row(m_w,i)
			);
		}
	}
	///\brief Calculates the first derivative w.r.t the inputs and summs them up over all patterns of the last computed batch 
	void weightedInputDerivative(
		BatchInputType const & patterns,
		BatchOutputType const & coefficients,
		State const& state,
		BatchInputType& derivative
	)const{
		SIZE_CHECK(coefficients.size2() == outputSize());
		SIZE_CHECK(coefficients.size1() == patterns.size1());

		derivative.resize(patterns.size1(),inputSize());
		axpy_prod(coefficients,m_w,derivative);
	}

	/// From ISerializable
	void read(InArchive& archive){
		archive >> m_w;
	}
	/// From ISerializable
	void write(OutArchive& archive) const{
		archive << m_w;
	}
};


}
#endif
