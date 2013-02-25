/*!
* \brief Implements a Model using a linear function.
*
*  \author  T. Glasmachers, O. Krause
*  \date    2010-2011
*
*  \par Copyright (c) 1999-2011:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-27974<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
*
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
#include <shark/Models/AbstractModel.h>
#include <shark/LinAlg/BLAS/Initialize.h>
namespace shark {

namespace detail {

/// \brief Baseclass of LinearModelWrapper.
/// This class is needed for the type
/// erasure in LinearModel.
template <class InputType, class OutputType>
class LinearModelWrapperBase : public AbstractModel<InputType, OutputType>
{
protected:
	OutputType m_offset;

	LinearModelWrapperBase()
	{ }

	LinearModelWrapperBase(OutputType const& offset)
	: m_offset(offset)
	{ }

public:
	bool hasOffset() const
	{ return (m_offset.size() > 0); }

	virtual std::size_t inputSize() const = 0;
	virtual std::size_t outputSize() const = 0;
	virtual void matrixRow(std::size_t index, RealVector& row) const = 0;
	virtual void matrixColumn(std::size_t index, RealVector& column) const = 0;
	virtual void matrix(RealMatrix& mat) const = 0;
	OutputType const& offset() const
	{ return m_offset; }

	void read(InArchive& archive){
		archive >> m_offset;
	}
	void write(OutArchive& archive) const{
		archive << m_offset;
	}
	
	//workaround for base pointer serialization
	template<class Archive>
	void serialize(Archive& ar,unsigned int version){
		ar & m_offset;
	}

	virtual LinearModelWrapperBase<InputType, OutputType>* clone()const = 0;
};

///\brief Implementation of the Linear Model for specific matrix types
///
/// The Matrix template parameter is a real matrix.
/// It is open to sparse and dense representations.
template <class Matrix, class InputType, class OutputType>
class LinearModelWrapper : public LinearModelWrapperBase<InputType, OutputType>
{
	typedef LinearModelWrapperBase<InputType, OutputType> base_type;
	typedef typename Batch<InputType>::type BatchInputType;
	typedef typename Batch<OutputType>::type BatchOutputType;
public:
	LinearModelWrapper(){}
	LinearModelWrapper(unsigned int inputs, unsigned int outputs, bool offset){
		m_matrix = RealZeroMatrix(outputs, inputs);
		base_type::m_offset = RealZeroVector(offset ? outputs : 0);
		if(!traits::IsSparse<Matrix>::value){
			base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
			//base_type::m_features |= base_type::HAS_SECOND_PARAMETER_DERIVATIVE;
		}
		base_type::m_name = "LinearModelWrapper";
	}

	LinearModelWrapper(Matrix const& matrix)
	: m_matrix(matrix){
		if(!traits::IsSparse<Matrix>::value){
			base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
			//base_type::m_features |= base_type::HAS_SECOND_PARAMETER_DERIVATIVE;
		}
		base_type::m_name = "LinearModelWrapper";
	}

	LinearModelWrapper(Matrix const& matrix, OutputType offset)
	: base_type(offset)
	, m_matrix(matrix){
		if(!traits::IsSparse<Matrix>::value){
			base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
			//base_type::m_features |= base_type::HAS_SECOND_PARAMETER_DERIVATIVE;
		}
		base_type::m_name = "LinearModelWrapper";
	}

	std::size_t inputSize() const{ 
		return m_matrix.size2(); 
	}
	std::size_t outputSize() const{ 
		return m_matrix.size1(); 
	}
	void matrix(RealMatrix& mat) const{ 
		mat = m_matrix; 
	}
	void matrixRow(std::size_t index, RealVector& rowStorage) const{ 
		rowStorage = row(m_matrix, index); 
	}
	void matrixColumn(std::size_t index, RealVector& columnStorage) const{ 
		columnStorage = column(m_matrix, index); 
	}

	RealVector parameterVector() const{
		RealVector ret(numberOfParameters());
		if(base_type::hasOffset())
			init(ret) << toVector(m_matrix),base_type::m_offset;
		else
			init(ret) << toVector(m_matrix);
		
		return ret;
	}

	void setParameterVector(RealVector const& newParameters){
		SHARK_CHECK(newParameters.size() == numberOfParameters(), "[LinearModelWrapper::setParameterVector] invalid size of the parameter vector");
		if(base_type::hasOffset())
			init(newParameters) >> toVector(m_matrix),base_type::m_offset;
		else
			init(newParameters) >> toVector(m_matrix);
	}

	std::size_t numberOfParameters() const{
		if(traits::IsSparse<Matrix>::value){
			std::size_t elements = base_type::m_offset.size();
			//works for sparse, but not very efficient for big dense matrices
			for(typename Matrix::const_iterator1 pos = m_matrix.begin1(); pos != m_matrix.end1(); ++pos){
				typename Matrix::const_iterator2 pos1Iter=pos.begin();
				typename Matrix::const_iterator2 pos2Iter=pos.end();
				elements += std::distance(pos1Iter,pos2Iter);
			}
			return elements;
		}
		else{
			return m_matrix.size1()*m_matrix.size2()+base_type::m_offset.size();
		}
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	void eval(BatchInputType const& inputs, BatchOutputType& outputs)const{
		outputs.resize(inputs.size1(),m_matrix.size1()
					,false   // Oh thanks ublas for making this necessary!!! damn!
				);
		//we multiply with a set of row vectors from the left
		fast_prod(inputs,trans(m_matrix),outputs);
		if (base_type::hasOffset()){
			noalias(outputs)+=repeat(base_type::m_offset,inputs.size1());
		}
	}
	
	
	/// \param state Batch of input patterns
	/// \param state Outputs corresponding to input batch
	/// \param state Internal state (e.g., for passing to derivative computation) of the model
	void eval(BatchInputType const& inputs, BatchOutputType& outputs, State& state)const{
		eval(inputs,outputs);
	}

	void weightedParameterDerivative(
		BatchInputType const& patterns, RealMatrix const& coefficients, State const& state, RealVector& gradient
	)const{
		//todo: doesn't work for sparse.
		SIZE_CHECK(coefficients.size2()==outputSize());
		SIZE_CHECK(coefficients.size1()==patterns.size1());
		SHARK_CHECK(!traits::IsSparse<Matrix>::value,"Derivative for sparse matrices is not implemented.");
		gradient.resize(numberOfParameters());
		std::size_t inputs = inputSize();
		std::size_t outputs = outputSize();
		gradient.clear();
		std::size_t first = 0;
		for (std::size_t output = 0; output < outputs; output++){
			//sum_i coefficients(output,i)*pattern(i))
			fast_prod(trans(patterns),column(coefficients,output),subrange(gradient, first, first + inputs),0.0);
			first += inputs;
		}
		if (base_type::hasOffset()){
			for(std::size_t i = 0; i != patterns.size1();++i){
				noalias(subrange(gradient, first, first + outputs))+= row(coefficients,i);
			}
		}
	}
	void weightedParameterDerivative(
		BatchInputType const & pattern,
		BatchOutputType const & coefficients,
		Batch<RealMatrix>::type const & errorHessian,//maybe a batch of matrices is bad?,
		State const& state,
		RealVector& derivative,
		RealMatrix& hessian
	)const{
		//~ //todo: doesn't work for sparse.
		//~ SIZE_CHECK(coefficients.size()==outputSize());
		//~ SIZE_CHECK(errorHessian.size1()==outputSize() && errorHessian.size2()==outputSize());
		//~ derivative.resize(numberOfParameters());
		//~ hessian.resize(numberOfParameters(),numberOfParameters());
		//~ hessian.clear();
		//~ std::size_t inputs = inputSize();
		//~ std::size_t outputs = outputSize();
		//~ std::size_t offsetStart = inputs*outputs;
		//~ std::size_t offsetEnd = offsetStart+outputs;

		//~ for (std::size_t i = 0; i < outputs; i++) {
			//~ std::size_t startI = i*inputs;
			//~ std::size_t endI = startI + inputs;
			//~ subrange(derivative, startI, endI) = coefficients(i) * pattern;

			//~ //The creation of the 2nd derivative matrix is a bit complicated. what it does is the following:
			//~ //it calculates sum_ij  HE_ij*dF_i^T*dF_j
			//~ //where HE is the hessian of the loss function and dF_i is the the derivative of the ith output of the Model
			//~ //Because we don't save the weights for one output continuously, the derivative looks like this:
			//~ //dF_1 = [x_1,x_2,0,0,b_1,0]
			//~ //dF_2 = [0,0,x_1,x_2,0,b_2]
			//~ //where x is a pattern and b the bias weight. this is the deriavtive of a 2x2 model
			//~ //the outer product of these two than has entries in this form:
			//~ //[0,0,0,0,0,0]
			//~ //[0,0,0,0,0,0]
			//~ //[1,1,0,0,1,0]
			//~ //[1,1,0,0,1,0]
			//~ //[0,0,0,0,0,0]
			//~ //[1,1,0,0,1,0]
			//~ //and this is what the code below computes
			//~ for (std::size_t j = 0; j < outputs; j++) {
				//~ std::size_t startJ = j*inputs;
				//~ std::size_t endJ = startJ + inputs;

				//~ subrange(hessian, startI, endI,startJ,endJ)  += errorHessian(i,j) * outer_prod(pattern,pattern);
				//~ if (base_type::hasOffset()) {
					//~ for(std::size_t k =0; k != inputs; ++k){
						//~ hessian(offsetStart+i,startJ+k)=hessian(startI+k,offsetStart+j) = errorHessian(i,j) *pattern(k);
					//~ }
					//~ hessian(offsetStart+i,offsetStart+j) = errorHessian(i,j);

				//~ }
			//~ }
		//~ }
		//~ if (base_type::hasOffset()) {
			//~ subrange(derivative, offsetStart, offsetEnd) = coefficients;
		//~ }
	}

	void read(InArchive& archive){
		archive >> boost::serialization::base_object<base_type>(*this);
		archive >> m_matrix;
	}
	void write(OutArchive& archive) const{
		//VS2010 workaround, produces warnings in gcc :(
		archive << boost::serialization::base_object<base_type>(
			const_cast<LinearModelWrapper<Matrix,InputType, OutputType>& >(*this)
		);
//		archive & boost::serialization::base_object<base_type >(*this);
		archive << m_matrix;
	}
	
	/**
	* \brief Versioned loading of components, calls read(...).
	*/
	void load(InArchive & archive,unsigned int version){
		read(archive);
	}

	/**
	* \brief Versioned storing of components, calls write(...).
	*/
	void save(OutArchive & archive,unsigned int version)const{
		write(archive);
	}
	BOOST_SERIALIZATION_SPLIT_MEMBER();

	LinearModelWrapperBase<InputType, OutputType>* clone()const{
		return new LinearModelWrapper<Matrix,InputType, OutputType>(*this);
	}

protected:
	Matrix m_matrix;
};


}
}
