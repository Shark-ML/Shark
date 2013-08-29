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
	
	LinearModelWrapperBase(bool offset, std::size_t outputSize)
	:m_offset(offset? outputSize:0,0.0){}

public:
	
	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearModelWrapperBase"; }

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
	LinearModelWrapper(unsigned int inputs, unsigned int outputs, bool offset)
	:base_type(offset,outputs),m_matrix(outputs,inputs,0.0){
		if(!blas::traits::IsSparse<Matrix>::value){
			base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		}
	}

	LinearModelWrapper(Matrix const& matrix)
	: m_matrix(matrix){
		if(!blas::traits::IsSparse<Matrix>::value){
			base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		}
	}

	LinearModelWrapper(Matrix const& matrix, OutputType const& offset)
	: base_type(offset)
	, m_matrix(matrix){
		if(!blas::traits::IsSparse<Matrix>::value){
			base_type::m_features |= base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		}
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearModelWrapper"; }

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
		if(blas::traits::IsSparse<Matrix>::value){
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
	
	
	/// \param inputs Batch of input patterns
	/// \param outputs Outputs corresponding to input batch
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
		SHARK_CHECK(!blas::traits::IsSparse<Matrix>::value,"Derivative for sparse matrices is not implemented.");
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
