/*!
 * \brief       Implements a simple sigmoidal model for sigmoidal fitting in a 2-d problem
 * 
 * \author     
 * \date        
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
#ifndef SHARK_MODEL_ML_SIGMOIDMODEL_H
#define SHARK_MODEL_ML_SIGMOIDMODEL_H

#include <shark/Core/DLLSupport.h>
#include <shark/Models/AbstractModel.h>
namespace shark {


//! \brief Standard sigmoid function.
//!
//! \par
//! This model maps a real-valued input to the unit interval by the sigmoid function
//! \f$ f(x) = \frac{1}{1 + \exp(-Ax+B))} \f$,
//! where the real-valued model parameter A controls the slope, and the real-valued
//! offset model parameter B controls the position of the symmetry point.
//! This is a special case of a feed forward neural network consisting of
//! a single sigmoid layer.
//! Note that the parameter A is expected to be non-negative 
//! (and hence does not incorporate the minus sign in the sigmoid's equation).
//! Also, the offset parameter can be disabled using the setOffsetActivity() 
//! member function.
//!
//! \sa TanhSigmoidModel SimpleSigmoidModel
class SigmoidModel : public AbstractModel<RealVector,RealVector>
{
private:
	struct InternalState:public State{
		RealVector result;
		
		void resize(std::size_t patterns){
			result.resize(patterns);
		}
	};
public:

	//! default ctor
	//! \param transform_for_unconstrained when a new paramVector is set, should the exponent of the first parameter be used as the sigmoid's slope?
	SHARK_EXPORT_SYMBOL SigmoidModel( bool transform_for_unconstrained = true );

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SigmoidModel"; }

	SHARK_EXPORT_SYMBOL RealVector parameterVector() const;
	
	//! note that the parameters are not expected to incorporate the minus sign in the sigmoid's equation
	//! \param newParameters the new parameter vector A and offset B concatenated
	SHARK_EXPORT_SYMBOL void setParameterVector(RealVector const& newParameters);
	
	std::size_t numberOfParameters() const {
		return 2; //we always return 2, even if the offset is hard-clamped to zero.
	}
	// \brief whether to use the offset, or clamp it to zero. offset is active by default.
	SHARK_EXPORT_SYMBOL void setOffsetActivity( bool enable_offset );
	
	bool hasOffset()const{
		return m_useOffset;
	}
	
	bool slopeIsExpEncoded()const{
		return m_transformForUnconstrained;
	}
	
	/*!
	*  \brief activation function \f$g_{output}(x)\f$
	*/
	SHARK_EXPORT_SYMBOL virtual double sigmoid(double x)const;
	/*!
	*  \brief Computes the derivative of the activation function
	*         \f$g_{output}(x)\f$ for the output given the
	*		  last response of the model gx=g(x)
	*/
	virtual double sigmoidDerivative(double gx)const;
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}

	SHARK_EXPORT_SYMBOL void eval(BatchInputType const&pattern, BatchOutputType& output, State& state)const;
	SHARK_EXPORT_SYMBOL void eval(BatchInputType const&pattern, BatchOutputType& output)const;
	using AbstractModel<RealVector,RealVector>::eval;

	SHARK_EXPORT_SYMBOL void weightedParameterDerivative(
		BatchInputType const& pattern, BatchOutputType const& coefficients, State const& state, RealVector& gradient
	)const;
	
	SHARK_EXPORT_SYMBOL void weightedInputDerivative(
		BatchInputType const& pattern, BatchOutputType const& coefficients, State const& state, BatchInputType& derivative
	)const;
	
	std::size_t inputSize()const{
		return 1;
	}
	std::size_t outputSize()const{
		return 1;
	}
	
	//! set the minimum log value that should be returned as log-encoded slope if the true slope is actually zero. default in ctor sets -230.
	//! param logvalue the new minimum log value
	void setMinLogValue( double logvalue = -230.0 );

	/// From ISerializable, reads a model from an archive
	void read( InArchive & archive );

	/// From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const;

protected:
	RealVector m_parameters; ///< the parameter vector
	bool m_useOffset; ///< whether or not to allow non-zero offset values
    
	bool m_transformForUnconstrained; ///< flag for encoding variant
	double m_minLogValue; ///< what value should be returned as log-encoded slope if the true slope is actually zero

};


//! \brief Simple sigmoid function
//!
//! \par
//! This model maps the reals to the unit interval by the sigmoid function
//! \f$ f(x) =  \frac{1}{2} \frac{st}{1+|<A,x>+b|} + \frac{1}{2} \f$.
class SimpleSigmoidModel : public SigmoidModel
{
public:
	SHARK_EXPORT_SYMBOL SimpleSigmoidModel( bool transform_for_unconstrained = true );
	SHARK_EXPORT_SYMBOL double sigmoid(double a)const;
	SHARK_EXPORT_SYMBOL double sigmoidDerivative(double ga)const;

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SimpleSigmoidModel"; }
};

//! \brief scaled Tanh sigmoid function
//!
//! \par
//! This model maps the reals to the unit interval by the sigmoid function
//! \f$ f(x) =  \frac{1}{2} \tanh(<A,x>+b) + \frac{1}{2} \f$.
class TanhSigmoidModel : public SigmoidModel
{
public:
	SHARK_EXPORT_SYMBOL TanhSigmoidModel( bool transform_for_unconstrained = true );
	SHARK_EXPORT_SYMBOL double sigmoid(double a)const;
	SHARK_EXPORT_SYMBOL double sigmoidDerivative(double ga)const;

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "TanhSigmoidModel"; }
};
}
#endif

