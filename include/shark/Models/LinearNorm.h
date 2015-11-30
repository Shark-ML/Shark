/*!
 * \brief       Implements the linear norm class
 * 
 * \author      O. Krause
 * \date        2012
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
#ifndef SHARK_ML_MODEL_LINEAR_NORM_H
#define SHARK_ML_MODEL_LINEAR_NORM_H

#include <shark/Core/DLLSupport.h>
#include <shark/Models/AbstractModel.h>
namespace shark {
/*!
 *  \brief Normalizes the (non-negative) input by dividing by the overall sum.
 */
class LinearNorm : public AbstractModel<RealVector,RealVector>
{
private:
	struct InternalState: public State{
		RealVector norm;
		
		void resize(std::size_t patterns){
			norm.resize(patterns);
		}
	};
public:
	SHARK_EXPORT_SYMBOL LinearNorm();
	SHARK_EXPORT_SYMBOL LinearNorm(std::size_t inputSize);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Linear Norm"; }

	RealVector parameterVector()const{
		return RealVector();
	}
	void setParameterVector(RealVector const& newParameters){}

	std::size_t inputSize()const{
		return m_inputSize;
	}
	std::size_t outputSize()const{
		return m_inputSize;
	}
	std::size_t numberOfParameters()const{
		return 0;
	}
	
	void setStructure(std::size_t inputDimension){
		m_inputSize = inputDimension;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}

	using AbstractModel<RealVector,RealVector>::eval;
	SHARK_EXPORT_SYMBOL void eval(BatchInputType const& patterns,BatchOutputType& output)const;
	SHARK_EXPORT_SYMBOL void eval(BatchInputType const& patterns,BatchOutputType& output, State& state)const;

	void weightedParameterDerivative(
		BatchInputType const& patterns, BatchOutputType const& coefficients,  State const& state, RealVector& gradient
	)const{
		SIZE_CHECK(patterns.size1()==coefficients.size1());
		gradient.resize(0);
	}
	SHARK_EXPORT_SYMBOL void weightedInputDerivative(
		BatchInputType const& pattern,BatchOutputType const& coefficients, State const& state, BatchOutputType& gradient
	)const;

	/// From ISerializable, reads a model from an archive
	SHARK_EXPORT_SYMBOL void read( InArchive & archive );

	/// From ISerializable, writes a model to an archive
	SHARK_EXPORT_SYMBOL void write( OutArchive & archive ) const;

protected:
    std::size_t m_inputSize;
};

}
#endif
