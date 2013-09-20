//===========================================================================
/*!
 *  \brief Soft-max transformation.
 *
 *  \author O. Krause, T. Glasmachers
 *  \date 2010-2011
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
//===========================================================================


#ifndef SHARK_MODELS_SOFTMAX_H
#define SHARK_MODELS_SOFTMAX_H

#include <shark/Models/AbstractModel.h>
namespace shark {


///
/// \brief Softmax function
///
/// \par
/// Squash an n-dimensional real vector space
/// to the (n-1)-dimensional probability simplex:
/// \f[
///      f_i(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
/// \f]
/// This also corresponds to the exponential norm of the input.
///
class Softmax : public AbstractModel<RealVector,RealVector>
{
private:
	struct InternalState : public State{
		RealMatrix results;

		void resize(std::size_t numPatterns,std::size_t inputs){
			results.resize(numPatterns,inputs);
		}
	};

public:
	/// Constructor
	Softmax(size_t dim);
	/// Constructor
	Softmax();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Softmax"; }

	RealVector parameterVector()const{
		return RealVector();
	}
	void setParameterVector(RealVector const& newParameters){
		SIZE_CHECK(newParameters.size()==0);
	}

	size_t inputSize()const{
		return m_inputSize;
	}
	size_t outputSize()const{
		return m_inputSize;
	}
	size_t numberOfParameters()const{
		return 0;
	}

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}

	void eval(BatchInputType const& patterns,BatchOutputType& output)const;
	void eval(BatchInputType const& patterns,BatchOutputType& output, State & state)const;
	using AbstractModel<RealVector,RealVector>::eval;
	
	void weightedParameterDerivative(
		BatchInputType const& patterns, BatchOutputType const& coefficients,  State const& state, RealVector& gradient
	)const;
	void weightedInputDerivative(
		BatchInputType const& patterns, RealMatrix const& coefficients,  State const& state, BatchOutputType& gradient
	)const;

	void setStructure(std::size_t inputSize){
		m_inputSize = inputSize;
	}
	
	/// From ISerializable, reads a model from an archive
	void read( InArchive & archive );

	/// From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const;
	
private:
	std::size_t m_inputSize;
};

}
#endif
