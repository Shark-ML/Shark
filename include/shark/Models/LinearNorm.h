#ifndef SHARK_ML_MODEL_LINEAR_NORM_H
#define SHARK_ML_MODEL_LINEAR_NORM_H

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
	LinearNorm();
	LinearNorm(std::size_t inputSize);

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
	void eval(BatchInputType const& patterns,BatchOutputType& output)const;
	void eval(BatchInputType const& patterns,BatchOutputType& output, State& state)const;

	void weightedParameterDerivative(
		BatchInputType const& patterns, BatchOutputType const& coefficients,  State const& state, RealVector& gradient
	)const{
		SIZE_CHECK(patterns.size1()==coefficients.size1());
		gradient.resize(0);
	}
	void weightedInputDerivative(
		BatchInputType const& pattern,BatchOutputType const& coefficients, State const& state, BatchOutputType& gradient
	)const;

	/// From ISerializable, reads a model from an archive
	void read( InArchive & archive );

	/// From ISerializable, writes a model to an archive
	void write( OutArchive & archive ) const;

protected:
    std::size_t m_inputSize;
};

}
#endif
