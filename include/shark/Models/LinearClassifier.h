#ifndef SHARK_ML_MODEL_LINEARCLASSIFIER_H
#define SHARK_ML_MODEL_LINEARCLASSIFIER_H

#include <shark/Models/LinearModel.h>
namespace shark {

/*! \brief Basic linear classifier.
 *
 *  The LinearClassifier class is a multi class classifier model
 *  suited for linear discriminant analysis. For c classes
 *  \f$ 0, \dots, c-1 \f$  the model computes
 *   
 *  \f$ \arg \max_i w_i^T x + b_i \f$
 *  
 *  Thus is it a linear model with arg max computation.
 *  Th internal linear model can be queried using linear().
 */ 
template<class VectorType = RealVector>
class LinearClassifier : public AbstractModel<VectorType,unsigned int>
{
protected:
	LinearModel<VectorType,RealVector> m_linear;
	typedef AbstractModel<VectorType,unsigned int> base_type;

public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
	LinearClassifier(){}
	

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearClassifier"; }
	
	LinearModel<VectorType,RealVector> const& linear()const{
		return m_linear;
	}
	LinearModel<VectorType,RealVector>& linear(){
		return m_linear;
	}
	
	size_t inputSize() const{
		return m_linear.inputSize();
	}
	
	/// obtain the parameter vector
	RealVector parameterVector() const{
		return m_linear.parameterVector();
	}

	/// overwrite the parameter vector
	void setParameterVector(RealVector const& newParameters){
		m_linear.setParameterVector(newParameters);
	}

	/// return the number of parameter
	std::size_t numberOfParameters() const{
		return m_linear.numberOfParameters();
	}
	
	/// Evaluate the model
	void eval(BatchInputType const& input, BatchOutputType& output)const{
		RealMatrix linearResult;
		m_linear.eval(input,linearResult);
		std::size_t batchSize = linearResult.size1();
		output.resize(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			output(i) = arg_min(row(linearResult,i));
		}
	}
	/// Evaluate the model:
	void eval(BatchInputType const& input, BatchOutputType& output, State& state)const{
		eval(input,output);
	}
	
	using base_type::eval;
};
}
#endif
