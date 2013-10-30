//===========================================================================
/*!
*
*  \brief Weighted sum of m_base kernels.
*
*  \author  T.Glasmachers, O. Krause, M. Tuma
*  \date    2010, 2011
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
//===========================================================================

#ifndef SHARK_MODELS_KERNELS_WEIGHTED_SUM_KERNEL_H
#define SHARK_MODELS_KERNELS_WEIGHTED_SUM_KERNEL_H


#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Core/Traits/IsVector.h>

#include <boost/utility/enable_if.hpp>
namespace shark {


/// \brief Weighted sum of kernel functions
///
/// For a set of positive definite kernels \f$ k_1, \dots, k_n \f$
/// with positive coeffitients \f$ w_1, \dots, w_n \f$ the sum
/// \f[ \tilde k(x_1, x_2) := \sum_{i=1}^{n} w_i \cdot k_i(x_1, x_2) \f]
/// is again a positive definite kernel function.
/// Internally, the weights are represented as
/// \f$ w_i = \exp(\xi_i) \f$
/// to allow for unconstrained optimization.
///
/// This variant of the weighted sum kernel eleminates one redundant
/// degree of freedom by fixing the first kernel weight to 1.0.
///
/// The result of the kernel evaluation is devided by the sum of the
/// kernel weights, so that in total, this amounts to fixing the sum
/// of the of the weights to one.
///
template<class InputType=RealVector>
class WeightedSumKernel : public AbstractKernelFunction<InputType>
{
private:
	typedef AbstractKernelFunction<InputType> base_type;

	struct InternalState: public State{
		RealMatrix result;
		std::vector<RealMatrix> kernelResults;
		std::vector<boost::shared_ptr<State> > kernelStates;

		InternalState(std::size_t numSubKernels)
		:kernelResults(numSubKernels),kernelStates(numSubKernels){}

		void resize(std::size_t sizeX1, std::size_t sizeX2){
			result.resize(sizeX1, sizeX2);
			for(std::size_t i = 0; i != kernelResults.size(); ++i){
				kernelResults[i].resize(sizeX1, sizeX2);
			}
		}
	};
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstInputReference ConstInputReference;
	typedef typename base_type::ConstBatchInputReference ConstBatchInputReference;

	WeightedSumKernel(const std::vector<AbstractKernelFunction<InputType>* >& base){
		SHARK_CHECK( base.size() > 0, "[WeightedSumKernel::WeightedSumKernel] There should be at least one sub-kernel.");

		m_base.resize( base.size() );
		m_numParameters = m_base.size()-1;

		for (std::size_t i=0; i != m_base.size() ; i++) {
			SHARK_ASSERT( base[i] != NULL );
			m_base[i].kernel = base[i];
			m_base[i].weight = 1.0;
			m_base[i].adaptive = false;
		}
		m_weightsum = m_base.size();

		// set m_base class features
		bool hasFirstParameterDerivative = true;
		for ( unsigned int i=0; i<m_base.size(); i++ ){
			if ( !m_base[i].kernel->hasFirstParameterDerivative() ) {
				hasFirstParameterDerivative = false;
				break;
			}
		}
		bool hasFirstInputDerivative = true;
		for ( unsigned int i=0; i<m_base.size(); i++ ){
			if ( !m_base[i].kernel->hasFirstInputDerivative() ) {
				hasFirstInputDerivative = false;
				break;
			}
		}

		if ( hasFirstParameterDerivative )
			this->m_features|=base_type::HAS_FIRST_PARAMETER_DERIVATIVE;

		if ( hasFirstInputDerivative )
			this->m_features|=base_type::HAS_FIRST_INPUT_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "WeightedSumKernel"; }

	/// Check whether m_base kernel index is adaptive.
	bool isAdaptive( std::size_t index ) const {
		return m_base[index].adaptive;
	}
	/// Set adaptivity of m_base kernel index.
	void setAdaptive( std::size_t index, bool b = true ) {
		m_base[index].adaptive = b;
		updateNumberOfParameters();
	}
	/// Set adaptivity of all m_base kernels.
	void setAdaptiveAll( bool b = true ) {
		for (std::size_t i=0; i!= m_base.size(); i++)
			m_base[i].adaptive = b;
		updateNumberOfParameters();
	}

	/// Get the weight of a kernel
	double weight(std::size_t index){
		RANGE_CHECK(index < m_base.size());
		return m_base[index].weight;
	}

	/// return the parameter vector. The first N-1 entries are the (log-encoded) kernel
	/// weights, the sub-kernel's parameters are stacked behind each other after that.
	RealVector parameterVector() const {
		std::size_t num = numberOfParameters();
		RealVector ret(num);

		std::size_t index = 0;
		for (; index != m_base.size()-1; index++){
			ret(index) = std::log(m_base[index+1].weight);

		}
		for (std::size_t i=0; i != m_base.size(); i++){
			if (m_base[i].adaptive){
				std::size_t n = m_base[i].kernel->numberOfParameters();
				subrange(ret,index,index+n) = m_base[i].kernel->parameterVector();
				index += n;
			}
		}
		return ret;
	}

	///\brief creates the internal state of the kernel
	boost::shared_ptr<State> createState()const{
		InternalState* state = new InternalState(m_base.size());
		for(std::size_t i = 0; i != m_base.size(); ++i){
			state->kernelStates[i]=m_base[i].kernel->createState();
		}
		return boost::shared_ptr<State>(state);
	}

	/// set the parameter vector. The first N-1 entries are the (log-encoded) kernel
	/// weights, the sub-kernel's parameters are stacked behind each other after that.
	void setParameterVector(RealVector const& newParameters) {
		SIZE_CHECK(newParameters.size() == numberOfParameters());

		m_weightsum = 1.0;
		std::size_t index = 0;
		for (; index != m_base.size()-1; index++){
			double w = newParameters(index);
			m_base[index+1].weight = std::exp(w);
			m_weightsum += m_base[index+1].weight;

		}
		for (std::size_t i=0; i != m_base.size(); i++){
			if (m_base[i].adaptive){
				std::size_t n = m_base[i].kernel->numberOfParameters();
				m_base[i].kernel->setParameterVector(subrange(newParameters,index,index+n));
				index += n;
			}
		}
	}

	std::size_t numberOfParameters() const {
		return m_numParameters;
	}

	/// Evaluate the weighted sum kernel (according to the following equation:)
	/// \f$ k(x, y) = \frac{\sum_i \exp(w_i) k_i(x, y)}{sum_i exp(w_i)} \f$
	double eval(ConstInputReference x1, ConstInputReference x2) const{
		double numerator = 0.0;
		for (std::size_t i=0; i != m_base.size(); i++){
			double result = m_base[i].kernel->eval(x1, x2);
			numerator += m_base[i].weight*result;
		}
		return numerator / m_weightsum;
	}

	/// Evaluate the kernel according to the equation:
	/// \f$ k(x, y) = \frac{\sum_i \exp(w_i) k_i(x, y)}{sum_i exp(w_i)} \f$
	/// for two batches of inputs.
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result) const{
		std::size_t sizeX1=shark::size(batchX1);
		std::size_t sizeX2=shark::size(batchX2);
		ensure_size(result,sizeX1,sizeX2);
		result.clear();

		RealMatrix kernelResult(sizeX1,sizeX2);
		for (std::size_t i = 0; i != m_base.size(); i++){
			m_base[i].kernel->eval(batchX1, batchX2,kernelResult);
			result += m_base[i].weight*kernelResult;
		}
		result /= m_weightsum;
	}

	/// Evaluate the kernel according to the equation:
	/// \f$ k(x, y) = \frac{\sum_i \exp(w_i) k_i(x, y)}{sum_i exp(w_i)} \f$
	/// for two batches of inputs.
	/// (see the documentation of numberOfIntermediateValues for the workings of the intermediates)
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result, State& state) const{
		std::size_t sizeX1=shark::size(batchX1);
		std::size_t sizeX2=shark::size(batchX2);
		ensure_size(result,sizeX1,sizeX2);
		result.clear();

		InternalState& s = state.toState<InternalState>();
		s.resize(sizeX1,sizeX2);

		for (std::size_t i=0; i != m_base.size(); i++){
			m_base[i].kernel->eval(batchX1,batchX2,s.kernelResults[i],*s.kernelStates[i]);
			result += m_base[i].weight*s.kernelResults[i];
		}
		//store summed result
		s.result=result;
		result /= m_weightsum;
	}

	void weightedParameterDerivative(
		ConstBatchInputReference batchX1,
		ConstBatchInputReference batchX2,
		RealMatrix const& coefficients,
		State const& state,
		RealVector& gradient
	) const{
		ensure_size(gradient,numberOfParameters());

		std::size_t numKernels = m_base.size(); //how far the loop goes;

		InternalState const& s = state.toState<InternalState>();

		double sumSquared = sqr(m_weightsum); //denominator

		//first the derivative with respect to the (log-encoded) weight parameter
		//the first weight is not a parameter and does not need a gradient.
		//[Theoretically, we wouldn't need to store its result .]
		//calculate the weighted sum over all results
		double numeratorSum = sum(element_prod(coefficients,s.result));
		for (std::size_t i = 1; i != numKernels; i++) {
			//calculate the weighted sum over all results of this kernel
			double summedK=sum(element_prod(coefficients,s.kernelResults[i]));
			gradient(i-1) = m_base[i].weight * (summedK * m_weightsum - numeratorSum) / sumSquared;
		}

		std::size_t gradPos = numKernels-1; //starting position of subkerel gradient
		RealVector kernelGrad;
		for (std::size_t i=0; i != numKernels; i++) {
			if (isAdaptive(i)){
				double coeff = m_base[i].weight / m_weightsum;
				m_base[i].kernel->weightedParameterDerivative(batchX1,batchX2,coefficients,*s.kernelStates[i],kernelGrad);
				std::size_t n = kernelGrad.size();
				noalias(subrange(gradient,gradPos,gradPos+n)) = coeff * kernelGrad;
				gradPos += n;
			}
		}
	}

	/// Input derivative, calculated according to the equation:
	/// <br/>
	/// \f$ \frac{\partial k(x, y)}{\partial x}
	///     \frac{\sum_i \exp(w_i) \frac{\partial k_i(x, y)}{\partial x}}
	///          {\sum_i exp(w_i)} \f$
	/// and summed up over all  of the second batch
	void weightedInputDerivative(
		ConstBatchInputReference batchX1,
		ConstBatchInputReference batchX2,
		RealMatrix const& coefficientsX2,
		State const& state,
		BatchInputType& gradient
	) const{
		SIZE_CHECK(coefficientsX2.size1() == shark::size(batchX1));
		SIZE_CHECK(coefficientsX2.size2() == shark::size(batchX2));
		weightedInputDerivativeImpl<BatchInputType>(batchX1,batchX2,coefficientsX2,state,gradient);
	}

	void read(InArchive& ar){
		for(std::size_t i = 0;i != m_base.size(); ++i ){
			ar >> m_base[i].weight;
			ar >> m_base[i].adaptive;
			ar >> *(m_base[i].kernel);
		}
		ar >> m_weightsum;
		ar >> m_numParameters;
	}
	void write(OutArchive& ar) const{
		for(std::size_t i=0;i!= m_base.size();++i){
			ar << m_base[i].weight;
			ar << m_base[i].adaptive;
			ar << const_cast<AbstractKernelFunction<InputType> const&>(*(m_base[i].kernel));
		}
		ar << m_weightsum;
		ar << m_numParameters;
	}

protected:
	/// structure describing a single m_base kernel
	struct tBase
	{
		AbstractKernelFunction<InputType>* kernel;  ///< pointer to the m_base kernel object
		double weight;                              ///< weight in the linear combination
		bool adaptive;                              ///< whether the parameters of the kernel are part of the WeightedSumKernel's parameter vector?
	};

	void updateNumberOfParameters(){
		m_numParameters = m_base.size()-1;
		for (std::size_t i=0; i != m_base.size(); i++)
			if (m_base[i].adaptive)
				m_numParameters += m_base[i].kernel->numberOfParameters();
	}

	//we need to choose the correct implementation at compile time to ensure, that in the case, InputType
	//does not implement the needed operations, the implementation is replaced by a safe default which throws an exception
	//for this, we use enable_if/disable_if. The method is called with BatchInputType as template argument.
	//real implementation first.
	template <class T>
	void weightedInputDerivativeImpl(
		ConstBatchInputReference batchX1,
		ConstBatchInputReference batchX2,
		RealMatrix const& coefficientsX2,
		State const& state,
		BatchInputType& gradient,
		typename boost::enable_if<IsVector<T> >::type* dummy = 0
	)const{
		std::size_t numKernels = m_base.size(); //how far the loop goes;
		InternalState const& s = state.toState<InternalState>();


		//initialize gradient with the first kernel
		m_base[0].kernel->weightedInputDerivative(batchX1, batchX2, coefficientsX2, *s.kernelStates[0], gradient);
		gradient *= m_base[0].weight / m_weightsum;
		BatchInputType kernelGrad;
		for (std::size_t i=1; i != numKernels; i++){
			m_base[i].kernel->weightedInputDerivative(batchX1, batchX2, coefficientsX2, *s.kernelStates[i], kernelGrad);
			double coeff = m_base[i].weight / m_weightsum;
			gradient += coeff * kernelGrad;
		}
	}
	template <class T>
	void weightedInputDerivativeImpl(
		ConstBatchInputReference batchX1,
		ConstBatchInputReference batchX2,
		RealMatrix const& coefficientsX2,
		State const& state,
		BatchInputType& gradient,
		typename boost::disable_if<IsVector<T> >::type* dummy = 0
	)const{
		throw SHARKEXCEPTION("[WeightedSumKernel::weightdInputDerivative] The used BatchInputType is no Vector");
	}

	std::vector<tBase> m_base;                      ///< collection of m_base kernels
	double m_weightsum;                             ///< sum of all weights
	unsigned int m_numParameters;                   ///< total number of parameters
};

typedef WeightedSumKernel<> DenseWeightedSumKernel;
typedef WeightedSumKernel<CompressedRealVector> CompressedWeightedSumKernel;




//~ /// \brief Weighted sum of kernel functions
//~ ///
//~ /// For a set of positive definite kernels \f$ k_1, \dots, k_n \f$
//~ /// with positive coeffitients \f$ w_1, \dots, w_n \f$ the sum
//~ /// \f[ \tilde k(x_1, x_2) := \sum_{i=1}^{n} w_i \cdot k_i(x_1, x_2) \f]
//~ /// is again a positive definite kernel function.
//~ /// Internally, the weights are represented as
//~ /// \f$ w_i = \exp(\xi_i) \f$
//~ /// to allow for unconstrained optimization.
//~ ///
//~ /// This variant of the weighted sum kernel allows weights for all member
//~ /// kernels, whereas the simple #WeightedSumKernel fixes the first weight
//~ /// to 1.0, thus eliminating one redundant degree of freedom.
//~ ///
//~ /// The result of the kernel evaluation is devided by the sum of the
//~ /// kernel weights, so that in total, this amounts to fixing the sum
//~ /// of the of the weights to one.
//~ ///
//~ template<class InputType=RealVector>
//~ class FullyWeightedSumKernel : public AbstractKernelFunction<InputType>
//~ {
//~ public:
	//~ typedef typename VectorMatrixTraits<InputType>::SuperType InputSuperVectorType;
//~ private:
	//~ typedef AbstractKernelFunction<InputType> Base;
//~ public:
	//~ FullyWeightedSumKernel(const std::vector<AbstractKernelFunction<InputType>* >& base){
		//~ std::size_t kernels = base.size();
		//~ SHARK_CHECK( kernels > 0, "[FullyWeightedSumKernel::FullyWeightedSumKernel] There should be at least one sub-kernel.");
		//~ m_base.resize(kernels);
		//~ for (std::size_t i=0; i != kernels ; i++){
			//~ SHARK_ASSERT( base[i] != NULL );
			//~ m_base[i].kernel = base[i];
			//~ m_base[i].weight = 1.0;
			//~ m_base[i].adaptive = false;
		//~ }
		//~ m_weightsum = kernels;
		//~ bool tmp = true;
		//~ for ( unsigned int i=0; i<kernels; i++ ) if ( !m_base[i].kernel->hasFirstParameterDerivative() ) { tmp = false; break; }
		//~ if ( tmp ) this->m_features|=Base::HAS_FIRST_PARAMETER_DERIVATIVE;
		//~ tmp = true;
		//~ for ( unsigned int i=0; i<kernels; i++ ) if ( !m_base[i].kernel->hasFirstInputDerivative() ) { tmp = false; break; }
		//~ if ( tmp ) this->m_features|=Base::HAS_FIRST_INPUT_DERIVATIVE;
	//~ }
//~ /// Check whether m_base kernel index is adaptive.
//~ bool isAdaptive(std::size_t index) const
//~ { return m_base[index].adaptive; }
//~ /// Set adaptivity of m_base kernel index.
//~ void setAdaptive(std::size_t index, bool b = true)
//~ { m_base[index].adaptive = b; }
//~ /// Set adaptivity of all m_base kernels.
//~ void setAdaptiveAll(bool b = true){
//~     for (std::size_t i=0; i!= m_base.size(); i++)
//~         m_base[i].adaptive = b;
//~ }

//~ RealVector parameterVector() const{
//~     std::size_t num = numberOfParameters();
//~     std::size_t index = m_base.size();
//~     RealVector ret(num);
//~     for (std::size_t i=0; i!= m_base.size(); i++){
//~         ret(i) = std::log(m_base[i].weight);
//~         if (m_base[i].adaptive){
//~             std::size_t n = m_base[i].kernel->numberOfParameters();
//~             subrange(ret,index,index+n) = m_base[i].kernel->parameterVector();
//~             index += n;
//~         }
//~     }
//~     return ret;
//~ }

//~ void setParameterVector(RealVector const& newParameters){
//~     SIZE_CHECK(newParameters.size() == numberOfParameters());
//~     std::size_t index = m_base.size();
//~     m_weightsum = 0.0;
//~     for (std::size_t i=0; i != m_base.size(); i++){
//~         double w = newParameters(i);
//~         m_weightsum += std::exp(w);
//~         m_base[i].weight = std::exp(w);
//~         if (m_base[i].adaptive){
//~             std::size_t n = m_base[i].kernel->numberOfParameters();
//              m_base[i].kernel->setParameterVector(RealVector(subrange(newParameters,index,index+n)));
//~             index += n;
//~         }
//~     }
//~ }
	//~ std::size_t numberOfParameters() const{
		//~ std::size_t num = m_base.size();
		//~ for (std::size_t i=0; i != m_base.size(); i++)
			//~ if (m_base[i].adaptive)
				//~ num += m_base[i].kernel->numberOfParameters();
		//~ return num;
	//~ }

	//~ std::size_t numberOfIntermediateValues(InputType const& x1, InputType const& x2)const{
		//~ std::size_t num = m_base.size()+1;
		//~ for (std::size_t i=0; i != m_base.size(); i++){
			//~ num += m_base[i].kernel->numberOfIntermediateValues(x1,x2);
		//~ }
		//~ return num;
	//~ }

	//~ /// \f$ k(x, y) = \frac{\sum_i \exp(w_i) k_i(x, y)}{sum_i exp(w_i)} \f$
	//~ double eval(InputType const& x1, InputType const& x2) const{
		//~ SIZE_CHECK(x1.size() == x2.size());
		//~ double numerator = 0.0;
		//~ for (std::size_t i=0; i != m_base.size(); i++){
			//~ double result = m_base[i].kernel->eval(x1, x2);
			//~ numerator += m_base[i].weight*result;
		//~ }
		//~ return numerator / m_weightsum;
	//~ }
//~ /// \f$ k(x, y) = \frac{\sum_i \exp(w_i) k_i(x, y)}{sum_i exp(w_i)} \f$
//~ double eval(ConstInputReference x1, ConstInputReference x2, Intermediate& intermediate) const{
//~     SIZE_CHECK(x1.size() == x2.size());
//~     intermediate.resize(numberOfIntermediateValues(x1,x2));
//~     double numerator = 0.0;
//~     std::size_t currentIntermediate = 1;
//~     for (std::size_t i=0; i != m_base.size(); i++){
//~         std::size_t endIntermediate = currentIntermediate+1+m_base[i].kernel->numberOfIntermediateValues(x1,x2);
//~         Intermediate curIntermediate(intermediate,currentIntermediate+1,endIntermediate);
//~         double result = m_base[i].kernel->eval(x1, x2,curIntermediate);
//~         numerator += m_base[i].weight*result;
//~         intermediate[currentIntermediate]=result;
//~         currentIntermediate=endIntermediate;
//~     }
//~     intermediate[0]=numerator;
//~     return numerator / m_weightsum;
//~ }

//~ /// Weight parameter derivative:
//~ /// <br/>
//~ /// \f$ \frac{\partial k(x, y)}{\partial w_j} =
//~ ///     \frac{\exp(w_j) \left( k_j(x, y) \left[ \sum_i \exp(w_i) \right] - \left[ \sum_i \exp(w_i) k_i(x, y) \right] \right)}
//~ ///          {\left[ \sum_i \exp(w_i) \right]^2} \f$
//~ /// <br/>
//~ /// Adaptive kernel paramater derivative:
//~ /// <br/>
//~ /// \f$ \frac{\partial k(x, y)}{\partial p}
//~ ///     \frac{\exp(w_j) \frac{\partial k_j(x, y)}{\partial p}}
//~ ///          {\sum_i \exp(w_i)} \f$,
//~ /// where p is a parameter of the j-th kernel.
//~ void parameterDerivative(ConstInputReference x1, const InputReference x2, Intermediate const& intermediate, RealVector& gradient) const{
//~     SIZE_CHECK(x1.size() == x2.size());
//~     SIZE_CHECK(intermediate.size() == numberOfIntermediateValues(x1,x2));
//~     gradient.resize(numberOfParameters());
//~     std::size_t ic = m_base.size();
//~     std::size_t index = ic;
//~     RealVector kernelGrad;
//~     double sumSquared = sqr(m_weightsum);
//~     std::size_t currentIntermediate = 1;
//~     double numerator = intermediate[0];
//~     for (std::size_t i=0; i<ic; i++) {
//~         std::size_t endIntermediate = currentIntermediate+1+m_base[i].kernel->numberOfIntermediateValues(x1,x2);
//~         double result = intermediate[currentIntermediate];
//~         gradient(i) = m_base[i].weight * (result * m_weightsum - numerator) / sumSquared;
//~         if (m_base[i].adaptive){
//~             Intermediate curIntermediate(intermediate,currentIntermediate+1,endIntermediate);
//~             double coeff = m_base[i].weight / m_weightsum;
//~             size_t n = m_base[i].kernel->numberOfParameters();
//~             m_base[i].kernel->parameterDerivative(x1,x2,curIntermediate,kernelGrad);
//              noalias(subrange(gradient,index,index+n)) = coeff * kernelGrad;
//~             index += n;
//~         }
//~         currentIntermediate = endIntermediate;
//~     }
//~ }

//~ /// Input derivative:
//~ /// <br/>
//~ /// \f$ \frac{\partial k(x, y)}{\partial x}
//~ ///     \frac{\sum_i \exp(w_i) \frac{\partial k_i(x, y)}{\partial x}}
//~ ///          {\sum_i exp(w_i)} \f$
//~ void inputDerivative(ConstInputReference x1, const InputReference x2, Intermediate const& intermediate, InputSuperVectorType& gradient) const{
//~     SIZE_CHECK(x1.size() == x2.size());
//~     gradient.resize(x1.size());
//~     gradient.clear();
//~     std::size_t ic = m_base.size();
//~     InputSuperVectorType kernelGrad;
//~     std::size_t currentIntermediate = 1;
//~     for (std::size_t i=0; i<ic; i++){
//~         std::size_t endIntermediate = currentIntermediate+1+m_base[i].kernel->numberOfIntermediateValues(x1,x2);
//~         Intermediate curIntermediate(intermediate,currentIntermediate+1,endIntermediate);
//~         m_base[i].kernel->inputDerivative(x1, x2, curIntermediate, kernelGrad);
//~         noalias(gradient) += (m_base[i].weight / m_weightsum) * kernelGrad;
//~         currentIntermediate=endIntermediate;
//~     }
//~ }

//~ void read(InArchive& ar){
//~     ar >> m_weightsum;
//~     for(std::size_t i=0;i!= m_base.size();++i){
//~         ar >> m_base[i].weight;
//~         ar >> m_base[i].adaptive;
//~         ar >> *(m_base[i].kernel);
//~     }
//~ }
//~ void write(OutArchive& ar) const{
//~     ar << m_weightsum;
//~     for(std::size_t i=0;i!= m_base.size();++i){
//~         ar << m_base[i].weight;
//~         ar << m_base[i].adaptive;
//~         ar << *(m_base[i].kernel);
//~     }
//~ }

//protected:
//~ /// structure describing a single m_base kernel
//~ struct tBase
//~ {
//~     AbstractKernelFunction<InputType>* kernel;        ///< pointer to the m_base kernel object
//~     double weight;                                    ///< weight in the linear combination
//~     bool adaptive;                                    ///< whether the parameters of the kernel are part of the FullyWeightedSumKernel's parameter vector?
//~ };

//~ std::vector<tBase> m_base;                            ///< collection of m_base kernels
//~ double m_weightsum;                                   ///< sum of all weights
//};

//typedef FullyWeightedSumKernel<> DenseFullyWeightedSumKernel;
//typedef FullyWeightedSumKernel<CompressedRealVector> CompressedFullyWeightedSumKernel;


}
#endif
