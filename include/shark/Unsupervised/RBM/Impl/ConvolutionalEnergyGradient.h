#ifndef SHARK_UNSUPERVISED_RBM_IMPL_CONVOLUTIONALENERGYGRADIENT_H
#define SHARK_UNSUPERVISED_RBM_IMPL_CONVOLUTIONALENERGYGRADIENT_H

#include <shark/LinAlg/Base.h>
#include <shark/LinAlg/BLAS/matrix_set.hpp>
namespace shark{
namespace detail{
///\brief The gradient of the energy averaged over a set of cumulative added samples.
///
/// It is needed by log-likelihood gradient approximators because it delivers the information
/// how the derivatives of certain energy functions look like.  
///
///This is the special case for Neurons with one interaction term only.
template<class RBM>
class ConvolutionalEnergyGradient{
public:	
	ConvolutionalEnergyGradient(RBM const* rbm)
	: mpe_rbm(rbm)
	, m_deltaWeights(rbm->numFilters(),rbm->filterSize1(), rbm->filterSize2())
	, m_logWeightSum(-std::numeric_limits<double>::infinity()){
		SHARK_CHECK(mpe_rbm != 0, "rbm is not allowed to be 0");
		std::size_t const hiddenParameters = rbm->hiddenNeurons().numberOfParameters();
		std::size_t const visibleParameters = rbm->visibleNeurons().numberOfParameters();
		m_deltaBiasHidden.resize(hiddenParameters);
		m_deltaBiasVisible.resize(visibleParameters);
		m_deltaWeights.clear();
	}
	
	///\brief Calculates the weighted expectation of the energy gradient with respect to p(h|v) for a complete Batch.
	///
	///for numerical stability, the logarithm of the weights is used
	///
	///This method weights the members of the batch using the weights specified by the corresponding parameter
	///@param hiddens a batch of samples for the hidden layer
	///@param visibles a batch of samples of the visible layer
	///@param logWeights the logarithm of the weights for every sample
	template<class HiddenSampleBatch, class VisibleSampleBatch, class WeightVector>
	void addVH(HiddenSampleBatch const& hiddens, VisibleSampleBatch const& visibles, WeightVector const& logWeights){
		SIZE_CHECK(logWeights.size() == shark::size(hiddens));
		SIZE_CHECK(logWeights.size() == shark::size(visibles));
		
		///update the internal state and get the transformed weights for the batch
		RealVector weights = updateWeights(logWeights);
		if(weights.empty()) return;//weights are not relevant to the gradient
		
		std::size_t batchSize = shark::size(hiddens);
		
		//update the gradient
		RealMatrix weightedFeatures = mpe_rbm->visibleNeurons().phi(visibles.state);
		for(std::size_t i = 0; i != batchSize; ++i){
			row(weightedFeatures,i)*= weights(i);
		}
		updateConnectionDerivative(mpe_rbm->hiddenNeurons().expectedPhiValue(hiddens.statistics),weightedFeatures);
		mpe_rbm->visibleNeurons().parameterDerivative(m_deltaBiasVisible,visibles,weights);
		mpe_rbm->hiddenNeurons().expectedParameterDerivative(m_deltaBiasHidden,hiddens,weights);
	}

	///\brief Calculates the weighted expectation of the energy gradient with respect to p(v|h) for a complete Batch.
	///
	///for numerical stability, the logarithm of the weights is used
	///
	///This method weights the members of the batch using the weights in the specified by the corresponding parameter
	///@param hiddens a batch of samples for the hidden layer
	///@param visibles a batch of samples of the visible layer
	///@param logWeights the logarithm of the weights for every sample
	template<class HiddenSampleBatch, class VisibleSampleBatch, class WeightVector>
	void addHV(HiddenSampleBatch const& hiddens, VisibleSampleBatch const& visibles, WeightVector const& logWeights){
		SIZE_CHECK(logWeights.size() == shark::size(hiddens));
		SIZE_CHECK(logWeights.size() == shark::size(visibles));
		
		///update the internal state and get the transformed weights for the batch
		RealVector weights = updateWeights(logWeights);
		if(weights.empty()) return;
		
		std::size_t batchSize = shark::size(hiddens);
		
		//update the gradient
		RealMatrix weightedFeatures = mpe_rbm->hiddenNeurons().phi(hiddens.state);
		for(std::size_t i = 0; i != batchSize; ++i){
			row(weightedFeatures,i)*= weights(i);
		}
		updateConnectionDerivative(weightedFeatures,mpe_rbm->visibleNeurons().expectedPhiValue(visibles.statistics));
		mpe_rbm->hiddenNeurons().parameterDerivative(m_deltaBiasHidden,hiddens,weights);
		mpe_rbm->visibleNeurons().expectedParameterDerivative(m_deltaBiasVisible,visibles,weights);
	}
	
	ConvolutionalEnergyGradient& operator+=(ConvolutionalEnergyGradient const& gradient){
		double const minExp = minExpInput<double>();
		double const maxExp = maxExpInput<double>();
		
		double weightSumDiff = gradient.m_logWeightSum-m_logWeightSum;
		//check whether the weight is big enough to have an effect
		if(weightSumDiff <= minExp )
			return *this;
		
		//if the old weights are to small, there is no use in keeping them
		if(weightSumDiff >= maxExp ){
			(*this) = gradient;
			return *this;
		}

		double logWeightSumUpdate = softPlus(weightSumDiff);
		m_logWeightSum += logWeightSumUpdate;

		//scaling factor corrects by multiplying with 
		//Z/(Z+Z_new)=1/(1+exp(logZ_new - logZ))
		double const scalingFactor = std::exp(-logWeightSumUpdate);// factor is <=1
		m_deltaWeights *= scalingFactor;
		m_deltaBiasVisible *= scalingFactor;
		m_deltaBiasHidden *= scalingFactor;
		
		//now add the new gradient with its corrected weight
		double weight = std::exp(gradient.m_logWeightSum-m_logWeightSum);
		noalias(m_deltaWeights) += weight * gradient.m_deltaWeights;
		noalias(m_deltaBiasVisible) += weight * gradient.m_deltaBiasVisible;
		noalias(m_deltaBiasHidden) += weight * gradient.m_deltaBiasHidden;
	}
	
	///\brief Calculates the expectation of the energy gradient with respect to p(h|v) for a complete Batch.
	///
	///for numerical stability, the logarithm of the weights is used
	///
	///This method weights the memberas of the batch using the weights specified by the corresponding parameter
	///@param hiddens a batch of samples for the hidden layer
	///@param visibles a batch of samples of the visible layer
	template<class HiddenSampleBatch, class VisibleSampleBatch>
	void addVH(HiddenSampleBatch const& hiddens, VisibleSampleBatch const& visibles){
		addVH(hiddens,visibles, blas::repeat(0.0,shark::size(hiddens)));
	}

	///\brief Calculates the weighted expectation of the energy gradient with respect to p(v|h) for a complete Batch.
	///
	///for numerical stability, the logarithm of the weights is used
	///
	///This method weights the members of the batch using the weights specified by the corresponding parameter
	///@param hiddens a batch of samples for the hidden layer
	///@param visibles a batch of samples of the visible layer
	template<class HiddenSampleBatch, class VisibleSampleBatch>
	void addHV(HiddenSampleBatch const& hiddens, VisibleSampleBatch const& visibles){
		addHV(hiddens,visibles, blas::repeat(0.0,shark::size(hiddens)));
	}
	
	///Returns the log of the sum of the weights.
	///
	///@return the logarithm of the sum of weights
	double logWeightSum(){
		return m_logWeightSum;
	}

	///\brief Writes the derivatives of all parameters into a vector and returns it.
	RealVector result()const{
		RealVector derivative(mpe_rbm->numberOfParameters());
		init(derivative) << matrixSet(m_deltaWeights),m_deltaBiasHidden,m_deltaBiasVisible;
		return derivative;
	}
	
	///\brief Resets the internal state. 
	void clear(){
		m_deltaWeights.clear();
		m_deltaBiasVisible.clear();
		m_deltaBiasHidden.clear();
		m_logWeightSum = -std::numeric_limits<double>::infinity();
	}
	
private:
	RBM const* mpe_rbm; //structure of the corresponding RBM
	blas::matrix_set<RealMatrix> m_deltaWeights; //stores the average of the derivatives with respect to the weights
	RealVector m_deltaBiasHidden; //stores the average of the derivative with respect to the hidden biases
	RealVector m_deltaBiasVisible; //stores the average of the derivative with respect to the visible biases
	double m_logWeightSum; //log of sum of weights. Usually equal to the log of the number of samples used.
	

	template<class MatrixH, class MatrixV>
	void updateConnectionDerivative(MatrixH const& hiddens, MatrixV const& visibles){
		
		std::size_t numFilters = mpe_rbm->numFilters();
		std::size_t responseSize1 = mpe_rbm->responseSize1();
		std::size_t responseSize2 = mpe_rbm->responseSize2();
		for(std::size_t i = 0; i != hiddens.size1();++i){
			blas::dense_matrix_adaptor<double const> visible = 
				to_matrix(row(visibles,i),mpe_rbm->inputSize1(),mpe_rbm->inputSize2());
			blas::dense_matrix_adaptor<double const> hidden = 
				to_matrix(row(hiddens,i),numFilters*responseSize1,responseSize2);
			
			for (std::size_t x1=0; x1 != responseSize1; ++x1) {
				for(std::size_t x2=0; x2 != responseSize2; ++x2) {
					std::size_t end1 = x1+mpe_rbm->filterSize1();
					std::size_t end2 = x2+mpe_rbm->filterSize2();
					for(std::size_t f = 0; f != numFilters;++f){
						double val = hidden(f*responseSize1+x1,x2);
						noalias(m_deltaWeights[f]) += subrange(visible,x1,end1,x2,end2)*val;
					}
				}
			}
		}
	}
	
	template<class WeightVector>
	RealVector updateWeights(WeightVector const& logWeights){
		double const minExp = minExpInput<double>();
		double const maxExp = maxExpInput<double>();
		
		//calculate the gradient update with respect of only the current batch
		std::size_t batchSize = shark::size(logWeights);
		//first calculate the batchLogWeightSum
		double batchLogWeightSum = logWeights(0);
		for(std::size_t i = 1; i != batchSize; ++i){
			double const diff = logWeights(i) - batchLogWeightSum;
			if(diff >= maxExp || diff <= minExp){
				if(logWeights(i) > batchLogWeightSum)
					batchLogWeightSum = logWeights(i);
			}
			else
				batchLogWeightSum += softPlus(diff);
		}
		
		double weightSumDiff = batchLogWeightSum-m_logWeightSum;
		//check whether any new weight is big enough to have an effect
		if(weightSumDiff <= minExp )
			return RealVector();
		
		//if the old weights are to small, there is no use in keeping them
		if(weightSumDiff >= maxExp ){
			clear();
			m_logWeightSum = batchLogWeightSum;
		}
		else
		{
			double weightSumUpdate = softPlus(weightSumDiff);
			m_logWeightSum = m_logWeightSum + weightSumUpdate;

			//scaling factor corrects by multiplying with 
			//Z/(Z+Z_new)=1/(1+exp(logZ_new - logZ))
			double const scalingFactor = std::exp(-weightSumUpdate);// factor is <=1
			m_deltaWeights *= scalingFactor;
			m_deltaBiasVisible *= scalingFactor;
			m_deltaBiasHidden *= scalingFactor;
		}
			
		//now calculate the weights for the elements of the new batch
		RealVector weights(batchSize);
		for(std::size_t i = 0; i != batchSize; ++i){
			weights(i) = std::exp(logWeights(i)-m_logWeightSum);
		}
		return weights;
	}
};
}}

#endif