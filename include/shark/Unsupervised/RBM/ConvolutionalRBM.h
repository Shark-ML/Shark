/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_UNSUPERVISED_RBM_CONVOLUTIONALRBM_H
#define SHARK_UNSUPERVISED_RBM_CONVOLUTIONALRBM_H

#include <shark/Models/AbstractModel.h>
#include <shark/Unsupervised/RBM/Energy.h>
#include <shark/Unsupervised/RBM/Impl/ConvolutionalEnergyGradient.h>

#include <sstream>
#include <boost/serialization/string.hpp>
namespace shark{

///\brief Implements a convolutional RBM with a single greyscale input imge and a set of squared image filters
///
/// This class implements a simple RBM which interprets is input as images and instead of learning arbitrary filters, learns a convolution
/// Thus the ConvolutionalRBM is to an RBM what a ConvolutionalFFNet is to an FFNet.
template<class VisibleLayerT,class HiddenLayerT, class RngT>
class ConvolutionalRBM : public AbstractModel<RealVector, RealVector>{
private:
	typedef AbstractModel<RealVector, RealVector> base_type;
public:
	typedef detail::ConvolutionalEnergyGradient<ConvolutionalRBM> GradientType;
	typedef HiddenLayerT HiddenType; //< type of the hidden layer
	typedef VisibleLayerT VisibleType; //< type of the visible layer
	typedef RngT RngType;
	typedef Energy<ConvolutionalRBM<VisibleType,HiddenType,RngT> > EnergyType;//< Type of the energy function
	
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
	
private:
	std::size_t m_inputSize1;
	std::size_t m_inputSize2;

	/// \brief The weight matrix connecting hidden and visible layer.
	blas::matrix_set<RealMatrix> m_filters;

	///The layer of hidden Neurons
	HiddenType m_hiddenNeurons;

	///The Layer of visible Neurons
	VisibleType m_visibleNeurons;

	RngType* mpe_rng;
	bool m_forward;
	bool m_evalMean;

	///\brief Evaluates the input by propagating the visible input to the hidden neurons.
	///
	///@param patterns batch of states of visible units
	///@param outputs batch of (expected) states of hidden units
	void evalForward(BatchInputType const& state,BatchOutputType& output)const{
		std::size_t batchSize=state.size1();
		typename HiddenType::StatisticsBatch statisticsBatch(batchSize,numberOfHN());
		RealMatrix inputBatch(batchSize,numberOfHN());
		output.resize(state.size1(),numberOfHN());
		
		energy().inputHidden(inputBatch,state);
		hiddenNeurons().sufficientStatistics(inputBatch,statisticsBatch,blas::repeat(1.0,batchSize));

		if(m_evalMean){
			noalias(output) = hiddenNeurons().mean(statisticsBatch);
		}
		else{
			hiddenNeurons().sample(statisticsBatch,output,0.0,*mpe_rng);
		}
	}

	///\brief Evaluates the input by propagating the hidden input to the visible neurons.
	///
	///@param patterns batch of states of hidden units
	///@param outputs batch of (expected) states of visible units
	void evalBackward(BatchInputType const& state,BatchOutputType& output)const{
		std::size_t batchSize = state.size1();
		typename VisibleType::StatisticsBatch statisticsBatch(batchSize,numberOfVN());
		RealMatrix inputBatch(batchSize,numberOfVN());
		output.resize(batchSize,numberOfVN());
		
		energy().inputVisible(inputBatch,state);
		visibleNeurons().sufficientStatistics(inputBatch,statisticsBatch,blas::repeat(1.0,batchSize));
		
		if(m_evalMean){
			noalias(output) = visibleNeurons().mean(statisticsBatch);
		}
		else{
			visibleNeurons().sample(statisticsBatch,output,0.0,*mpe_rng);
		}
	}
public:
	ConvolutionalRBM(RngType& rng):mpe_rng(&rng),m_forward(true),m_evalMean(true)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ConvolutionalRBM"; }

	///\brief Returns the total number of parameters of the model.
	std::size_t numberOfParameters()const {
		std::size_t parameters = numFilters()*filterSize1()*filterSize2();
		parameters += m_hiddenNeurons.numberOfParameters();
		parameters += m_visibleNeurons.numberOfParameters();
		return parameters;
	}
	
	///\brief Returns the parameters of the Model as parameter vector.
	RealVector parameterVector () const {
		RealVector ret(numberOfParameters());
		init(ret) << matrixSet(m_filters),blas::parameters(m_hiddenNeurons),blas::parameters(m_visibleNeurons);
		return ret;
	};

	///\brief Sets the parameters of the model.
	///
	/// @param newParameters vector of parameters  
 	void setParameterVector(const RealVector& newParameters) {
		init(newParameters) >>matrixSet(m_filters),blas::parameters(m_hiddenNeurons),blas::parameters(m_visibleNeurons);
 	}
	
	///\brief Creates the structure of the ConvolutionalRBM.
	///
	///@param newInputSize1 width of input image
	///@param newInputSize2 height of input image
	///@param newNumFilters number of filters to train
	///@param filterSize size of the sides of the filter
	void setStructure(
		std::size_t newInputSize1, std::size_t newInputSize2,
		std::size_t newNumFilters,
		std::size_t filterSize
	){
		//check that we have at least one row/column of hidden units
		SIZE_CHECK(newInputSize1 > filterSize);
		SIZE_CHECK(newInputSize2 > filterSize);
		
		std::size_t numVisible = newInputSize1*newInputSize2;
		std::size_t numHidden = (newInputSize1-filterSize+1)*(newInputSize2-filterSize+1)*newNumFilters;
		
		m_filters= blas::matrix_set<RealMatrix>(newNumFilters,filterSize,filterSize);
		m_filters.clear();
		
		m_hiddenNeurons.resize(numHidden);
		m_visibleNeurons.resize(numVisible);

		m_inputSize1 = newInputSize1;
		m_inputSize2 = newInputSize2;
	}
	
	///\brief Returns the layer of hidden neurons.
	HiddenType const& hiddenNeurons()const{
		return m_hiddenNeurons;
	}
	///\brief Returns the layer of hidden neurons.
	HiddenType& hiddenNeurons(){
		return m_hiddenNeurons;
	}
	///\brief Returns the layer of visible neurons.
	VisibleType& visibleNeurons(){
		return m_visibleNeurons;
	}
	///\brief Returns the layer of visible neurons.
	VisibleType const& visibleNeurons()const{
		return m_visibleNeurons;
	}
	
	std::size_t numFilters()const{
		return m_filters.size();
	}
	std::size_t filterSize1()const{
		return m_filters.size1();
	}
	std::size_t filterSize2()const{
		return m_filters.size2();
	}
	
	std::size_t inputSize1()const{
		return m_inputSize1;
	}
	
	std::size_t inputSize2()const{
		return m_inputSize2;
	}
	
	
	std::size_t responseSize1()const{
		return m_inputSize1-m_filters.size1()+1;
	}
	std::size_t responseSize2()const{
		return m_inputSize2-m_filters.size2()+1;
	}
	
	///\brief Returns the weight matrix connecting the layers.
	blas::matrix_set<RealMatrix>& filters(){
		return m_filters;
	}
	///\brief Returns the weight matrix connecting the layers.
	blas::matrix_set<RealMatrix> const& weightMatrix()const{
		return m_filters;
	}
	
	///\brief Returns the energy function of the ConvolutionalRBM.
	EnergyType energy()const{
		return EnergyType(*this);
	}
	
	///\brief Returns the random number generator associated with this ConvolutionalRBM.
	RngType& rng(){
		return *mpe_rng;
	}
	
	///\brief Sets the type of evaluation, eval will perform.
	///
	///Eval performs its operation based on the state of this function.
	///There are two ways to pass data through an ConvolutionalRBM: either forward, setting the states of the
	///visible neurons and sample the hidden states or backwards, where the state of the hidden is fixed and the visible
	///are sampled. 
	///Instead of the state of the hidden/visible, one often wants the mean of the state \f$ E_{p(h|v)}\left(h\right)\f$. 
	///By default, the ConvolutionalRBM uses the forward evaluation and returns the mean of the state
	///
	///@param forward whether the forward view should be used false=backwards
	///@param evalMean whether the mean state should be returned. false=a sample is returned
	void evaluationType(bool forward,bool evalMean){
		m_forward = forward;
		m_evalMean = evalMean;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}
	
	///\brief Passes information through/samples from an ConvolutionalRBM in a forward or backward way. 
	///
	///Eval performs its operation based on the given evaluation type.
	///There are two ways to pass data through an ConvolutionalRBM: either forward, setting the states of the
	///visible neurons and sample the hidden states or backwards, where the state of the hidden is fixed and the visible
	///are sampled. 
	///Instead of the state of the hidden/visible, one often wants the mean of the state \f$ E_{p(h|v)}\left(h\right)\f$. 
	///By default, the ConvolutionalRBM uses the forward evaluation and returns the mean of the state,
	///but other evaluation modes can be set by evaluationType().
	///
	///@param patterns the batch of (visible or hidden) inputs
	///@param outputs the batch of (visible or hidden) outputs 
	void eval(BatchInputType const& patterns,BatchOutputType& outputs)const{
		if(m_forward){
			evalForward(patterns,outputs);
		}
		else{
			evalBackward(patterns,outputs);
		}
	}


	void eval(BatchInputType const& patterns, BatchOutputType& outputs, State& state)const{
		eval(patterns,outputs);
	}
	
	///\brief Calculates the input of the hidden neurons given the state of the visible in a batch-vise fassion.
	///
	///@param inputs the batch of vectors the input of the hidden neurons is stored in
	///@param visibleStates the batch of states of the visible neurons
	void inputHidden(RealMatrix& inputs, RealMatrix const& visibleStates)const{
		SIZE_CHECK(visibleStates.size1() == inputs.size1());
		SIZE_CHECK(inputs.size2() == numberOfHN());
		SIZE_CHECK( visibleStates.size2() == numberOfVN());
		inputs.clear();

		for(std::size_t i= 0; i != inputs.size1();++i){
			blas::dense_matrix_adaptor<double const> visibleState = 
				to_matrix(row(visibleStates,i),inputSize1(),inputSize2());
			blas::dense_matrix_adaptor<double> responses = 
				to_matrix(row(inputs,i),m_filters.size()*responseSize1(),responseSize2());
			
			for (std::size_t x1=0; x1 != responseSize1(); ++x1) {
				for (std::size_t x2=0; x2 != responseSize2(); ++x2) {
					std::size_t end1= x1+m_filters.size1();
					std::size_t end2= x2+m_filters.size2();
					for(std::size_t f = 0; f != m_filters.size();++f){
						responses(f*responseSize1()+x1,x2)=sum(m_filters[f]*subrange(visibleState,x1,end1,x2,end2));
					}
				}
			}
		}
	}


	///\brief Calculates the input of the visible neurons given the state of the hidden.
	///
	///@param inputs the vector the input of the visible neurons is stored in
	///@param hiddenStates the state of the hidden neurons
	void inputVisible(RealMatrix& inputs, RealMatrix const& hiddenStates)const{
		SIZE_CHECK(hiddenStates.size1() == inputs.size1());
		SIZE_CHECK(inputs.size2() == numberOfVN());
		SIZE_CHECK(hiddenStates.size2() == numberOfHN());
		typedef blas::dense_matrix_adaptor<double> Response;
		inputs.clear();
		
		//we slightly optimize this routine by checking whether the hiddens are 0 -
		//this is likely when the hiddens are binary.
		for(std::size_t i= 0; i != inputs.size1();++i){
			blas::dense_matrix_adaptor<double const> hiddenState = 
				to_matrix(row(hiddenStates,i),responseSize1()*m_filters.size(),responseSize2());
			Response responses = 
				to_matrix(row(inputs,i),m_inputSize1,m_inputSize2);
			
			for (std::size_t x1=0; x1 != responseSize1(); ++x1) {
				for (std::size_t x2=0; x2 != responseSize2(); ++x2) {
					std::size_t end1= x1+m_filters.size1();
					std::size_t end2= x2+m_filters.size2();
					blas::matrix_range<Response> receptiveArea = subrange(responses,x1,end1,x2,end2);
						
					for(std::size_t f = 0; f != m_filters.size();++f){
						double neuronResponse = hiddenState(f*responseSize1()+x1,x2);
						if(neuronResponse == 0.0) continue;
						noalias(receptiveArea) += neuronResponse * m_filters[f];
					}
				}
			}
		}
	}
	
	using base_type::eval;
	
	
	///\brief Returns the number of hidden Neurons, that is the number of filter responses
	std::size_t numberOfHN()const{
		return m_hiddenNeurons.size();
	}
	///\brief Returns the number of visible Neurons, which is the number of pixels of the image
	std::size_t numberOfVN()const{
		return m_visibleNeurons.size();
	}
	
	/// \brief Reads the network from an archive.
	void read(InArchive& archive){
		archive >> m_filters;
		archive >> m_hiddenNeurons;
		archive >> m_visibleNeurons;
		
		//serialization of the rng is a bit...complex
		//let's hope that we can remove this hack one time. But we really can't ignore the state of the rng.
		std::string str;
		archive>> str;
		std::stringstream stream(str);
		stream>> *mpe_rng;
	}

	/// \brief Writes the network to an archive.
	void write(OutArchive& archive) const{
		archive << m_filters;
		archive << m_hiddenNeurons;
		archive << m_visibleNeurons;
		
		std::stringstream stream;
		stream <<*mpe_rng;
		std::string str = stream.str();
		archive <<str;
	}

};

}

#endif
