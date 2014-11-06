#include <shark/Unsupervised/RBM/Sampling/GibbsOperator.h>
#include <shark/Unsupervised/RBM/Neuronlayers/BinaryLayer.h>
#include <shark/Unsupervised/RBM/Energy.h>
#include <shark/Unsupervised/RBM/RBM.h>
#include <shark/Rng/GlobalRng.h>

#define BOOST_TEST_MODULE RBM_GibbsOperator
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;


//this is a mockup test, meaning we implement a failsafe Energy& Neuron to test the Gibbs Operator without noise!

//mockup for the neuron
struct NeuronMockup{
	
	NeuronMockup(int statMove,int sampleMul):statMove(statMove),sampleMul(sampleMul){}
	
	void sufficientStatistics(RealVector input,RealVector& statistics, double beta){
		statistics(0) = input(0)*beta + statMove;
	}
	
	template<class Rng>
	void sample(RealVector stat,RealVector& state,Rng& rng){
		state(0) = stat(0)*sampleMul;
	}
	
	double statMove;
	double sampleMul;
};
//mockup for the Energy encapsulated in an empty RBM Mockup
struct RBMMockup{
public:
	typedef int VectorType;

	
	struct Energy{
		struct Structure{
			NeuronMockup hiddenNeurons(){
				return NeuronMockup(1,2);
			}
			NeuronMockup visibleNeurons(){
				return NeuronMockup(5,7);
			}
		};
		typedef double HiddenInput;
		typedef double HiddenStatistics;
		typedef double HiddenFeatures;
		typedef double HiddenState;
		
		typedef double VisibleInput;
		typedef double VisibleStatistics;
		typedef double VisibleFeatures;
		typedef double VisibleState;

		typedef double VectorType;
		
		Energy(Structure*){}
		
		//since we want to check that both versions are used, we will choose different return values for both
		void inputHidden(RealVector& input, RealVector state, RealVector& features){
			features(0) = state(0)*state(0);
			input(0) = features(0)+9;
		}
		void inputHidden(RealVector& input, RealVector state){
			input(0) = state(0)*state(0); //omit the +9
		}
		
		void inputVisible(RealVector& input, RealVector state, RealVector& features){
			features(0) = state(0)*state(0)*state(0);
			input(0) = features(0)/2.0;
		}
		void inputVisible(RealVector& input, RealVector state){
			input(0) = state(0)*state(0) *state(0); //omit the /2.0
		}
		
		RealVector energyFromVisibleInput(
			RealVector const& input, 
			RealVector const& hiddenState, 
			RealVector const& visibleState
		){
			RealVector ret(1);
			
			ret(0)=hiddenState(0)*visibleState(0) + input(0);
			return ret;
		}
		
	};
	Energy::Structure& structure(){
		static Energy::Structure structure;
		return structure;
	}
	
	NeuronMockup hiddenNeurons(){
		return structure().hiddenNeurons();
	}
	NeuronMockup visibleNeurons(){
		return structure().visibleNeurons();
	}
	
	int& rng(){
		static int i;
		return i;
	}
};


BOOST_AUTO_TEST_SUITE (RBM_GibbsOperator)

BOOST_AUTO_TEST_CASE( GibbsOperator_Test_Store )
{
	RBMMockup rbmMockup;
	GibbsOperator<RBMMockup> chain(&rbmMockup);
	GibbsOperator<RBMMockup>::HiddenSampleBatch hidden(1,0);
	GibbsOperator<RBMMockup>::HiddenSampleBatch hiddenTest(1,0);
	GibbsOperator<RBMMockup>::VisibleSampleBatch visible(1,0);
	GibbsOperator<RBMMockup>::VisibleSampleBatch visibleTest(1,0);
	chain.flags() |= StoreHiddenFeatures;
	chain.flags() |= StoreVisibleFeatures;
	
	hidden.state(0) = 3;
	hidden.features(0) = 4;
	hidden.input(0) = 5;
	hidden.statistics(0) = 6;
	
	visible.state(0) = 5;
	visible.features(0) = 6;
	visible.input(0) = 7;
	visible.statistics(0) = 8;
	
	
	//test hidden statistics
	visibleTest=visible;
	chain.precomputeHidden(hiddenTest,visibleTest,0.5);
	BOOST_CHECK_SMALL(visibleTest.features(0)-25.0,1.e-20);
	BOOST_CHECK_SMALL(hiddenTest.input(0)-34.0,1.e-20);
	BOOST_CHECK_SMALL(hiddenTest.statistics(0)-18.0,1.e-20);
	
	//test hidden sampling
	chain.sampleHidden(hiddenTest);
	//old values should not change
	BOOST_CHECK_SMALL(hiddenTest.input(0)-34.0,1.e-20);
	BOOST_CHECK_SMALL(hiddenTest.statistics(0)-18.0,1.e-20);
	BOOST_CHECK_SMALL(hiddenTest.state(0)-36,1.e-20);
	
	//test visible statistics
	hiddenTest=hidden;
	chain.precomputeVisible(hiddenTest,visibleTest,0.5);
	BOOST_CHECK_SMALL(hiddenTest.features(0)-27.0,1.e-20);
	BOOST_CHECK_SMALL(visibleTest.input(0)-13.5,1.e-20);
	BOOST_CHECK_SMALL(visibleTest.statistics(0)-11.75,1.e-20);
	
	//test visible sampling
	chain.sampleVisible(visibleTest);
	BOOST_CHECK_SMALL(visibleTest.input(0)-13.5,1.e-20);
	BOOST_CHECK_SMALL(visibleTest.statistics(0)-11.75,1.e-20);
	BOOST_CHECK_SMALL(visibleTest.state(0)-11.75*7,1.e-20);
}

BOOST_AUTO_TEST_CASE( GibbsOperator_Test )
{
	RBMMockup rbmMockup;
	GibbsOperator<RBMMockup> chain(&rbmMockup);
	GibbsOperator<RBMMockup>::HiddenSampleBatch hidden(1,0);
	GibbsOperator<RBMMockup>::HiddenSampleBatch hiddenTest(1,0);
	GibbsOperator<RBMMockup>::VisibleSampleBatch visible(1,0);
	GibbsOperator<RBMMockup>::VisibleSampleBatch visibleTest(1,0);
	
	hidden.state(0) = 3;
	hidden.features(0) = 4;
	hidden.input(0) = 5;
	hidden.statistics(0) = 6;
	
	visible.state(0) = 5;
	visible.features(0) = 6;
	visible.input(0) = 7;
	visible.statistics(0) = 8;
	
	
	//test hidden statistics
	chain.precomputeHidden(hiddenTest,visible,0.5);
	BOOST_CHECK_SMALL(hiddenTest.input(0)-25.0,1.e-20);
	BOOST_CHECK_SMALL(hiddenTest.statistics(0)-13.5,1.e-20);
	
	//test hidden sampling
	chain.sampleHidden(hiddenTest);
	//old values should not change
	BOOST_CHECK_SMALL(hiddenTest.input(0)-25.0,1.e-20);
	BOOST_CHECK_SMALL(hiddenTest.statistics(0)-13.5,1.e-20);
	BOOST_CHECK_SMALL(hiddenTest.state(0)-27,1.e-20);
	
	//test visible statistics
	chain.precomputeVisible(hidden,visibleTest,0.5);
	BOOST_CHECK_SMALL(visibleTest.input(0)-27,1.e-20);
	BOOST_CHECK_SMALL(visibleTest.statistics(0)-18.5,1.e-20);
	
	//test visible sampling
	chain.sampleVisible(visibleTest);
	BOOST_CHECK_SMALL(visibleTest.input(0)-27,1.e-20);
	BOOST_CHECK_SMALL(visibleTest.statistics(0)-18.5,1.e-20);
	BOOST_CHECK_SMALL(visibleTest.state(0)-18.5*7,1.e-20);
}

BOOST_AUTO_TEST_CASE( GibbsOperator_Energy)
{
	RBMMockup rbmMockup;
	GibbsOperator<RBMMockup> chain(&rbmMockup);
	GibbsOperator<RBMMockup>::HiddenSampleBatch hidden(1,0);
	GibbsOperator<RBMMockup>::VisibleSampleBatch visible(1,0);
	
	
	//test swap of hidden
	hidden.state(0) = 6;
	hidden.features(0) = 2;
	hidden.input(0) = 3;
	hidden.statistics(0) = 4;
	
	visible.state(0) = 5;
	visible.features(0) = 6;
	visible.input(0) = 7;
	visible.statistics(0) = 8;
	
	RealVector energy = chain.calculateEnergy(hidden,visible);
	BOOST_CHECK_SMALL(energy(0)-37.0, 1.e-20);
}

BOOST_AUTO_TEST_CASE( GibbsOperator_Sample_Swap)
{
	RBMMockup rbmMockup;
	GibbsOperator<RBMMockup> chain(&rbmMockup);
	GibbsOperator<RBMMockup>::HiddenSampleBatch hidden1(1,0);
	GibbsOperator<RBMMockup>::HiddenSampleBatch hidden2(1,0);
	GibbsOperator<RBMMockup>::VisibleSampleBatch visible1(1,0);
	GibbsOperator<RBMMockup>::VisibleSampleBatch visible2(1,0);
	
	
	//test swap of hidden
	hidden1.state(0) = 1;
	hidden1.features(0) = 2;
	hidden1.input(0) = 3;
	hidden1.statistics(0) = 4;
	
	hidden2.state(0) = 5;
	hidden2.features(0) = 6;
	hidden2.input(0) = 7;
	hidden2.statistics(0) = 8;
	
	swap(hidden1,hidden2);
	BOOST_CHECK_SMALL(hidden1.state(0)-5.0,1.e-20);
	BOOST_CHECK_SMALL(hidden1.features(0)-6.0,1.e-20);
	BOOST_CHECK_SMALL(hidden1.input(0)-7.0,1.e-20);
	BOOST_CHECK_SMALL(hidden1.statistics(0)-8.0,1.e-20);
	
	BOOST_CHECK_SMALL(hidden2.state(0)-1.0,1.e-20);
	BOOST_CHECK_SMALL(hidden2.features(0)-2.0,1.e-20);
	BOOST_CHECK_SMALL(hidden2.input(0)-3.0,1.e-20);
	BOOST_CHECK_SMALL(hidden2.statistics(0)-4.0,1.e-20);
	
	//test swap of visible
	visible1.state(0) = 1;
	visible1.features(0) = 2;
	visible1.input(0) = 3;
	visible1.statistics(0) = 4;
	
	visible2.state(0) = 5;
	visible2.features(0) = 6;
	visible2.input(0) = 7;
	visible2.statistics(0) = 8;
	
	swap(hidden1,hidden2);
	BOOST_CHECK_SMALL(visible2.state(0)-5.0,1.e-20);
	BOOST_CHECK_SMALL(visible2.features(0)-6.0,1.e-20);
	BOOST_CHECK_SMALL(visible2.input(0)-7.0,1.e-20);
	BOOST_CHECK_SMALL(visible2.statistics(0)-8.0,1.e-20);
	
	BOOST_CHECK_SMALL(visible1.state(0)-1.0,1.e-20);
	BOOST_CHECK_SMALL(visible1.features(0)-2.0,1.e-20);
	BOOST_CHECK_SMALL(visible1.input(0)-3.0,1.e-20);
	BOOST_CHECK_SMALL(visible1.statistics(0)-4.0,1.e-20);
}

////BOOST_AUTO_TEST_CASE_EXPECTED_FAILURES(GibbsOperator_Create_Sample, 1);
BOOST_AUTO_TEST_CASE( GibbsOperator_Create_Sample_Store)
{
	RBMMockup rbmMockup;
	GibbsOperator<RBMMockup> chain(&rbmMockup);
	chain.flags() |= StoreHiddenFeatures;
	chain.flags() |= StoreVisibleStatistics;
	chain.flags() |= StoreVisibleFeatures;
	chain.flags() |= StoreVisibleInput;
	
	GibbsOperator<RBMMockup>::HiddenSampleBatch hidden(1,0);
	GibbsOperator<RBMMockup>::VisibleSampleBatch visible(1,0);
	
	RealVector state(1);
	state(0) = 2;
	
	chain.createSample( hidden, visible, state, 0.5);
	
	//check the states
	BOOST_CHECK_SMALL(hidden.state(0)-15.0,1.e-20);
	BOOST_CHECK_SMALL(hidden.features(0)-3375,1.e-20);
	BOOST_CHECK_SMALL(hidden.input(0)-13.0,1.e-20);
	BOOST_CHECK_SMALL(hidden.statistics(0)-7.5,1.e-20);
	
	BOOST_CHECK_SMALL(visible.state(0) - 2,1.e-20);
	BOOST_CHECK_SMALL(visible.features(0) - 4.0,1.e-20);
	BOOST_CHECK_SMALL(visible.input(0) - 1687.5,1.e-20);
	BOOST_CHECK_SMALL(visible.statistics(0) - 848.75,1.e-20);
}
BOOST_AUTO_TEST_CASE( GibbsOperator_Create_Sample)
{
	RBMMockup rbmMockup;
	GibbsOperator<RBMMockup> chain(&rbmMockup);
	
	GibbsOperator<RBMMockup>::HiddenSampleBatch hidden(1,0);
	GibbsOperator<RBMMockup>::VisibleSampleBatch visible(1,0);
	
	RealVector state(1);
	state(0) = 2;
	
	chain.createSample( hidden, visible, state, 0.5);
	
	//check the states
	BOOST_CHECK_SMALL(visible.state(0)-state(0),1.e-20);
	
	BOOST_CHECK_SMALL(hidden.state(0)-6.0,1.e-20);
	BOOST_CHECK_SMALL(hidden.input(0)-4.0,1.e-20);
	BOOST_CHECK_SMALL(hidden.statistics(0)-3,1.e-20);
	
}

//BOOST_AUTO_TEST_CASE( GibbsOperator_Energy_Binary)
//{
//	typedef RBM<Energy<BinaryLayer,BinaryLayer>,Rng::rng_type > RBMType;
//	RBMType rbm(Rng::globalRng);
//	GibbsOperator<RBMMockup>::HiddenSampleBatch hidden(1,5);
//	GibbsOperator<RBMMockup>::VisibleSampleBatch visible(1,5);
//	GibbsOperator<RBMType> gibbsOperator(&rbm);
//	
//	
//	rbm.structure().setStructure(5,5);
//	rbm.structure().weightMatrix(0,0).clear();
//	
//	for(std::size_t i = 0; i != 5; ++i){
//		rbm.structure().weightMatrix(0,0)(i,i) = i;
//		rbm.hiddenNeurons().bias()(i) = i;
//		rbm.visibleNeurons().bias()(i) = 5+i;
//		visible.state(i) = 10+i;
//		hidden.state(i) = 15+i;
//	}
//	
//	visible.input = prod(hidden.state,rbm.structure().weightMatrix(0,0));
//	
//	double energyResult = -inner_prod(rbm.hiddenNeurons().bias(),hidden.state);
//	energyResult-= inner_prod(rbm.visibleNeurons().bias(),visible.state);
//	energyResult-= inner_prod(hidden.state,prod(rbm.structure().weightMatrix(0,0),visible.state));
//	
//	double testEnergy = gibbsOperator.calculateEnergy(hidden,visible);
//	BOOST_CHECK_SMALL(testEnergy-energyResult, 1.e-15);
//	
//}

BOOST_AUTO_TEST_SUITE_END()
