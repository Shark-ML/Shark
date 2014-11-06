#include <shark/Unsupervised/RBM/GradientApproximations/ContrastiveDivergence.h>
#include <shark/Unsupervised/RBM/Sampling/GibbsOperator.h>
#define BOOST_TEST_MODULE RBM_ContrastiveDivergence
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

//this is a mockup test, meaning we implement a failsafe RBM and use an template specialization of GibbsSampling to dive deep into the CDs Structure
//obviously, this is a hack!

namespace shark{
//mockup of the RBM
struct RBMMockup{
public:
	typedef IntVector VectorType;

	
	struct Energy{
		struct Structure{
			bool requireHiddenSample;
			bool parametersSet;
			
			Structure():requireHiddenSample(false), parametersSet(false){}
			template<class T>
			void configure(T const& t){}
			
			std::size_t numberOfParameters()const{
				return 1;
			}
			void setParameterVector(RealVector const&){
				parametersSet = true;
			}
			RealVector parameterVector(){
				return RealVector();
			}
		};
		
		struct AverageEnergyGradient{
			int id;
			Structure* m_structure;
			
			
			AverageEnergyGradient(Structure* structure):id(0),m_structure(structure){}
			
			template<class HiddenSampleBatch, class VisibleSampleBatch>
			void addVH(HiddenSampleBatch const& hidden,VisibleSampleBatch const& visible){
				BOOST_CHECK_EQUAL(visible.calledPrecompute,visible.calledSample);
				BOOST_CHECK_EQUAL(hidden.calledPrecompute,visible.calledPrecompute);
				if(m_structure->requireHiddenSample){
					BOOST_CHECK(hidden.calledSample == 0 ||hidden.calledSample == 5);
				}
				else{
					if(hidden.calledSample != 0)
						BOOST_CHECK_EQUAL( hidden.calledSample , 4);
				}
				BOOST_CHECK_EQUAL(hidden.id,visible.id);
				BOOST_CHECK_EQUAL(hidden.id,id);
				++id;
			}
			
			IntVector result()const{
				return IntVector(1);
			}
			
			
			
		};
		
		Energy(Structure*){}
	};
	Energy::Structure& structure(){
		static Energy::Structure structure;
		return structure;
	}
	
	std::size_t numberOfHN()const{
		return 0;
	}
	std::size_t numberOfVN()const{
		return 0;
	}
	
	std::size_t numberOfParameters()const{
		return 0;
	}
};

//specialization for Gibbs Sampling
//during sampling the order of the calls is checked
template<>
class GibbsOperator<RBMMockup>{
public:
	typedef RBMMockup RBM;
	typedef RBMMockup::Energy Energy;
	typedef IntVector VectorType;
	typedef Energy::Structure Structure;
	
	RBM* mpe_rbm;
	SamplingFlags m_flags;
	
	///\brief Represents the state of the hidden units
	struct HiddenSampleBatch{
		int id;
		int calledPrecompute;
		int calledSample;
		
		HiddenSampleBatch(std::size_t,std::size_t){}
	};
	
	template<class T>
	void configure(T const& t){}
	
	///\brief Represents the state of the visible units
	struct VisibleSampleBatch{
		int calledPrecompute;
		int calledSample;
		int id;
		
		VisibleSampleBatch(std::size_t,std::size_t){}
	};

	GibbsOperator(RBM* rbm):mpe_rbm(rbm){}

        void configure(PropertyTree const& ) {
        }
  
	RBM* rbm()const{
		return mpe_rbm;
	}
	
	SamplingFlags& flags(){
		return m_flags;
	}
	
	SamplingFlags const& flags()const{
		return m_flags;
	}

	//the methods themselves only check the ordering of the calls
	void precomputeHidden(HiddenSampleBatch& hidden,const VisibleSampleBatch& visible)const{
		BOOST_CHECK_EQUAL(hidden.calledPrecompute, hidden.calledSample);
		BOOST_CHECK_EQUAL(hidden.calledSample+1, visible.calledSample);
		BOOST_CHECK_EQUAL(hidden.calledPrecompute+1, visible.calledPrecompute);
		BOOST_CHECK_EQUAL(hidden.id, visible.id);
		++hidden.calledPrecompute;
	}
	void precomputeVisible(const HiddenSampleBatch& hidden, VisibleSampleBatch& visible)const{
		BOOST_CHECK_EQUAL(visible.calledPrecompute, visible.calledSample);
		BOOST_CHECK_EQUAL(visible.calledSample, hidden.calledSample);
		BOOST_CHECK_EQUAL(visible.calledPrecompute, hidden.calledPrecompute);
		BOOST_CHECK_EQUAL(hidden.id, visible.id);
		++visible.calledPrecompute;
	}
	
	void sampleHidden(HiddenSampleBatch& hidden)const{
		BOOST_CHECK_EQUAL(hidden.calledPrecompute, hidden.calledSample+1);
		++hidden.calledSample;
	}
	void sampleVisible(VisibleSampleBatch& visible)const{
		BOOST_CHECK_EQUAL(visible.calledPrecompute, visible.calledSample+1);
		++visible.calledSample;
	}
	
	template<class M>
	void createSample(HiddenSampleBatch& hidden,VisibleSampleBatch& visible, M const& state)const{
		visible.calledPrecompute = 0;
		visible.calledSample = 0;
		hidden.calledPrecompute = 0;
		hidden.calledSample = 0;
		hidden.id = (int) state(0,0);
		visible.id = (int) state(0,0);
	}
};
}

using namespace shark;

BOOST_AUTO_TEST_SUITE (RBM_ContrastiveDivergence)

BOOST_AUTO_TEST_CASE( ContrastiveDivergence_noHiddenSample )
{
	RBMMockup rbm;
	ContrastiveDivergence<GibbsOperator<RBMMockup> > cd(&rbm);
	
	std::vector<IntVector> vec(5);
	for(std::size_t i = 0; i != 5; ++i){
		vec[i].resize(1);
		vec[i](0)=i;
	}
	UnlabeledData<IntVector> data = createDataFromRange(vec,1);
	cd.setData(data);
	cd.setK(5);
	rbm.structure().requireHiddenSample = false;
	RealVector point;
	
	ContrastiveDivergence<GibbsOperator<RBMMockup> >::FirstOrderDerivative derivative;
	cd.evalDerivative(point,derivative);
	
	//test whether parameters were set
	BOOST_CHECK(rbm.structure().parametersSet);
	
}
BOOST_AUTO_TEST_CASE( ContrastiveDivergence_noHiddenSample_batch )
{
	RBMMockup rbm;
	ContrastiveDivergence<GibbsOperator<RBMMockup> > cd(&rbm);
	
	std::vector<IntVector> vec(10);
	for(std::size_t i = 0; i != 10; i+=2){
		vec[i].resize(1);
		vec[i+1].resize(1);
		vec[i](0)=i/2;
		vec[i+1](0)=i/2;
	}
	UnlabeledData<IntVector> data = createDataFromRange(vec,2);
	cd.setData(data);
	cd.setK(5);
	rbm.structure().requireHiddenSample = false;
	RealVector point;
	
	ContrastiveDivergence<GibbsOperator<RBMMockup> >::FirstOrderDerivative derivative;
	cd.evalDerivative(point,derivative);
	
	//test whether parameters were set
	BOOST_CHECK(rbm.structure().parametersSet);	
}
BOOST_AUTO_TEST_CASE( ContrastiveDivergence_WithHiddenSample )
{
	RBMMockup rbm;
	ContrastiveDivergence<GibbsOperator<RBMMockup> > cd(&rbm);
	
	std::vector<IntVector> vec(5);
	for(std::size_t i = 0; i != 5; ++i){
		vec[i].resize(1);
		vec[i](0)=i;
	}
	rbm.structure().requireHiddenSample = true;
	UnlabeledData<IntVector> data = createDataFromRange(vec,1);
	cd.setData(data);
	cd.setK(5);
	RealVector point;
	
	ContrastiveDivergence<GibbsOperator<RBMMockup> >::FirstOrderDerivative derivative;
	cd.evalDerivative(point,derivative);
	
	//test whether parameters were set
	BOOST_CHECK(rbm.structure().parametersSet);
	
}

BOOST_AUTO_TEST_SUITE_END()
