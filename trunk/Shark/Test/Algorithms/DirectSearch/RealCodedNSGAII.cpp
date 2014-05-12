#define BOOST_TEST_MODULE DirectSearch_RealCodedNSGAII
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/RealCodedNSGAII.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

using namespace shark;

struct PointExtractor{

	template<class T>
	RealVector const& operator()(T const& arg)const{
		return arg.value;
	}
};

double testObjectiveFunctionMOOHelper(
	MultiObjectiveFunction const& f, 
	std::size_t mu, 
	std::size_t iterations,
	RealVector const& reference
){
	RealCodedNSGAII realCodedNSGAII;
	realCodedNSGAII.mu() = mu;
	realCodedNSGAII.init(f);
	
	for(std::size_t i = 0; i != iterations; ++i){
		realCodedNSGAII.step(f);
		std::clog<<"\r"<<i<<" "<<std::flush;
	}
	BOOST_REQUIRE_EQUAL(realCodedNSGAII.solution().size(), mu);
	HypervolumeCalculator hyp;
	double volume = hyp(PointExtractor(),realCodedNSGAII.solution(),reference);
	std::cout<<"\r"<<f.name()<<": "<<volume<<std::endl;
	return volume;
//	BOOST_CHECK_SMALL(volume - targetVolume, 5.e-3);
}

void testObjectiveFunctionMOO(
	MultiObjectiveFunction const& f, 
	std::size_t mu, 
	double targetVolume, 
	std::size_t iterations,
	RealVector const& reference
){
	std::vector<double> result(10);
	for (std::size_t i=0; i<result.size(); i++) result[i] = testObjectiveFunctionMOOHelper(f, mu, iterations, reference);
	double best = *std::max_element(result.begin(), result.end());
	BOOST_CHECK_SMALL(best - targetVolume, 5.e-3);
}


BOOST_AUTO_TEST_CASE( RealCodedNSGAII_HYPERVOLUME_Functions ) {
	RealVector reference(2);
	reference(0) = 11;
	reference(1) = 11;
	DTLZ2 dtlz2(5);
	double dtlz2Volume = 120.178966;
	testObjectiveFunctionMOO(dtlz2,10,dtlz2Volume,10000,reference);
	DTLZ4 dtlz4(5);
	double dtlz4Volume = 120.178966;
	testObjectiveFunctionMOO(dtlz4,10,dtlz4Volume,10000,reference);
	//~ DTLZ7 dtlz7(5); //not sure whether correctly implemented
	//~ double dtlz7Volume = 115.964708;
	//~ testObjectiveFunctionMOO(dtlz7,10,dtlz7Volume,10000,reference);
	ZDT1 zdt1(5);
	double zdt1Volume = 120.613761;
	testObjectiveFunctionMOO(zdt1,10,zdt1Volume,10000,reference);
	ZDT2 zdt2(5); //fails somehow
	double zdt2Volume = 120.286820;
	testObjectiveFunctionMOO(zdt2,10,zdt2Volume,10000,reference);
	ZDT3 zdt3(5);
	double zdt3Volume = 128.748470;
	testObjectiveFunctionMOO(zdt3,10,zdt3Volume,10000,reference);
	ZDT6 zdt6(5);
	double zdt6Volume = 117.483246;
	testObjectiveFunctionMOO(zdt6,10,zdt6Volume,10000,reference);
}

