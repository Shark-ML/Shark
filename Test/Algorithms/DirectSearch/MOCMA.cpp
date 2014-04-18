#define BOOST_TEST_MODULE DirectSearch_MOCMA
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/MOCMA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

using namespace shark;

struct PointExtractor{

	template<class T>
	RealVector const& operator()(T const& arg)const{
		return arg.value;
	}
};

void testObjectiveFunctionMOO(
	MultiObjectiveFunction const& f, 
	std::size_t mu, 
	double targetVolume, 
	std::size_t iterations,
	RealVector const& reference
){
	MOCMA mocma;
	mocma.mu() = mu;
	mocma.init(f);
	
	for(std::size_t i = 0; i != iterations; ++i){
		mocma.step(f);
		std::clog<<"\r"<<i<<" "<<std::flush;
	}
	BOOST_REQUIRE_EQUAL(mocma.solution().size(), mu);
	HypervolumeCalculator hyp;
	double volume = hyp(PointExtractor(),mocma.solution(),reference);
	std::cout<<"\r"<<f.name()<<": "<<volume<<std::endl;
	BOOST_CHECK_SMALL(volume - targetVolume, 5.e-3);
}


BOOST_AUTO_TEST_CASE( MOCMA_HYPERVOLUME_Functions ) {
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
	ZDT2 zdt2(5);
	double zdt2Volume = 120.286820;
	testObjectiveFunctionMOO(zdt2,10,zdt2Volume,10000,reference);
	ZDT3 zdt3(5);
	double zdt3Volume = 128.748470;
	testObjectiveFunctionMOO(zdt3,10,zdt3Volume,10000,reference);
	ZDT6 zdt6(5);
	double zdt6Volume = 117.483246;
	testObjectiveFunctionMOO(zdt6,10,zdt6Volume,10000,reference);
}


BOOST_AUTO_TEST_CASE( MOCMA_SERIALIZATION ) {
	MOCMA mocma;

	DTLZ1 dtlz1;
	dtlz1.setNumberOfObjectives( 3 );
	dtlz1.setNumberOfVariables( 10 );
	BOOST_CHECK_NO_THROW( mocma.init( dtlz1 ) );
	BOOST_CHECK_NO_THROW( mocma.step( dtlz1 ) );
	
	{
		std::stringstream ss;
		boost::archive::text_oarchive oa( ss );

		BOOST_CHECK_NO_THROW( (oa << mocma) );

		MOCMA mocma2;

		boost::archive::text_iarchive ia( ss );
		BOOST_CHECK_NO_THROW( (ia >> mocma2) );

		Rng::seed( 1 );
		FastRng::seed( 1 );
		mocma.step( dtlz1 );
		MOCMA::SolutionType set1 = mocma.solution();
		Rng::seed( 1 );
		FastRng::seed( 1 );
		mocma2.step( dtlz1 );
		MOCMA::SolutionType set2 = mocma2.solution();

		for( unsigned int i = 0; i < set1.size(); i++ ) {
			BOOST_CHECK_SMALL( norm_2( set1.at( i ).value - set2.at( i ).value ), 1E-20 );
			BOOST_CHECK_SMALL( norm_2( set1.at( i ).point- set2.at( i ).point), 1E-20 );
		}

	}
}
