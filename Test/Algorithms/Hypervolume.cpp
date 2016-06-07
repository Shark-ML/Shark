#define BOOST_TEST_MODULE ALGORITHMS_HYPERVOLUME
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculator.h>
#include <shark/Core/utility/functional.h>

#include <shark/Rng/GlobalRng.h>

#include <boost/assign.hpp>


using namespace shark;


double inclusionExclusionHelper(std::vector< RealVector > const& points, RealVector const& reference, RealVector point, std::size_t setsize = 0, std::size_t index = 0)
{
	if (index == points.size())
	{
		if (setsize > 0)
		{
			double hv = 1.0;
			for (std::size_t i=0; i<point.size(); i++) hv *= reference[i] - point[i];
			return (setsize & 1) ? hv : -hv;
		}
		else return 0.0;
	}
	else
	{
		double ret = inclusionExclusionHelper(points, reference, point, setsize, index + 1);
		for (std::size_t i=0; i<point.size(); i++) point[i] = std::max(point[i], points[index][i]);
		ret += inclusionExclusionHelper(points, reference, point, setsize + 1, index + 1);
		return ret;
	}
}

// brute-force exact hypervolume computation in O(2^n m) time
double inclusionExclusion(std::vector< RealVector > const& points, RealVector const& reference)
{
	RealVector point(reference.size(), -1e100);
	return inclusionExclusionHelper(points, reference, point, 0, 0);
}


struct Fixture {

	Fixture() {
		BOOST_TEST_MESSAGE( "Setting up test sets and reference points " );
		m_refPoint3D = boost::assign::list_of( 1.1 )( 1.1 )( 1.1 );
		m_testSet3D.push_back( boost::assign::list_of( 6.56039859404455e-2 ) (0.4474014917277) (0.891923776019316) );
		m_testSet3D.push_back( boost::assign::list_of( 3.74945443950542e-2)(3.1364039802686e-2)(0.998804513479922 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.271275894554688)(0.962356894778677)(1.66911984440026e-2 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.237023460537611)(0.468951509833942)(0.850825693417425 ) );
		m_testSet3D.push_back( boost::assign::list_of( 8.35813910785332e-2)(0.199763306732937)(0.97627289850149 ) );
		m_testSet3D.push_back( boost::assign::list_of( 1.99072649788403e-2)(0.433909411793732)(0.900736544810901 ) );
		m_testSet3D.push_back( boost::assign::list_of( 9.60698311356187e-2)(0.977187045721721)(0.18940978121319 ) );
		m_testSet3D.push_back( boost::assign::list_of( 2.68052822856208e-2)(2.30651870780559e-2)(0.999374541394087 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.356223209018184)(0.309633114503212)(0.881607826507812 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.127964409429531)(0.73123479272024)(0.670015513129912 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.695473366395562)(0.588939459338073)(0.411663831140169 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.930735605917613)(0.11813654121718)(0.346085234453039 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.774030940645471)(2.83363630460836e-2)(0.632513362272141 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.882561783965009)(4.80931050853475e-3)(0.470171849451808 ) );
		m_testSet3D.push_back( boost::assign::list_of( 4.92340623346446e-3)(0.493836329534438)(0.869540936185878 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.305054163869799)(0.219367569077876)(0.926725324323535 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.575227233936948)(0.395585597387712)(0.715978815661927 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.914091673974525)(0.168988399705031)(0.368618138912863 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.225088318852838)(0.796785147906617)(0.560775067911755 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.306941172015014)(0.203530333828304)(0.929710987422322 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.185344081015371)(0.590388202293731)(0.785550343533082 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.177181921358634)(0.67105558509432)(0.719924279669315 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.668494587335475)(0.22012845825454)(0.710393164782469 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.768639363955671)(0.256541291890516)(0.585986427942633 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.403457020846225)(0.744309886218013)(0.532189088208334 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.659545359568811)(0.641205442223306)(0.39224418355721 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.156141960251846)(8.36191498217669e-2)(0.984188765446851 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.246039496399399)(0.954377757574506)(0.16919711007753 ) );
		m_testSet3D.push_back( boost::assign::list_of( 3.02243260456876e-2)(0.43842801306405)(0.898257962656493 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.243139979715573)(0.104253945099703)(0.96437236853565 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.343877707314699)(0.539556201272222)(0.768522757034998 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.715293885551218)(0.689330705208567)(0.114794756629825 ) );
		m_testSet3D.push_back( boost::assign::list_of( 1.27610149409238e-2)(9.47996983636579e-2)(0.995414573777096 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.30565381275615)(0.792827267212719)(0.527257689476066 ) );
		m_testSet3D.push_back( boost::assign::list_of( 0.43864576057661)(3.10389339442242e-2)(0.8981238674636 ) );
		HV_TEST_SET_3D = 0.60496383631719475;

		m_refPoint2D = boost::assign::list_of( 11 )( 11 );
		//~ m_refPoint2D.push_back(11);
		//~ m_refPoint2D.push_back(11);
		m_testSet2D.push_back( boost::assign::list_of( 0.0000000000 )(1.0000000000 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.1863801385)(0.9824777066 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.2787464911)(0.9603647191 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.3549325314)(0.9348919179 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.4220279525)(0.9065828188 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.4828402120)(0.8757084730 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.5388359009)(0.8424107501 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.5908933491)(0.8067496824 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.6395815841)(0.7687232254 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.6852861036)(0.7282739568 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.7282728148)(0.6852873173 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.7687221637)(0.6395828601 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.8067485614)(0.5908948796 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.8424097574)(0.5388374529 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.8757076053)(0.4828417856 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.9065821569)(0.4220293744 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.9348914021)(0.3549338901 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.9603643728)(0.2787476843 ) );
		m_testSet2D.push_back( boost::assign::list_of( 0.9824775804)(0.1863808035 ) );
		m_testSet2D.push_back( boost::assign::list_of( 1.0000000000)(0.0000000000 ) );
		HV_TEST_SET_2D = 120.196858;
	}

	RealVector m_refPoint2D;
	std::vector< RealVector > m_testSet2D;

	RealVector m_refPoint3D;
	std::vector< RealVector > m_testSet3D;
	
	double HV_TEST_SET_2D;
	double HV_TEST_SET_3D;
};


//creates points on a front defined by points x in [0,1]^d
// 1 is a linear front, 2 a convex front, 1/2 a concave front
//reference point is 1^d
std::vector<RealVector> createRandomFront(std::size_t numPoints, std::size_t numObj, double p){
	std::vector<RealVector> points(numPoints);
	for (std::size_t i = 0; i != numPoints; ++i) {
		points[i].resize(numObj);
		for (std::size_t j = 0; j != numObj - 1; ++j) {
			points[i][j] = Rng::uni(0.0, 1.0);
		}
		if (numObj > 2 && Rng::coinToss())
		{
			// make sure that some objective values coincide
			std::size_t jj = Rng::discrete(0, numObj - 2);
			points[i][jj] = std::round(4.0 * points[i][jj]) / 4.0;
		}
		double sum = 0.0;
		for (std::size_t j = 0; j != numObj - 1; ++j) sum += points[i][j];
		points[i][numObj - 1] = pow(1.0 - sum / (numObj - 1.0), p);
	}
	return points;
}
template<class Algorithm>
void testRandomFrontNormP(Algorithm algorithm, std::size_t numTests, std::size_t numPoints, std::size_t numObj, double p){
	RealVector reference(numObj,1.0);
	for(std::size_t t = 0; t != numTests;++t){
		std::vector<RealVector> points = createRandomFront(numPoints,numObj,p);

		double hv1 = algorithm(points, reference);
		double hv2 = inclusionExclusion(points, reference);
		BOOST_CHECK_SMALL(hv1 - hv2, 1e-12);
	}
}

template<class Algorithm>
void testRandomFrontNormPApprox(std::size_t evals, double epsilon, Algorithm algorithm, std::size_t numTests, std::size_t numPoints, std::size_t numObj, double p){
	RealVector reference(numObj,1.0);
	for(std::size_t t = 0; t != numTests;++t){
		std::vector<RealVector> points = createRandomFront(numPoints,numObj,p);
		std::vector<double> results(evals);
		for(std::size_t e = 0; e != evals; e++)
			results[e] = algorithm(points, reference);
		double hv2 = inclusionExclusion(points, reference);
		double hv1 = *median_element(results);
		//check bounds
		BOOST_CHECK_LT(hv1, (1+epsilon) * hv2);
		BOOST_CHECK_GT(hv1, (1-epsilon) * hv2);
	}
}


// scalable test with many equal objective values
template<class Algorithm>
void testEqualFront(Algorithm algorithm, std::size_t numTests, std::size_t numPoints, std::size_t numObj){
	for(std::size_t t = 0; t != numTests;++t){
		{
			std::vector<RealVector> points;
			points.push_back(RealVector(numObj, 0.5));
			for (unsigned int i=0; i<numObj; i++)
			{
				RealVector point(numObj, 0.25);
				point[i] = 0.75;
				points.push_back(point);
			}
			RealVector reference(numObj, 1.0);

			double hv1 = algorithm(points, reference);
			double hv2 = inclusionExclusion(points, reference);
			BOOST_CHECK_SMALL(hv1 - hv2, 1e-12);
		}
		{
			std::vector<RealVector> points;
			points.push_back(RealVector(numObj, 0.5));
			for (unsigned int i=0; i<numObj; i++)
			{
				RealVector point(numObj, 0.75);
				point[i] = 0.25;
				points.push_back(point);
			}
			RealVector reference(numObj, 1.0);

			double hv1 = algorithm(points, reference);
			double hv2 = inclusionExclusion(points, reference);
			BOOST_CHECK_SMALL(hv1 - hv2, 1e-12);
		}
	}
}

BOOST_FIXTURE_TEST_SUITE (Algorithms_Hypervolume, Fixture)

BOOST_AUTO_TEST_CASE( Algorithms_ExactHypervolume2D ) {

	HypervolumeCalculator2D hc;
	const std::size_t numTests = 100;
	const std::size_t numPoints = 15;
	
	
	//test 1: computes the value of a reference front correctly
	BOOST_CHECK_CLOSE( hc( m_testSet2D, m_refPoint2D ), HV_TEST_SET_2D, 1E-5 );

	// test with random fronts of different shapes
	testRandomFrontNormP(hc, numTests, numPoints, 2, 1);
	testRandomFrontNormP(hc, numTests, numPoints, 2, 2);
	testRandomFrontNormP(hc, numTests, numPoints, 2, 0.5);
	testEqualFront(hc, numTests, numPoints, 2);
}

BOOST_AUTO_TEST_CASE( Algorithms_ExactHypervolume3D ) {

	HypervolumeCalculator3D hc;
	const std::size_t numTests = 100;
	const std::size_t numPoints = 15;
	
	
	//test 1: computes the value of a reference front correctly
	BOOST_CHECK_CLOSE( hc( m_testSet3D, m_refPoint3D ), HV_TEST_SET_3D, 1E-5 );

	// test with random fronts of different shapes
	testRandomFrontNormP(hc, numTests, numPoints, 3, 1);
	testRandomFrontNormP(hc, numTests, numPoints, 3, 2);
	testRandomFrontNormP(hc, numTests, numPoints, 3, 0.5);
	testEqualFront(hc, numTests, numPoints, 3);
}

BOOST_AUTO_TEST_CASE( Algorithms_ExactHypervolumeMD ) {

	HypervolumeCalculatorMD hc;
	const std::size_t numTests = 10;
	const std::size_t numPoints = 15;
	
	
	//test 1: computes the value of a reference front correctly
	BOOST_CHECK_CLOSE( hc( m_testSet3D, m_refPoint3D ), HV_TEST_SET_3D, 1E-5 );

	// test with random fronts of different shapes
	for(std::size_t numObj = 3; numObj < 5; ++numObj){
		testRandomFrontNormP(hc, numTests, numPoints, numObj, 1);
		testRandomFrontNormP(hc, numTests, numPoints, numObj, 2);
		testRandomFrontNormP(hc, numTests, numPoints, numObj, 0.5);
		testEqualFront(hc, numTests, numPoints, numObj);
	}
}

BOOST_AUTO_TEST_CASE( Algorithms_ExactHypervolumeMDApprox ) {

	HypervolumeApproximator hc;
	const std::size_t numTests = 10;
	const std::size_t numPoints = 15;
	const std::size_t evals = 10;
	const double epsilon = 0.05;//runtime is quadratic in this :(
	hc.epsilon() = epsilon;
	// we take the median of 10 runs, so 1-delta=0.7 is relatively safe,
	// especially as the bound appears to be rather loose
	hc.delta() = 0.3;
	
	//test 1: computes the value of a reference front correctly
	std::vector<double> results(evals);
	for(std::size_t e = 0; e != evals; e++)
		results[e] = hc( m_testSet3D, m_refPoint3D );
	double hv = *median_element(results);
	//check bounds
	BOOST_CHECK_LT( hv, (1+epsilon) * HV_TEST_SET_3D );
	BOOST_CHECK_GT( hv, (1-epsilon) * HV_TEST_SET_3D );
	
	// test with random fronts of different shapes
	for(std::size_t numObj = 3; numObj < 5; ++numObj){
		testRandomFrontNormPApprox(evals,epsilon, hc, numTests, numPoints, numObj, 1);
		testRandomFrontNormPApprox(evals,epsilon, hc, numTests, numPoints, numObj, 2);
		testRandomFrontNormPApprox(evals,epsilon, hc, numTests, numPoints, numObj, 0.5);
	}
}

BOOST_AUTO_TEST_CASE( Algorithms_HypervolumeCalculator) {
	HypervolumeCalculator hc;
	const std::size_t numTests = 10;
	const std::size_t numPoints = 15;
	const std::size_t evals = 10;
	const double epsilon = 0.05;//runtime is quadratic in this :(
	
	//first exact
	hc.useApproximation(false);
	
	//test 1: computes the value of a reference front correctly
	BOOST_CHECK_CLOSE( hc( m_testSet3D, m_refPoint3D ), HV_TEST_SET_3D, 1E-5 );
	BOOST_CHECK_CLOSE( hc( m_testSet2D, m_refPoint2D ), HV_TEST_SET_2D, 1E-5 );

	// test with random fronts of different shapes
	for(std::size_t numObj = 2; numObj < 5; ++numObj){
		testRandomFrontNormP(hc, numTests, numPoints, numObj, 1);
		testRandomFrontNormP(hc, numTests, numPoints, numObj, 2);
		testRandomFrontNormP(hc, numTests, numPoints, numObj, 0.5);
		testEqualFront(hc, numTests, numPoints, numObj);
	}
	
	//test 3: now use the approximator
	hc.useApproximation(true);
	hc.approximationEpsilon() = epsilon;
	// we take the median of 10 runs, so 1-delta=0.7 is relatively safe,
	// especially as the bound appears to be rather loose
	hc.approximationDelta() = 0.3;
	testRandomFrontNormPApprox(evals,epsilon, hc, numTests, numPoints, 8, 1);
	testRandomFrontNormPApprox(evals,epsilon, hc, numTests, numPoints, 8, 2);
	testRandomFrontNormPApprox(evals,epsilon, hc, numTests, numPoints, 8, 0.5);
}

BOOST_AUTO_TEST_SUITE_END()
