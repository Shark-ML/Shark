#include <shark/Rng/Weibull.h>
#include <shark/Rng/Bernoulli.h>
#include <shark/Rng/Binomial.h>
#include <shark/Rng/Cauchy.h>
#include <shark/Rng/DiffGeometric.h>
#include <shark/Rng/Dirichlet.h>
#include <shark/Rng/DiscreteUniform.h>
#include <shark/Rng/Erlang.h>
#include <shark/Rng/Gamma.h>
#include <shark/Rng/Geometric.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Rng/HyperGeometric.h>
#include <shark/Rng/LogNormal.h>
#include <shark/Rng/NegExponential.h>
#include <shark/Rng/Normal.h>
#include <shark/Rng/Poisson.h>
#include <shark/Rng/Uniform.h>

#include <shark/Statistics/Statistics.h>

#define BOOST_TEST_MODULE Rng_Distributions
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/bernoulli.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/lognormal.hpp>
#include <boost/math/distributions/poisson.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/math/distributions/exponential.hpp>
#include <boost/math/distributions/hypergeometric.hpp>

namespace shark {
	template<typename Distribution>
	void check_distribution( Distribution & dist, double mean, double variance, unsigned int noTrials = 100000 ) {
		shark::Statistics stats;

		for( unsigned int i = 0; i < noTrials; i++ )
			BOOST_CHECK_NO_THROW( stats( dist() ) );

		BOOST_CHECK_CLOSE( stats( shark::Statistics::Mean() ), mean, 1. );
		BOOST_CHECK_CLOSE( stats( shark::Statistics::Variance() ), variance, 1. );
	}
}

BOOST_AUTO_TEST_SUITE (Rng_Rng)

BOOST_AUTO_TEST_CASE( Distribution_DefaultTemplateArgumentCheck ) {
	shark::Weibull<> dist1( shark::Rng::globalRng );
	shark::Bernoulli<> dist2( shark::Rng::globalRng );
	shark::Binomial<> dist3( shark::Rng::globalRng );
	shark::Cauchy<> dist4( shark::Rng::globalRng );
	shark::DiffGeometric<> dist5( shark::Rng::globalRng );
	shark::Dirichlet<> dist6( shark::Rng::globalRng );
	shark::DiscreteUniform<> dist7( shark::Rng::globalRng );
	shark::Erlang<> dist8( shark::Rng::globalRng );
	shark::Gamma<> dist9( shark::Rng::globalRng );
	shark::Geometric<> dist10( shark::Rng::globalRng );
	shark::HyperGeometric<> dist12( shark::Rng::globalRng );
	shark::LogNormal<> dist13( shark::Rng::globalRng );
	shark::NegExponential<> dist14( shark::Rng::globalRng );
	shark::Normal<> dist15( shark::Rng::globalRng );
	shark::Poisson<> dist16( shark::Rng::globalRng );
	shark::Uniform<> dist17( shark::Rng::globalRng );
}

BOOST_AUTO_TEST_CASE( Distribution_Bernoulli )
{
	shark::Rng::seed(42);
	shark::Bernoulli< shark::Rng::rng_type > bernoulli( shark::Rng::globalRng, 0.3 );
	BOOST_CHECK_CLOSE( bernoulli.prob(), 0.3, 1E-10 );
	BOOST_CHECK_CLOSE( bernoulli.p( true ), 0.3, 1E-10 );
	BOOST_CHECK_CLOSE( bernoulli.p( false ), 0.7, 1E-10 );

	shark::check_distribution( bernoulli, 0.3, 0.3 * (1. - 0.3) );

	boost::math::bernoulli_distribution<double> bd( 0.3 );
}

BOOST_AUTO_TEST_CASE( Distribution_Binomial )
{
	shark::Rng::seed(42);
	double probs[] = {
		0,
		0,
		0,
		0,
		0.00001,
		0.00004,
		0.00022,
		0.00102,
		0.00386,
		0.01201,
		0.03082,
		0.06537,
		0.11440,
		0.16426,
		0.19164,
		0.17886,
		0.13042,
		0.07160,
		0.02785,
		0.00684,
		0.00080
	};

	unsigned int n = 20;
	double p = 0.7;
	shark::Binomial< shark::Rng::rng_type > binomial( shark::Rng::globalRng, n, p );
	BOOST_CHECK( binomial.n() == n );
	BOOST_CHECK_CLOSE( binomial.prob(), p, 1E-1 );

	shark::check_distribution( binomial, n * p, n*p*(1-p) );

	for( unsigned int i = 0; i < n; i++ )
		BOOST_CHECK_SMALL( binomial.p( i ) - probs[ i ], 1E-5 );

}

BOOST_AUTO_TEST_CASE( Distribution_Cauchy )
{
	shark::Rng::seed(42);
	struct Pair {
		double m_first;
		double m_second;
	} pairs[] = {
		{-5 	,0.01224268793},
		{-4.9 	,0.01272730453},
		{-4.8 	,0.01324084385},
		{-4.7 	,0.01378561655},
		{-4.6 	,0.01436416454},
		{-4.5 	,0.01497928876},
		{-4.4 	,0.01563408085},
		{-4.3 	,0.01633195927},
		{-4.2 	,0.01707671063},
		{-4.1 	,0.01787253712},
		{-4 	,0.01872411095},
		{-3.9 	,0.01963663703},
		{-3.8 	,0.02061592527},
		{-3.7 	,0.02166847421},
		{-3.6 	,0.02280156778},
		{-3.5 	,0.02402338764},
		{-3.4 	,0.0253431438},
		{-3.3 	,0.02677122676},
		{-3.2 	,0.02831938489},
		{-3.1 	,0.03000093178},
		{-3 	,0.03183098862},
		{-2.9 	,0.03382676793},
		{-2.8 	,0.03600790568},
		{-2.7 	,0.03839684996},
		{-2.6 	,0.04101931523},
		{-2.5 	,0.04390481189},
		{-2.4 	,0.04708726127},
		{-2.3 	,0.05060570528},
		{-2.2 	,0.0545051175},
		{-2.1 	,0.05883731722},
		{-2 	,0.06366197724},
		{-1.9 	,0.06904769765},
		{-1.8 	,0.07507308636},
		{-1.7 	,0.08182773424},
		{-1.6 	,0.08941288938},
		{-1.5 	,0.09794150344},
		{-1.4 	,0.1075371237},
		{-1.3 	,0.1183308127},
		{-1.2 	,0.1304548714},
		{-1.1 	,0.1440316227},
		{-1 	,0.1591549431},
		{-0.9 	,0.1758618156},
		{-0.8 	,0.194091394},
		{-0.7 	,0.2136307961},
		{-0.6 	,0.2340513869},
		{-0.5 	,0.2546479089},
		{-0.4 	,0.2744050743},
		{-0.3 	,0.2920274185},
		{-0.2 	,0.3060671983},
		{-0.1 	,0.3151583032},
		{0 	,0.3183098862},
		{0.1 	,0.3151583032},
		{0.2 	,0.3060671983},
		{0.3 	,0.2920274185},
		{0.4 	,0.2744050743},
		{0.5 	,0.2546479089},
		{0.6 	,0.2340513869},
		{0.7 	,0.2136307961},
		{0.8 	,0.194091394},
		{0.9 	,0.1758618156},
		{1 	,0.1591549431},
		{1.1 	,0.1440316227},
		{1.2 	,0.1304548714},
		{1.3 	,0.1183308127},
		{1.4 	,0.1075371237},
		{1.5 	,0.09794150344},
		{1.6 	,0.08941288938},
		{1.7 	,0.08182773424},
		{1.8 	,0.07507308636},
		{1.9 	,0.06904769765},
		{2 	,0.06366197724},
		{2.1 	,0.05883731722},
		{2.2 	,0.0545051175},
		{2.3 	,0.05060570528},
		{2.4 	,0.04708726127},
		{2.5 	,0.04390481189},
		{2.6 	,0.04101931523},
		{2.7 	,0.03839684996},
		{2.8 	,0.03600790568},
		{2.9 	,0.03382676793},
		{3 	,0.03183098862},
		{3.1 	,0.03000093178},
		{3.2 	,0.02831938489},
		{3.3 	,0.02677122676},
		{3.4 	,0.0253431438},
		{3.5 	,0.02402338764},
		{3.6 	,0.02280156778},
		{3.7 	,0.02166847421},
		{3.8 	,0.02061592527},
		{3.9 	,0.01963663703},
		{4 	,0.01872411095},
		{4.1 	,0.01787253712},
		{4.2 	,0.01707671063},
		{4.3 	,0.01633195927},
		{4.4 	,0.01563408085},
		{4.5 	,0.01497928876},
		{4.6 	,0.01436416454},
		{4.7 	,0.01378561655},
		{4.8 	,0.01324084385},
		{4.9 	,0.01272730453},
		{5 	,0.01224268793},
	};

	double median = 0;
	double sigma = 1;
	shark::Cauchy< shark::Rng::rng_type > cauchy( shark::Rng::globalRng, median, sigma );
	BOOST_CHECK_CLOSE( cauchy.median(), median, 1E-10 );
	BOOST_CHECK_CLOSE( cauchy.sigma(), sigma, 1E-10 );

	cauchy.median( 5. );
	BOOST_CHECK_CLOSE( cauchy.median(), 5., 1E-10 );

	cauchy.sigma( 2. );
	BOOST_CHECK_CLOSE( cauchy.sigma(), 2, 1E-10 );

	cauchy.median( 0. );
	BOOST_CHECK_CLOSE( cauchy.median(), 0., 1E-10 );

	cauchy.sigma( 1. );
	BOOST_CHECK_CLOSE( cauchy.sigma(), 1., 1E-10 );

	for( unsigned int i = 0; i < 100; i++ ) {
		BOOST_CHECK_SMALL( cauchy.p( pairs[ i ].m_first ) - pairs[ i ].m_second, 1E-5 );
	}


}

BOOST_AUTO_TEST_CASE( Distribution_DiscreteUniform ) {
	shark::Rng::seed(42);
	long low = 0, high = 100;
	shark::DiscreteUniform< shark::Rng::rng_type > disc( shark::Rng::globalRng, low, high );
	BOOST_CHECK( disc.low() == low );
	BOOST_CHECK( disc.high() == high );

	disc.setRange( 10, 30 );
	BOOST_CHECK( disc.low() == 10 );
	BOOST_CHECK( disc.high() == 30 );

	disc.setRange( low, high );
	BOOST_CHECK( disc.low() == low );
	BOOST_CHECK( disc.high() == high );

	shark::check_distribution( disc, (high+low)/2., (boost::math::pow<2>(high-low+1)-1)/12 );
}

BOOST_AUTO_TEST_CASE( Distribution_Erlang ) {
	shark::Rng::seed(42);
	double mean = 1., variance = 1.;
	shark::Erlang< shark::Rng::rng_type > erlang( shark::Rng::globalRng, mean, variance );
	BOOST_CHECK_CLOSE( erlang.mean() , mean, 1E-10 );
	BOOST_CHECK_CLOSE( erlang.variance() , variance, 1E-10 );

	erlang.mean( 10 );
	BOOST_CHECK_CLOSE( erlang.mean() , 10, 1E-10 );
	erlang.variance( 30 );
	BOOST_CHECK_CLOSE( erlang.variance() , 30, 1E-10 );

	erlang.mean( mean );
	erlang.variance( variance );
	BOOST_CHECK_CLOSE( erlang.mean() , mean, 1E-10 );
	BOOST_CHECK_CLOSE( erlang.variance() , variance, 1E-10 );

	shark::check_distribution( erlang, mean, variance, 100000 );
}

BOOST_AUTO_TEST_CASE( Distribution_Geometric ) {
	shark::Rng::seed(42);
	double p = 0.3;
	shark::Geometric< shark::Rng::rng_type > geom( shark::Rng::globalRng, p );
	BOOST_CHECK_SMALL( geom.prob() - p, 1E-10 );

	geom.prob( 0.7 );
	BOOST_CHECK_SMALL( geom.prob() - 0.7, 1E-10 );

	geom.prob( p );
	BOOST_CHECK_SMALL( geom.prob() - p, 1E-10 );

	shark::check_distribution( geom, 1./p, (1.-p)/boost::math::pow<2>( p ) );
}

BOOST_AUTO_TEST_CASE( Distribution_HyperGeometric ) {
	//TODO
}

BOOST_AUTO_TEST_CASE( Distribution_LogNormal ) {
	shark::Rng::seed(42);
	shark::LogNormal< shark::Rng::rng_type > logNormal( shark::Rng::globalRng );

	double location = logNormal.location();
	double scale = logNormal.scale();

	logNormal.location( 2. );
	BOOST_CHECK_SMALL( logNormal.location() - 2., 1E-10 );
	logNormal.scale( 5 );
	BOOST_CHECK_SMALL( logNormal.scale() - 5., 1E-10 );

	logNormal.location( location );
	BOOST_CHECK_SMALL( logNormal.location() - location, 1E-10 );
	logNormal.scale( scale );
	BOOST_CHECK_SMALL( logNormal.scale() - scale, 1E-10 );

	boost::math::lognormal_distribution<double> lnd( location, scale );

	shark::check_distribution(
		logNormal,
		boost::math::mean( lnd ),
		boost::math::variance( lnd ),
		5000000
	);

	for( double x = 0.1; x < 10; x += 0.1 )
		BOOST_CHECK_SMALL( logNormal.p( x ) - boost::math::pdf( lnd, x ), 1E-5 );
}

BOOST_AUTO_TEST_CASE( Distribution_NegExponential ) {
	shark::Rng::seed(42);
	shark::NegExponential< shark::Rng::rng_type > negExponential(shark::Rng::globalRng,2.0);
	BOOST_CHECK_SMALL( negExponential.mean() - 2., 1E-10 );

	negExponential.mean( 5. );
	BOOST_CHECK_SMALL( negExponential.mean() - 5., 1E-10 );

	boost::math::exponential_distribution<double> lnd( 0.2 );

	shark::check_distribution(
		negExponential,
		boost::math::mean( lnd ),
		boost::math::variance( lnd )
	);

	for( double x = 0.1; x < 10; x += 0.1 )
		BOOST_CHECK_SMALL( negExponential.p( x ) - boost::math::pdf( lnd, x ), 1E-5 );
}

BOOST_AUTO_TEST_CASE( Distribution_Normal ) {
	shark::Rng::seed(42);
	shark::Normal< shark::Rng::rng_type > normal( shark::Rng::globalRng, 1., 1. );

	double mean = normal.mean();
	double variance = normal.variance();

	BOOST_CHECK_SMALL( mean - 1., 1E-10 );
	BOOST_CHECK_SMALL( variance - 1., 1E-10 );

	normal.mean( 2. );
	BOOST_CHECK_SMALL( normal.mean() - 2., 1E-10 );
	normal.variance( 5. );
	BOOST_CHECK_SMALL( normal.variance() - 5., 1E-10 );

	normal.mean( mean );
	BOOST_CHECK_SMALL( normal.mean() - mean, 1E-10 );
	normal.variance( variance );
	BOOST_CHECK_SMALL( normal.variance() - variance, 1E-10 );

	shark::check_distribution( normal, mean, variance );

	boost::math::normal_distribution<double> nd( 1., 1. );

	for( double x = -5; x < 5; x += 0.1 )
		BOOST_CHECK_SMALL( normal.p( x ) - boost::math::pdf( nd, x ), 1E-10 );

	// Test that normal distribution implementation in Shark works correctly with different mean and variance
	{
		using namespace shark;

		// Calculate posterior probability of height given male
		const double mean = 5.855;
		const double variance = 3.5033e-02;
		const double standardDeviation = sqrt(variance);
		const double theX = 6.0;
		const double expected = 1.5789;

		// Construct two Normal-s
		boost::math::normal_distribution<double> boostNormal( mean, standardDeviation );
		Normal<> sharkNormal(Rng::globalRng, mean, variance);

		// Tolerances will be used in the validation
		const double tolerancePercentage = 0.1;
		const double toleranceAbsolute = 1E-10;

		// Test shark Normal gets expected result
		BOOST_CHECK_CLOSE(sharkNormal.p(theX), expected, tolerancePercentage);

		// Test shark and boost implementations should get same numbers in range [-10, 10]
		for (double x = -10.; x < 10.; x += 0.1)
			BOOST_CHECK_SMALL( sharkNormal.p( x ) - boost::math::pdf( boostNormal, x ), toleranceAbsolute );
	}
}

BOOST_AUTO_TEST_CASE( Distribution_Poisson ) {
	shark::Rng::seed(42);
	shark::Poisson< shark::Rng::rng_type > poisson( shark::Rng::globalRng, 1. );

	BOOST_CHECK_SMALL( poisson.mean() - 1., 1E-10 );
	poisson.mean( 5 );
	BOOST_CHECK_SMALL( poisson.mean() - 5., 1E-10 );

	shark::check_distribution( poisson, 5., 5. );

	boost::math::poisson_distribution<double> pd( 5. );

	for( unsigned int k = 0; k < 100; k++ )
		BOOST_CHECK_SMALL( poisson.p( k ) - boost::math::pdf( pd, k ), 1E-5 );
}

BOOST_AUTO_TEST_CASE( Distribution_Uniform ) {
	shark::Rng::seed(42);
	shark::Uniform< shark::Rng::rng_type > uniform( shark::Rng::globalRng, 1, 5 );

	BOOST_CHECK_SMALL( uniform.low() - 1, 1E-10 );
	BOOST_CHECK_SMALL( uniform.high() - 5, 1E-10 );

	uniform.setRange( 3, 7 );

	BOOST_CHECK_SMALL( uniform.low() - 3, 1E-10 );
	BOOST_CHECK_SMALL( uniform.high() - 7, 1E-10 );

	uniform.setRange( 9, 2 );
	BOOST_CHECK_SMALL( uniform.low() - 2, 1E-10 );
	BOOST_CHECK_SMALL( uniform.high() - 9, 1E-10 );

	shark::check_distribution(
		uniform,
		uniform.low() + (uniform.high() - uniform.low())/2.,
		1./12. * boost::math::pow<2>( uniform.high() - uniform.low() ),
		1000000
	);

	boost::math::uniform_distribution<double> ud( 2., 9. );

	for( double x = 1.; x <= 10.; x += 0.1 )
		BOOST_CHECK_SMALL( uniform.p( x ) - boost::math::pdf( ud, x ), 1E-5 );
}

BOOST_AUTO_TEST_SUITE_END()
