//###begin<includes>
#include <shark/Rng/GlobalRng.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

using namespace shark;
using namespace std;
//###end<includes>


int main(int argc, char** argv)
{

//###begin<seed>
	Rng::seed( 1234 );
//###end<seed>

	// Get random "numbers" for all subsumed random number generators:
//###begin<draw>
	bool   rn1 = Rng::coinToss( );
	long   rn2 = Rng::discrete( );
	double rn3 = Rng::uni( );
	double rn4 = Rng::gauss( );
	double rn5 = Rng::cauchy( );
	long   rn6 = Rng::geom( );
	long   rn7 = Rng::diffGeom( );
//###end<draw>

	// Output of random numbers:
	cout << "Bernoulli trial                              = " << rn1 << endl;
	cout << "Discrete distribution number                 = " << rn2 << endl;
	cout << "Uniform distribution number                  = " << rn3 << endl;
	cout << "Normal distribution number                   = " << rn4 << endl;
	cout << "Cauchy distribution number                   = " << rn5 << endl;
	cout << "Geometric distribution number                = " << rn6 << endl;
	cout << "Differential Geometric distribution number   = " << rn7 << endl;

//###begin<list>
	Weibull<> dist1( shark::Rng::globalRng );
	Bernoulli<> dist2( shark::Rng::globalRng );
	Binomial<> dist3( shark::Rng::globalRng );
	Cauchy<> dist4( shark::Rng::globalRng );
	DiffGeometric<> dist5( shark::Rng::globalRng );
	Dirichlet<> dist6( shark::Rng::globalRng );
	DiscreteUniform<> dist7( shark::Rng::globalRng );
	Erlang<> dist8( shark::Rng::globalRng );
	Gamma<> dist9( shark::Rng::globalRng );
	Geometric<> dist10( shark::Rng::globalRng );
	HyperGeometric<> dist12( shark::Rng::globalRng );
	LogNormal<> dist13( shark::Rng::globalRng );
	NegExponential<> dist14( shark::Rng::globalRng );
	Normal<> dist15( shark::Rng::globalRng );
	Poisson<> dist16( shark::Rng::globalRng );
	Uniform<> dist17( shark::Rng::globalRng );
//###end<list>

//###begin<normal>
	Normal< Rng::rng_type > normal( Rng::globalRng, 1., 1. );

        double mean = normal.mean();
        double variance = normal.variance();
	cout << mean << " (" << variance << ")" << endl;
//###end<normal>

//###begin<uniform>
	Uniform< shark::Rng::rng_type > uniform( shark::Rng::globalRng, 1, 5 );
//###end<uniform>

//###begin<multivariate>
	RealMatrix Sigma(2,2);
	Sigma(0,0) = 1;
	Sigma(0,1) = 2;
	Sigma(1,0) = .5;
	Sigma(1,1) = 1;
	MultiVariateNormalDistribution M(Sigma);

	for(unsigned i=0; i<100; i++) {
		cout << M().first(0) << " " << M().first(1) << endl;
	}
//###end<multivariate>

}
