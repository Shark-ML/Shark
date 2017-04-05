//###begin<includes>
#include <shark/Core/Random.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

using namespace shark;
using namespace std;
//###end<includes>


int main(int argc, char** argv)
{

//###begin<seed>
	random::seed( 1234 );
//###end<seed>

	// Get random "numbers" for all subsumed random number generators:
//###begin<draw>
	bool   rn1 = random::coinToss( );
	long   rn2 = random::discrete( );
	double rn3 = random::uni( );
	double rn4 = random::gauss( );
	double rn5 = random::cauchy( );
	long   rn6 = random::geom( );
	long   rn7 = random::diffGeom( );
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
	Weibull<> dist1( shark::random::globalrandom );
	Bernoulli<> dist2( shark::random::globalrandom );
	Binomial<> dist3( shark::random::globalrandom );
	Cauchy<> dist4( shark::random::globalrandom );
	DiffGeometric<> dist5( shark::random::globalrandom );
	Dirichlet<> dist6( shark::random::globalrandom );
	DiscreteUniform<> dist7( shark::random::globalrandom );
	Erlang<> dist8( shark::random::globalrandom );
	Gamma<> dist9( shark::random::globalrandom );
	Geometric<> dist10( shark::random::globalrandom );
	HyperGeometric<> dist12( shark::random::globalrandom );
	LogNormal<> dist13( shark::random::globalrandom );
	NegExponential<> dist14( shark::random::globalrandom );
	Normal<> dist15( shark::random::globalrandom );
	Poisson<> dist16( shark::random::globalrandom );
	Uniform<> dist17( shark::random::globalrandom );
//###end<list>

//###begin<normal>
	Normal< random::rng_type > normal( random::globalrandom, 1., 1. );

        double mean = normal.mean();
        double variance = normal.variance();
	cout << mean << " (" << variance << ")" << endl;
//###end<normal>

//###begin<uniform>
	Uniform< shark::random::rng_type > uniform( shark::random::globalrandom, 1, 5 );
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
