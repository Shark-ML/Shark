//###begin<includes>
#include <shark/Rng/GlobalRng.h>
//###end<includes>

using namespace shark;


int main(int argc, char** argv)
{

//###begin<sample>
	// Sample 10000 standard normally distributed random numbers
	// and update statistics for these numbers iteratively.
	for (std::size_t i = 0; i < 100000; i++)
		stats( Rng::gauss() );
//###end<sample>

}
