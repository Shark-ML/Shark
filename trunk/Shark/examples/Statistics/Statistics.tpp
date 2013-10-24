/**
*
*  \brief Illustrates usage of the statistics component.
*
*  \author T.Voss
*  \date 2010
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
*
*/

//###begin<includes>
#include <shark/Statistics/Statistics.h>
#include <shark/Rng/GlobalRng.h>
//###end<includes>

using namespace shark;


int main(int argc, char** argv)
{
//###begin<stat>
	Statistics stats;
//###end<stat>

//###begin<sample>
	// Sample 10000 standard normally distributed random numbers
	// and update statistics for these numbers iteratively.
	for (std::size_t i = 0; i < 100000; i++)
		stats( Rng::gauss() );
//###end<sample>

//###begin<cout>
	// Output results to the console.
	std::cout << stats << std::endl;
//###end<cout>

//###begin<indiv>
	std::cout << 
		stats( Statistics::NumSamples() ) << " " <<
		stats( Statistics::Min() ) << " " <<
		stats( Statistics::Max() ) << " " <<
		stats( Statistics::Mean() ) << " " << 
		stats( Statistics::Variance() ) << " " <<
		stats( Statistics::Median() ) << " " <<
		stats( Statistics::LowerQuartile() ) << " " <<
		stats( Statistics::UpperQuartile() ) << std::endl;
//###end<indiv>
}
