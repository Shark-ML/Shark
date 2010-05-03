/*!
*  \file TSP_GA.cpp
*
*  \author Pavel Saviankou
*
*  \brief A 10-city traveling salesman problem as presented in <p>
*    D. E. Goldberg and R. Lingle, Alleles, loci, and traveling
*    salesman problem. In <em> Proc. of the International Conference on
*    Genetic Algorithms and Their Applications</em>, pages 154-159,
*    Pittsburg, PA, 1985 </p> 
*
* The traveling salsman problem is a combinatorial optimization task. A
* salesman is supposed to visit $n$ cities. Each travelling connection
* is associated with a cost (i.e. the time fot the trip). The problem is
* to find the cheapest round-route that visits each city exactly once
* and returns to the starting point.
*
*
* \begin{figure}(4cm,4cm)
* \put(0.1cm,0.2cm){\textbf{A}}
* \put(3.7cm,0.8cm){\textbf{B}}
* \put(3.3cm,3.7cm){\textbf{C}}
* \put(0.7cm,3.2cm){\textbf{D}}
* \put(0.5cm,0.5cm){\circle{0.2cm}}
* \put(3.5cm,1.cm){\circle{0.2cm}}
* \put(3cm,3.5cm){\circle{0.2cm}}
* \put(1cm,3cm){\circle{0.2cm}}
* \put(0.5cm,0.5cm){\line(3,0.5){3.041cm}}
* \put(3.5cm,1.0cm){\line(-0.5,2.5){-2.550cm}}
* \put(3.0cm,3.5cm){\line(2,0.5){-2.062cm}}
* \put(1.0cm,3.0cm){\line(0.5,2.5){-2.550cm}}
* \put(0.5cm,0.5cm){\line(2.5,3){3.905cm}}
* \put(1.0cm,3.0cm){\line(2.5,2){3.202cm}}
* \end{figure}
*
* 
*  \par
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR> 
*
*  \par Project:
*      EALib
*
*
*  <BR><HR>
*  This file is part of ReClaM. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 2, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, write to the Free Software
*  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <list>
#include <vector>
#include <EALib/PopulationT.h>
#include <Array/ArrayOp.h>
#include <Array/ArrayIo.h>
#include <SharkDefs.h>

using namespace std;

#define __NO_BITPACKING__ 1

// ==================================================================

const unsigned int cities10[10][10] =
                        {
                                {       0,      28,     57,     72,     81,     85,     80,     113,    89,     80},
                                {       28,     0,      28,     45,     54,     57,     63,     85,     63,     63},
                                {       57,     28,     0,      20,     30,     28,     57,     57,     40,     57},
                                {       72,     45,     20,     0,      10,     20,     72,     45,     20,     45},
                                {       81,     54,     30,     10,     0,      22,     81,     41,     10,     41},
                                {       85,     57,     28,     20,     22,     0,      63,     28,     28,     63},
                                {       80,     63,     57,     72,     81,     63,     0,      80,     89,     113},
                                {       113,    85,     57,     45,     41,     28,     80,     0,      40,     80},
                                {       89,     63,     40,     20,     10,     28,     89,     40,     0,      40},
                                {       80,     63,     57,     45,     41,     63,     113,    80,     40,     0}
                        };


// ==================================================================


signed int evaluate(ChromosomeT<unsigned int>& chromo)
{
  unsigned	size 	= chromo.size();
  signed int	sum	= 0;
  for(unsigned i=0; i<size; i++)
    {
      if (cities10[chromo[i]][chromo[ (i+1) % size ]] == 0)
	{
	  throw SHARKEXCEPTION("ERROR, the same city ");
	}
      sum += cities10[chromo[i]][chromo[ (i+1) % size ]];
    }
  return sum;
}
// =====================================================================


// ==========================================================================

void generatePermutation(ChromosomeT<unsigned int>& chromo)
{
        unsigned                size	= chromo.size();
	unsigned		i	= 0;
        Array<unsigned int>     numbers(size-1);
        numbers.resize(size-1);

        for(i=0; i<numbers.nelem(); i++)
        {
          numbers(i)            = i+1;
        }

        chromo[0] = 0;
        for(i=1; i<size; i++)
        {
                unsigned index  = Rng::discrete (0,numbers.nelem()-1);
                chromo[i]       = numbers(index);
                numbers.remove_row(index);
        }
}




unsigned _find_city(unsigned city_name,unsigned n_city, std::vector<unsigned>& tour)
{
        unsigned j1 = 0;
        while((j1 < n_city) && (tour[j1]!=city_name))
          {
            j1 = j1+1;
          }

        return j1;
}



void _swap_city(unsigned city_pos1,unsigned city_pos2, std::vector<unsigned>& tour)
{
        unsigned          temp;
        temp            = tour[city_pos1];
        tour[city_pos1] = tour[city_pos2];
        tour[city_pos2] = temp;
}



void _cross_tour(
                unsigned n_city, unsigned lo_cross, unsigned hi_cross,
                std::vector<unsigned>& tour1_old, std::vector<unsigned>& tour2_old,
                std::vector<unsigned>& tour1_new, std::vector<unsigned>& tour2_new
                )
{
        SIZE_CHECK( tour1_old.size() == tour2_old.size());

        unsigned        j1 = 0;
        unsigned        hi_test = hi_cross + 1;

        if (hi_test > (n_city-1)) hi_test = 0;

        tour1_new.resize(n_city);
        tour2_new.resize(n_city);

        for(unsigned i = 0; i<n_city; i++)
        {
                tour1_new[i]    = tour1_old[i];
                tour2_new[i]    = tour2_old[i];
        }

        if ( (lo_cross != hi_cross ) && (lo_cross != hi_test ))
        {
                j1      = lo_cross;
                while( j1 != hi_test)
                {
                  _swap_city(j1, _find_city(tour1_old[j1], n_city, tour2_new), tour2_new);
                  _swap_city(j1, _find_city(tour2_old[j1], n_city, tour1_new), tour1_new);
                  j1=j1+1; if (j1>(n_city-1)) j1 = 0;
                }
        }
}



void goldbergsCrossoverPMX(ChromosomeT<unsigned int>& mom, ChromosomeT<unsigned int>& dad)
{
        SIZE_CHECK( mom.size() == dad.size());

        unsigned                n_city          = mom.size()-1;
        unsigned                lo_cross        = 0;
        unsigned                hi_cross        = 0;
	unsigned		i		= 0;

        while(lo_cross == hi_cross || lo_cross == hi_cross+1)
          {
            lo_cross = Rng::discrete(0,n_city-1);
            hi_cross = Rng::discrete(0,n_city-1);
          }

        std::vector<unsigned>   mom_vector(n_city);
        std::vector<unsigned>   dad_vector(n_city);
        std::vector<unsigned>   new_mom_vector(n_city);
        std::vector<unsigned>   new_dad_vector(n_city);

        new_mom_vector.resize(n_city);
        new_dad_vector.resize(n_city);
        mom_vector.resize(n_city);
        dad_vector.resize(n_city);

        for(i = 0; i<n_city; i++)
        {
                mom_vector[i] = mom[i+1];
                dad_vector[i] = dad[i+1];
        }

        _cross_tour(n_city, lo_cross, hi_cross, mom_vector, dad_vector, new_mom_vector, new_dad_vector);

	for(i = 0 ; i<n_city; i++){
                mom[i+1] = new_mom_vector[i];
                dad[i+1] = new_dad_vector[i];
	}
}


//=======================================================================
//
// main program
//
int main( int argc, char **argv )
{
  //
  // constants
  //
  const	unsigned Mu           	= 200;
  const	unsigned Lambda       	= 200;
  const	unsigned cities    	= 10;
  const	unsigned MaxEvals	= 200*30;
  const	unsigned Runs		= 1;
  const	unsigned Iterations	= (unsigned int)(MaxEvals/Lambda);
  const	unsigned numElistis	= 0;
  unsigned	 i, t;
  const double	 Pc       	= 0.6;
  vector<double> window;

  //for self-testing, please ignore
  double meanfitness=0.;
  //end selft-testing block

  std::cout << "Running GA for TSP problem with " << cities << " cities..."<<std::endl;
  std::cout.flush();
  //
  // initialize random number generator
  //
  
  for (unsigned tt=0; tt < Runs; tt++)
    {

      // initialize random number generator
 
      Rng::seed(98765433+tt);
      
      // define populations
      //
      PopulationT<unsigned int> parents   ( Mu,     ChromosomeT< unsigned int >( cities ) );
      parents.spinWheelMultipleTimes( );
      parents   .setMinimize( );

      PopulationT<unsigned int> offsprings( Lambda, ChromosomeT< unsigned int >( cities ) );
      offsprings.spinWheelMultipleTimes( );
      offsprings.setMinimize( );

      window.resize(Lambda);

      // initialize parent population
      for( i = 0; i < parents.size( ); ++i )
	{
	  generatePermutation(parents[ i ][ 0 ]);
	}

      // iterate
      //
      for( t = 0; t < Iterations; ++t )
	{
	  for( i = 0; i < parents.size( ); i++ )
	    {
	      double fitness = evaluate(parents[i][0]);
	      parents[i].setFitness(fitness);
	    }
	  
	  parents.linearDynamicScaling(window,0);

	  // generate output
	  //	 

	  cout<<"Run: "<<tt<<", Iteration: "<<t<<", mean Fitness: "<< (parents.meanFitness( ))<<endl;
	  
	  
	  // generate new offsprings: parents, offspring? OK
	  //
	  offsprings.selectProportional(parents, numElistis);
	  
	  
	  for(i=0; i<offsprings.size()-1; i+=2)
	    {
	      //cout << "Iter " << i << endl;  
	      if(Rng::coinToss(Pc)) goldbergsCrossoverPMX(offsprings[i][ 0 ], offsprings[i+1][ 0 ]);
	     }
	  parents = offsprings;
       }
      //for self-testing, please ignore
      meanfitness+=parents.meanFitness();
      //end self-testing block
    }

  // lines below are for self-testing this example, please ignore
  if(meanfitness/Runs<382.556) exit(EXIT_SUCCESS);
  else exit(EXIT_FAILURE);
}




