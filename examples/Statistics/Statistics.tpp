/*!
 * 
 *
 * \brief       Illustrates usage of the statistics component.
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

//###begin<includes>
#include <shark/Statistics/Statistics.h>
#include <shark/Core/Random.h>
//###end<includes>

using namespace shark;


int main(int argc, char** argv)
{
//###begin<table>
	statistics::ResultTable<double> table(2,"VarianceOfGaussian");//set a name for the results
	table.setDimensionName(0,"input1");
	table.setDimensionName(1,"input2");
//###end<table>

//###begin<sample>
	// Fill the table with randomly generated numbers for different variances and mean and also add missing values
	for(std::size_t k = 1; k != 10; ++k){
		double var= 10.0*k;
		for (std::size_t i = 0; i < 10000; i++){
			double value1=random::gauss(random::globalRng, 0,var);
			double value2=random::gauss(random::globalRng, 0,var);
			if(random::coinToss(random::globalRng) == 1)
				value2=statistics::missingValue();
			table.update(var,value1,value2 );
		}
	}
//###end<sample>
	
//###begin<statistics>
	statistics::Statistics<double> stats(&table);
	stats.addStatistic(statistics::Mean());//adds a statistic "Mean" for each variable
	stats.addStatistic("Variance", statistics::Variance());//explicit name
	stats.addStatistic("Missing", statistics::FractionMissing());
//###end<statistics>

//###begin<csv>
	printCSV(stats);
//###end<csv>
}
