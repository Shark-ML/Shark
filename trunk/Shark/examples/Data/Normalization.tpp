//===========================================================================
/*!
 * 
 *
 * \brief       Data Normalization
 * 
 * This file is part of the tutorial "Normalization of Input Data".
 * By itself, it does not do anything particularly useful.
 *
 * \author      T. Glasmachers
 * \date        2014
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
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
//===========================================================================

#include <shark/Data/Csv.h>

//###begin<includes1>
#include <shark/Models/Normalizer.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsUnitVariance.h>
using namespace shark;
//###end<includes1>

//###begin<includes2>
#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/NormalizeComponentsWhitening.h>
//###end<includes2>

int main()
{
	// load data
	UnlabeledData<RealVector> data;
	try
	{
		import_csv(data, "faithful.csv");
	}
	catch (std::exception const& ex)
	{
		std::cerr << "EXCEPTION loading data: " << ex.what() << std::endl
			<< "Please make sure that the file 'faithful.csv' can be found." << std::endl;
		exit(1);
	}

//###begin<training1>
	// create and train data normalizer
	bool removeMean = true;
	Normalizer<RealVector> normalizer;
	NormalizeComponentsUnitVariance<RealVector> normalizingTrainer(removeMean);
	normalizingTrainer.train(normalizer, data);
//###end<training1>

//###begin<transform1>
	// transform data
	UnlabeledData<RealVector> normalizedData = transform(data, normalizer);
//###end<transform1>

//###begin<training2>
	// create and train data normalizer
	LinearModel<RealVector> whitener;
	NormalizeComponentsWhitening whiteningTrainer;
	whiteningTrainer.train(whitener, data);
//###end<training2>

//###begin<transform2>
	// transform data
	UnlabeledData<RealVector> whitenedData = transform(data, whitener);
//###end<transform2>
}
