/*!
 * 
 *
 * \brief       Illustration of the OpenML component.
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2016
 *
 *
 * \par Copyright 1995-2016 Shark Development Team
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

//###begin<includes>
#include <shark/OpenML/OpenML.h>
#include <shark/Data/Arff.h>
//###end<includes>
#include <iostream>

using namespace shark;


std::string api_key = /*"<insert your OpenML api key here>"*/ "3f31e02d6bcce75d4a869c54f3bd05fa";


int main(int argc, char** argv)
{
//###begin<key>
	openML::connection.setKey(api_key);
//###end<key>

	openML::Dataset dataset(11);
	dataset.download();
	dataset.print();

	LabeledData<CompressedRealVector, unsigned int> data;
	importARFF(dataset.filename().string(), dataset.labelname(), data);
	exportARFF("dataset.arff", data, "dataset-11");
}
