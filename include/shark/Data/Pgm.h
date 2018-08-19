//===========================================================================
/*!
 * 
 *
 * \brief       Importing and exporting PGM images
 * 
 * 
 *
 * \author      C. Igel
 * \date        2011
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
//===========================================================================

#ifndef SHARK_DATA_IMPORT_PGM_H
#define SHARK_DATA_IMPORT_PGM_H


#include <shark/LinAlg/Base.h>
#include <shark/Core/Images/ReadImage.h>
#include <shark/Core/Images/WriteImage.h>
#include <shark/Data/Dataset.h>
#include <boost/filesystem.hpp>
namespace shark {

/// \brief Exports a set of filters as a grid image
///
/// It is assumed that the filters each form a row in the filter-matrix.
/// Moreover, the sizes of the filter images has to be given and it must gold width*height=W.size2().
/// The filters a re printed on a single image as a grid. The grid will be close to square. And the
/// image are separated by a black 1 pixel wide line. 
/// The output will be normalized so that all images are on the same scale.
/// \param  basename   File to write to. ".pgm" is appended to the filename
/// \param  filters    Matrix storing the filters row by row
/// \param  width      Width of the filter image
/// \param  height     Height of th filter image
inline void exportFiltersToPGMGrid(std::string const& basename, RealMatrix const& filters,std::size_t width, std::size_t height) {
	SIZE_CHECK(filters.size2() == width*height);
	//try to get a square image
	std::size_t gridX = std::size_t(std::sqrt(double(filters.size1())));
	std::size_t gridY = gridX;
	while(gridX*gridY < filters.size1()) ++gridX;
	
	RealMatrix image((height+1)*gridY, (width+1)*gridX,min(filters));
	
	for(std::size_t filter = 0; filter != filters.size1(); ++filter){
		//get grid position from filter
		std::size_t i = filter/gridX;
		std::size_t j = filter%gridX;
		std::size_t startY = (height+1)*i;
		std::size_t startX = (width+1)*j;
		//copy images
		noalias(subrange(image,startY,startY+height,startX,startX+width)) = to_matrix(row(filters,filter),height,width);
	}
	//normalize to [0,1]
	image -= min(filters);
	image /=(max(filters) - min(filters));
	shark::image::writeImageToFile<double>(basename+".pgm", to_vector(image), {(height+1)*gridY, (width+1)*gridX, 1}, PixelType::Luma);
}

/// \brief Exports a set of filters as a grid image
///
/// It is assumed that the filters each form a row in the filter-matrix.
/// Moreover, the sizes of the filter images has to be given and it must gold width*height=W.size2().
/// The filters a re printed on a single image as a grid. The grid will be close to square. And the
/// image are separated by a black 1 pixel wide line. 
/// The output will be normalized so that all images are on the same scale.
/// \param  basename   File to write to. ".pgm" is appended to the filename
/// \param  filters    Matrix storing the filters row by row
/// \param  width      Width of the filter image
/// \param  height     Height of th filter image
inline void exportFiltersToPGMGrid(std::string const& basename, Data<RealVector> const& filters,std::size_t width, std::size_t height) {
	SIZE_CHECK(dataDimension(filters) == width*height);
	//try to get a square image
	std::size_t numFilters = filters.numberOfElements();
	std::size_t gridX = std::size_t(std::sqrt(double(numFilters)));
	std::size_t gridY = gridX;
	while(gridX*gridY < numFilters) ++gridX;
	
	double minimum = std::numeric_limits<double>::max();
	double maximum = -std::numeric_limits<double>::max();
	for(std::size_t i = 0; i != filters.size(); ++i){
		minimum =std::min(minimum,min(filters[i]));
		maximum =std::max(maximum,max(filters[i]));
	}
	
	RealMatrix image((height+1)*gridY,(width+1)*gridX,minimum);
	
	std::size_t filter = 0;
	for(auto element: elements(filters)){
		//get grid position from filter
		std::size_t i = filter/gridX;
		std::size_t j = filter%gridX;
		std::size_t startY = (height+1)*i;
		std::size_t startX = (width+1)*j;
		RealVector filterImage = element;
		//copy images
		noalias(subrange(image,startY,startY+height,startX,startX+width)) = to_matrix(filterImage,height,width);
		++filter;
	}
	//normalize to [0,1]
	image -= minimum;
	image /=(maximum - minimum);
	shark::image::writeImageToFile<double>(basename+".pgm", to_vector(image), {(height+1)*gridY, (width+1)*gridX, 1}, PixelType::Luma);
}

/// \brief Import PGM images scanning a directory recursively
///
/// All images are required to have the same size. the shape of the images is stored in set.shape()
///
/// \param  p       Directory
/// \param  set     Set storing images
template<class T>
void importPGMSet(std::string const&p, Data<T> &set){
	std::vector<T> container;
	std::vector<std::pair<std::size_t,std::size_t> > info;
	if (boost::filesystem::is_directory(p)) {
		for (boost::filesystem::recursive_directory_iterator itr(p); itr!=boost::filesystem::recursive_directory_iterator(); ++itr) {
			if (boost::filesystem::is_regular(itr->status())) {
				if ((boost::filesystem::extension(itr->path()) == ".PGM") ||
				    (boost::filesystem::extension(itr->path()) == ".pgm")) {
					T img;
					std::pair<std::size_t,std::size_t> imgInfo;
					auto result = shark::image::readImageFromFile<double>(itr->path().string());
					container.push_back(result.first);
					imgInfo.first = result.second[1];
					imgInfo.second = result.second[0];
					container.push_back(img);
					info.push_back(imgInfo);
				}
			}
		}
	} else {
		throw( std::invalid_argument( "[importPGMDir] cannot open file" ) );
	}
	
	//check all images have same size
	for(auto const& i: info){
		if(i.first != info.front().first || i.second != info.front().second){
			throw SHARKEXCEPTION("[importPGMSet] all images are required to have the same size");
		}
	}
	set = createDataFromRange(container);
	set.setShape({info.front().second,info.front().first});
}

/** @}*/

} // end namespace shark
#endif
