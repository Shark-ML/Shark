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

#include <fstream>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/archive/text_oarchive.hpp>


#include <shark/LinAlg/Base.h>
#include <shark/Data/Dataset.h>

namespace shark {

namespace detail {
void importPGM( std::string const& fileName, std::vector<unsigned char>& ppData, std::size_t& sx, std::size_t& sy )
{
	std::ifstream file(fileName.c_str(), std::ios::binary);
	SHARK_RUNTIME_CHECK(file, "Can not open File");
	
	std::string id;
	std::size_t nGrayValues = 0;
	file>> id;
	SHARK_RUNTIME_CHECK(id == "P5" , "File " + fileName+ "is not a pgm");
	//ignore comments
	file >> std::ws;//skip white space
	while(file.peek() == '#'){
		file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}
	file >> sx >> sy >> nGrayValues;
	SHARK_RUNTIME_CHECK(file, "Error reading file!");
	SHARK_RUNTIME_CHECK(nGrayValues <= 255, "File " + fileName+ "unsupported format");
	
	ppData.resize(sx*sy);
	file.read((char*)ppData.data(),sx*sy);
	SHARK_RUNTIME_CHECK(file, "Error reading file!");
}

/**
 * \ingroup shark_globals
 *
 * @{
 */

/// \brief Writes a PGM file.
///
/// \param  fileName   File to write to
/// \param  pData      unsigned char pointer to the data
/// \param  sx         Width of image
/// \param  sy         Height of image
void writePGM( std::string const& fileName, std::vector<unsigned char> const& data, std::size_t sx, std::size_t sy )
{
	std::ofstream file(fileName.c_str(), std::ios::binary);
	SHARK_RUNTIME_CHECK(file, "Can not open File");

	file<<"P5\n"<<sx<<" "<<sy<<"\n"<<255<<"\n";
	file.write((char*)data.data(),sx*sy);
}
} // end namespace detail

/// \brief Import a PGM image from file
///
/// \param  fileName   The file to read from
/// \param  data       Linear object for storing image 
/// \param  sx         Width of imported image
/// \param  sy         Height of imported image
template <class T>
void importPGM( std::string const& fileName, T& data, std::size_t& sx, std::size_t& sy ) {
	std::vector<unsigned char> rawData;
	detail::importPGM(fileName, rawData, sx, sy);
	data.resize(sx*sy);
	std::copy(rawData.begin(), rawData.end(), data.begin());
}

/// \brief Export a PGM image to file
///
/// \param  fileName   File to write to
/// \param  data       Linear object storing image 
/// \param  sx         Width of image
/// \param  sy         Height of image
/// \param  normalize  Adjust values to [0,255], default false
template <class T>
void exportPGM(std::string const& fileName, T const& data, std::size_t sx, std::size_t sy, bool normalize = false) {
	SIZE_CHECK(sx*sy == data.size());
	std::vector<unsigned char> rawData(data.size());
	typename T::const_iterator it = data.begin();
	std::size_t i = 0;
	if(normalize) {
		double lb = *std::min_element(data.begin(),data.end());
		double ub = *std::max_element(data.begin(), data.end());
		for( it = data.begin() ; it != data.end(); ++it, ++i )
			rawData[i] = (unsigned char)( (*it - lb) / (ub - lb) * 255 );
	} else {
		for( it = data.begin() ; it != data.end(); ++it, ++i )
			rawData[i] = (unsigned char)( *it );
	}
	detail::writePGM(fileName, rawData, sx, sy);
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
inline void exportFiltersToPGMGrid(std::string const& basename, RealMatrix const& filters,std::size_t width, std::size_t height) {
	SIZE_CHECK(filters.size2() == width*height);
	//try to get a square image
	std::size_t gridX = std::size_t(std::sqrt(double(filters.size1())));
	std::size_t gridY = gridX;
	while(gridX*gridY < filters.size1()) ++gridX;
	
	RealMatrix image((height+1)*gridY,(width+1)*gridX,min(filters));
	
	for(std::size_t filter = 0; filter != filters.size1(); ++filter){
		//get grid position from filter
		std::size_t i = filter/gridX;
		std::size_t j = filter%gridX;
		std::size_t startY = (height+1)*i;
		std::size_t startX = (width+1)*j;
		//copy images
		noalias(subrange(image,startY,startY+height,startX,startX+width)) = to_matrix(row(filters,filter),height,width);
	}
	exportPGM(
		(basename+".pgm").c_str(), 
		blas::adapt_vector((height+1)*gridY*(width+1)*gridX,&image(0,0)),
		(width+1)*gridX, (height+1)*gridY,
		true
	);
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
	for(std::size_t i = 0; i != filters.numberOfBatches(); ++i){
		minimum =std::min(minimum,min(filters.batch(i)));
	}
	
	RealMatrix image((height+1)*gridY,(width+1)*gridX,minimum);
	
	for(std::size_t filter = 0; filter != numFilters; ++filter){
		//get grid position from filter
		std::size_t i = filter/gridX;
		std::size_t j = filter%gridX;
		std::size_t startY = (height+1)*i;
		std::size_t startX = (width+1)*j;
		RealVector filterImage =filters.element(filter);
		//copy images
		noalias(subrange(image,startY,startY+height,startX,startX+width)) = to_matrix(filterImage,height,width);
	}
	exportPGM(
		(basename+".pgm").c_str(), 
		blas::adapt_vector((height+1)*gridY*(width+1)*gridX,&image(0,0)),
		(width+1)*gridX, (height+1)*gridY,
		true);
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
					importPGM(itr->path().string().c_str(), img, imgInfo.first, imgInfo.second);
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
	set.shape() = {info.front().second,info.front().first};
}

/** @}*/

} // end namespace shark
#endif
