//===========================================================================
/*!
 * 
 * \file        Pgm.h
 *
 * \brief       Importing and exporting PGM images
 * 
 * 
 *
 * \author      C. Igel
 * \date        2011
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

#ifndef SHARK_DATA_IMPORT_PGM_H
#define SHARK_DATA_IMPORT_PGM_H

#include <iostream>
#include <exception>

#include <stdio.h>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <shark/SharkDefs.h>
#include <shark/LinAlg/Base.h>
#include <shark/Data/Dataset.h>

namespace shark {

namespace detail {
void importPGM( const char * fileName, unsigned char ** ppData, int & sx, int & sy )
{
	FILE * fp = fopen(fileName, "rb");
	
	if ( 0 == fp ) throw( SHARKEXCEPTION( "[importPGM] cannot open file" ) );

	char format[16];
	const int nParamRead0 = fscanf(fp, "%s\n", (char *) &format);
	if ( 0 == nParamRead0 ) throw( std::invalid_argument( "[importPGM] cannot read file" ) );
	
	// Ignore comments
	char tmpCharBuf[256];
	fpos_t position;
	fgetpos (fp, &position);
	while ( true ) {
		char *s = fgets( tmpCharBuf, 255, fp );
		if (!s)  throw( SHARKEXCEPTION( "[importPGM] error reading file" ) );
		const int cnt = strncmp( tmpCharBuf, "#", 1 );
		if (0 != cnt) {
			fsetpos(fp, &position);
			break;
		} else {
			fgetpos (fp, &position);
		}
	}
	
	int nGrayValues;
	const int nParamRead1 = fscanf( fp, "%d %d\n", &sx, &sy );
	const int nParamRead2 = fscanf( fp, "%d\n", &nGrayValues );
	
	if ( (nParamRead1 != 2) || (nParamRead2 != 1) ) {
		fclose(fp);
		throw( std::invalid_argument( "[importPGM] file corrupted or format not recognized" ) );
	} else {
		if ( (0 == strncmp("P5", format, 2)) && ( (255 == nGrayValues) || (256 == nGrayValues) ) ) {
			//delete[] *ppData;
			*ppData = new unsigned char[sx*sy];
			fseek(fp, -sx*sy*sizeof(unsigned char), SEEK_END);
			
			const int readcount = (int)( fread(*ppData, sx*sizeof(unsigned char), sy, fp) );
			if (sy != readcount) {
				fclose(fp);
				throw( std::invalid_argument( "[importPGM] file corrupted or format not recognized" ) );
			}
		} else 	{
			fclose(fp);
			throw( std::invalid_argument( "[importPGM] file corrupted or format not recognized" ) );
		}
	}
	fclose(fp);
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
void writePGM( const char * fileName, const unsigned char * pData, const unsigned int sx, const unsigned int sy )
{
	FILE* fp = fopen(fileName, "wb");
	if( !fp ) throw( std::invalid_argument( "[writePGM] cannot open file" ) );
	
	fprintf(fp, "P5\n");
	fprintf(fp, "%d %d\n255\n", sx, sy);
	
	if( 1 != fwrite(pData, sx*sy, 1, fp) ) 	{
		fclose(fp);
		throw( std::invalid_argument( "[writePGM] write dat to file" ) );
	}
	fclose(fp);
}
} // end namespace detail

/// \brief Import a PGM image from file
///
/// \param  fileName   The file to read from
/// \param  data       Linear object for storing image 
/// \param  sx         Width of imported image
/// \param  sy         Height of imported image
template <class T>
void importPGM( const char * fileName, T &data, int & sx, int & sy ) {
	unsigned char *pData;
	unsigned i=0;
	detail::importPGM(fileName, &pData, sx, sy);
	data.resize(sx*sy);
	typename T::iterator it = data.begin();
	for( ; it != data.end(); ++it, ++i ) *it = pData[i];
	delete [] pData;
}

/// \brief Export a PGM image to file
///
/// \param  fileName   File to write to
/// \param  data       Linear object storing image 
/// \param  sx         Width of image
/// \param  sy         Height of image
/// \param  normalize  Adjust values to [0,255], default false
template <class T>
void exportPGM( const char * fileName, const T &data, int sx, int sy, bool normalize = false ) {
	unsigned i=0;
	unsigned char *pData = new unsigned char[data.size()];
	typename T::const_iterator it = data.begin();
	if(normalize) {
		double lb, ub;
		ub = lb = *it; 
		it++;
		for( ; it != data.end(); ++it) {
			if(*it > ub) ub = *it;
			if(*it < lb) lb = *it;
		}
		for( it = data.begin() ; it != data.end(); ++it, ++i ) pData[i] = (unsigned char)( (*it - lb) / (ub - lb) * 255 );
	} else {
		for( it = data.begin() ; it != data.end(); ++it, ++i ) pData[i] = (unsigned char)( *it );
	}
	detail::writePGM(fileName, pData, sx, sy);
	delete [] pData;
}


/// \brief Stores name and size of image externally
///
struct ImageInformation {
	int x;
	int y;
	std::string name;

	template<typename Archive>
	void serialize(Archive & ar, const unsigned int) {
		ar & x;
		ar & y;
		ar & name;
	}
};

/// \brief Import PGM images scanning a directory recursively
///
/// \param  p          Directory
/// \param  container  Container storing images
/// \param  info       Vector storing image informations
template<class T>
void importPGMDir(const std::string &p, T &container, std::vector<ImageInformation> &info)
{
	typedef typename T::value_type InputType;


	if (boost::filesystem::is_directory(p)) {
		for (boost::filesystem::recursive_directory_iterator itr(p); itr!=boost::filesystem::recursive_directory_iterator(); ++itr) {
			if (boost::filesystem::is_regular(itr->status())) {
				if ((boost::filesystem::extension(itr->path()) == ".PGM") ||
				    (boost::filesystem::extension(itr->path()) == ".pgm")) {
					InputType img;
					ImageInformation imgInfo;
					importPGM(itr->path().string().c_str(), img, imgInfo.x, imgInfo.y);
					imgInfo.name = itr->path().filename().string().c_str();
					container.push_back(img);
					info.push_back(imgInfo);
				}
			}
		}
	} else {
		throw( std::invalid_argument( "[importPGMDir] cannot open file" ) );
	}
}

/// \brief Import PGM images scanning a directory recursively
///
/// \param  p       Directory
/// \param  set     Set storing images
/// \param  setInfo Vector storing image informations
template<class T>
void importPGMSet(const std::string &p, Data<T> &set, Data<ImageInformation> &setInfo)
{
	std::vector<T> tmp;
	std::vector<ImageInformation> tmpInfo;
	importPGMDir(p, tmp, tmpInfo);
	set = createDataFromRange(tmp);
	setInfo = createDataFromRange(tmpInfo);
}

/** @}*/

} // end namespace shark
#endif
