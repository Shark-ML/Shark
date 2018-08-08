/*!
 * 
 *
 * \brief       Implements support for Zip-Archives
 * 
 * 
 *
 * \author      O.Krause
 * \date        2018
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
#ifndef SHARK_CORE_ZIP_SUPPORT_H
#define SHARK_CORE_ZIP_SUPPORT_H


#include <memory>
#include <string>
#include <vector>
namespace shark{
/// \brief Class for opening zip files for reading
class ZipReader{
public:
	/// \brief Opens an unencrpyted zip-file for reading.
	ZipReader(std::string const& pathToArchive);
	~ZipReader();

	/// \brief Returns the number of files in the archive.
	std::size_t numFiles() const;
	/// \brief Returns the name of the i-th file
	std::string const& fileName(std::size_t i) const;
	/// \brief Compiles a list of files and returns it.
	///
	/// Note that this operation is expensive if the archive contains many files
	std::vector<std::string> fileNames() const;
	
	/// \brief Loads the i-th file with name fileName(i)
	std::vector<unsigned char> readFile(std::size_t i) const;
	/// \brief Returns the file with given name
	std::vector<unsigned char> readFile(std::string const& name) const;
private:
	
	struct ZipImpl;//Pimpl-Idiom to prevent having to include zip.h
	std::unique_ptr<ZipImpl> m_impl;
};
}

#endif 
