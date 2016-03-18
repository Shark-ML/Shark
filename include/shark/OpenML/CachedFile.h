//===========================================================================
/*!
 * 
 *
 * \brief       Handling of downloadable files in a disk cache.
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
//===========================================================================

#ifndef SHARK_OPENML_CACHEDFILE_H
#define SHARK_OPENML_CACHEDFILE_H

#include "Base.h"
#include <boost/filesystem.hpp>


namespace shark {
namespace openML {


/// \brief Representation of a possibly cached downloadable file.
SHARK_EXPORT_SYMBOL class CachedFile
{
public:
	/// \brief Construct a cached file object with a cache location.
	CachedFile(std::string const& url, std::string const& filename)
	: m_url(url)
	, m_filename(m_cacheDirectory / filename)
	{ }


	/// \brief Obtain the path of the directory where dataset files are stored.
	static PathType const& cacheDirectory()
	{ return m_cacheDirectory; }

	/// \brief Set the path of the directory where dataset files are stored.
	static void setCacheDirectory(PathType const& cacheDirectory)
	{ m_cacheDirectory = cacheDirectory; }

	std::string const& url() const
	{ return m_url; }

//	void setUrl(std::string const& url) const
//	{ m_url = url; }

	PathType const& filename() const
	{ return m_filename; }

//	void setFilename(std::string const& filename) const
//	{ m_filename = m_cacheDirectory / filename; }

	bool downloaded() const
	{ return boost::filesystem::exists(m_filename); }

	bool download();

protected:
	std::string m_url;                 ///< URL of the original resource
	PathType m_filename;               ///< filename of the cached resource
	static PathType m_cacheDirectory;  ///< directory where new cache entries are placed
};


};  // namespace openML
};  // namespace shark
#endif
