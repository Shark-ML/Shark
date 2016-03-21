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
#include <shark/Core/Exception.h>
#include <boost/filesystem.hpp>
#include <mutex>


namespace shark {
namespace openML {


/// \brief Representation of a possibly cached downloadable file.
SHARK_EXPORT_SYMBOL class CachedFile
{
public:
	/// \brief Construct a cached file object with a cache location.
	CachedFile(std::string const& filename = "", std::string const& url = "")
	: m_url(url)
	{
		if (! filename.empty()) m_filename = m_cacheDirectory / filename;
	}


	/// \brief Obtain the path of the directory where dataset files are stored.
	static PathType const& cacheDirectory()
	{ return m_cacheDirectory; }

	/// \brief Set the path of the directory where dataset files are stored.
	static void setCacheDirectory(PathType const& cacheDirectory)
	{ m_cacheDirectory = cacheDirectory; }

	/// \brief Obtain the URL from which the file is obtained.
	std::string const& url() const
	{ return m_url; }

	/// \brief Set the URL of this file.
	///
	/// Use this function only once, and only if the URL was not set in the constructor.
	void setUrl(std::string const& url)
	{
		if (! m_url.empty()) throw SHARKEXCEPTION("[CachedFile::setUrl] the URL is already defined");
		m_url = url;
	}

	/// \brief Set the name of this file (placed in the cache directory).
	///
	/// Use this function only once, and only if the filename was not set in the constructor.
	void setFilename(std::string const& filename)
	{
		if (! m_filename.empty()) throw SHARKEXCEPTION("[CachedFile::setUrl] the filename is already defined");
		m_filename = m_cacheDirectory / filename;
	}

	/// \brief Obtain the path where the file is cached in the file system.
	PathType const& filename() const
	{ return m_filename; }

	/// \brief Check whether the file was already downloaded and cached.
	bool downloaded() const;

	/// \brief Download the file from the URL and store it as a (local) file.
	void download() const;

private:
	PathType m_filename;               ///< filename of the cached resource
	std::string m_url;                 ///< URL of the original resource
	static PathType m_cacheDirectory;  ///< directory where new cache entries are placed
	mutable std::mutex m_mutex;        ///< mutex protecting the file
};


};  // namespace openML
};  // namespace shark
#endif
