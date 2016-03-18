
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


#include <shark/OpenML/CachedFile.h>
#include <shark/OpenML/Connection.h>
#include <fstream>


namespace shark {
namespace openML {


// static
PathType CachedFile::m_cacheDirectory = boost::filesystem::current_path();


bool CachedFile::download()
{
	if (downloaded()) return true;

	std::string host;
	std::string resource;
	if (m_url.size() >= 7 && m_url.substr(0, 7) == "http://")
	{
		std::size_t slash = m_url.find('/', 7);
		if (slash == std::string::npos) return false;
		host = m_url.substr(7, slash - 7);
		resource = m_url.substr(slash);
	}
	else
	{
		std::size_t slash = m_url.find('/');
		if (slash == std::string::npos) return false;
		host = m_url.substr(0, slash);
		resource = m_url.substr(slash);
	}

	Connection conn(host);
	detail::HttpResponse response = conn.getHTTP(resource);
	if (response.statusCode() != 200) return false;

	std::ofstream os(m_filename.string());
	os << response.body();
	os.close();
	return true;
}


};  // namespace openML
};  // namespace shark
