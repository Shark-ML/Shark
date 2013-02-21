/**
 *
 *  \brief Simple utility function for http server programming.
 *
 *  \author  T. Glasmachers
 *  \date    2013
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *  
 */
#ifndef SHARK_NETWORK_TOOLS_H
#define SHARK_NETWORK_TOOLS_H


#include <string>


namespace shark {
namespace http {


/// \brief Encode a string so it becomes a valid URL.
///
/// \par
/// E.g., replace ' ' with "%20" or '+'.
std::string cgiEncode(std::string s);

/// \brief Decode a URL-encoded string.
///
/// E.g., replace "%20" with ' '.
std::string cgiDecode(std::string s);

/// \brief Read and return the contents of a file.
///
/// On error (i.e., file not found) the function throws an exception.
std::string readFile(std::string filename);


}}
#endif
