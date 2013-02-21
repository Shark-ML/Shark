/**
 *
 *  \brief Representation of an http request.
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
#ifndef SHARK_NETWORK_REQUEST_H
#define SHARK_NETWORK_REQUEST_H


#include <string>
#include <map>


namespace shark {
namespace http {


class RequestParser;
// class Server;
// class Connection;


//
// HTTP request (GET, POST, or HEAD)
//
// This class acts as a container describing an http request.
// It provides access to the request method, the requested
// resource, and parameters of the request, all as strings.
//
// A request can contain "multipart" file uploads. In this case
// the value of the parameter holds the file contents, while the
// filename field holds the filename proposed by the client.
//
class Request
{
public:
	typedef std::map<std::string, std::string> stringmap;

	Request();
	~Request();


	// request method (GET/HEAD/POST)
	inline std::string method() const
	{ return m_method; }

	// requested resource, e.g., file name
	inline std::string resource() const
	{ return m_resource; }

	// return the request parameters
	inline stringmap const& parameters() const
	{ return m_parameter; }

	// check for the existence of a parameter
	inline bool hasParameter(std::string parameter) const
	{
		stringmap::const_iterator it = m_parameter.find(parameter);
		return (it != m_parameter.end());
	}

	// return the value of an existing parameter or the empty string
	inline std::string operator [] (std::string parameter) const
	{
		stringmap::const_iterator it = m_parameter.find(parameter);
		if (it == m_parameter.end()) return std::string();
		return (*it).second;
	}

	// check for the existence of a filename belonging to a
	// file upload
	inline bool hasFilename(std::string parameter) const
	{
		stringmap::const_iterator it = m_filename.find(parameter);
		return (it != m_parameter.end());
	}

	// return the filename associated with a parameter value, under
	// the assumption that this parameter describes a file upload
	inline std::string filename(std::string parameter) const
	{
		stringmap::const_iterator it = m_filename.find(parameter);
		if (it == m_filename.end()) return std::string();
		return (*it).second;
	}

private:
	friend class RequestParser;
	// friend class Server;
	// friend class Connection;

	// request description
	std::string m_method;
	std::string m_resource;
	stringmap m_parameter;
	stringmap m_filename;
};


}}
#endif
