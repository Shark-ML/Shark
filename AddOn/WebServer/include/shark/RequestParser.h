/**
 *
 *  \brief Parser for an http request.
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
#ifndef SHARK_NETWORK_REQUESTPARSER_H
#define SHARK_NETWORK_REQUESTPARSER_H


#include <shark/Network/Request.h>
#include <string>
#include <map>


namespace shark {
namespace http {


//
// Incremental parser for HTTP requests.
//
// The class supports iterative parsing of a possibly long http
// request. It processes GET and POST requests with url-encoding
// and multipart-form-data encoding, including file uploads. The
// information is stored in a Request object.
//
class RequestParser
{
public:
	typedef std::map<std::string, std::string> stringmap;

	RequestParser(Request& request);
	~RequestParser();


	// Start handling a new request; invalidate the underlying request object.
	void reset();

	// Add more data to the request as it arrives.
	// A return value of false indicates an invalid request
	// that should result in closing the socket.
	bool processData(const char* buffer, std::size_t size, std::size_t& used);

	// is request parsing finished?
	inline bool isReady() const
	{ return (m_status == esReady); }

private:
	// Check for an end-of-line (CR-LF or LF) in the buffer.
	// If found return true and set content and length to the
	// size of the content (excluding the end-of-line) and of
	// the line (including the end-of-line).
	static bool extractLine(const char* buffer, std::size_t size, std::size_t& content, std::size_t& length);

	// convert a hex digit to a number, return -1 for non hex digits
	static int tohex(char c);

	// decode an url-encoded string
	static bool decode(std::string str, std::string& ret, bool decodeplus = true);

	// extract an url-encoded parameter
	bool extractParam(std::string str);

	// extract a list of url-encoded parameters
	bool extractParams(std::string str);

	// extract data from a multipart message
	bool handleMultipartFormData();

	// parser status
	enum eStatus
	{
		esRequestLine,
		esHeaderLine,
		esBody,
		esInterpret,
		esReady,
	};

	// body encoding (if any)
	enum eBodyType
	{
		ebNone,
		ebLength,
		ebChunked,
	};

	// request object to be filled in
	Request& m_request;

	// data and parser state
	eStatus m_status;
	eBodyType m_bodytype;
	std::size_t m_current;
	std::size_t m_bodylength;
	std::string m_uri;
	stringmap m_header;
	std::string m_body;
};


}}
#endif
