//===========================================================================
/*!
 * 
 *
 * \brief       Implementation of the OpenML Connection.
 * 
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


#include <shark/OpenML/OpenML.h>
#include <shark/OpenML/detail/Tools.h>
#include <shark/OpenML/detail/LZ77.h>
#include <shark/Core/Exception.h>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>


#define SYNCHRONIZE std::unique_lock<std::mutex> lock(m_mutex);


namespace shark {
namespace openML {


////////////////////////////////////////////////////////////


static const std::string OPENML_REST_API_HOST = "www.openml.org";
static const unsigned short OPENML_REST_API_PORT = 80;
static const std::string OPENML_REST_API_PREFIX = "/api_new/v1/json";

static const std::string TEST_REST_API_HOST = "test.openml.org";
static const unsigned short TEST_REST_API_PORT = 80;
static const std::string TEST_REST_API_PREFIX = "/api/v1/json";


////////////////////////////////////////////////////////////


Connection::Connection()
: m_host(OPENML_REST_API_HOST)
, m_port(OPENML_REST_API_PORT)
, m_prefix(OPENML_REST_API_PREFIX)
{ }

Connection::Connection(std::string const& host, unsigned short port, std::string const& prefix)
: m_host(host)
, m_port(port)
, m_prefix(prefix)
{ }


void Connection::enableTestMode()
{
	// redirect all traffic to the OpenML test server
	m_host = TEST_REST_API_HOST;
	m_port = TEST_REST_API_PORT;
	m_prefix = TEST_REST_API_PREFIX;
}

detail::HttpResponse Connection::getHTTP(std::string const& request, ParamType const& parameters)
{
	SYNCHRONIZE

	detail::HttpResponse response;

	if (! m_socket.connected()) m_socket.connect(m_host, m_port);
	if (! m_socket.connected()) return response;

	std::string url = m_prefix + request;
	bool q = true;
	if (! m_key.empty())
	{
		url += "?api_key=" + m_key;
		q = false;
	}
	for (ParamType::const_iterator it = parameters.begin(); it != parameters.end(); ++it)
	{
		if (q) url += '?'; else url += '&';
		url += detail::urlencode(it->first);
		url += "=";
		url += detail::urlencode(it->second);
		q = false;
	}

	std::string msg = "GET " + url + " HTTP/1.1\r\n"
			"host: " + m_host + "\r\n"
			"accept-encoding: gzip\r\n"
			"\r\n";

	if (! m_socket.writeAll(msg.c_str(), msg.size()))
	{
		m_socket.close();
		return response;
	}
	receiveResponse(response);
	return response;
}

detail::Json Connection::get(std::string const& request, ParamType const& parameters)
{
	detail::HttpResponse response = getHTTP(request, parameters);
	if (response.statusCode() == 0)
	{
		// general communication or protocol violation error
		throw SHARKEXCEPTION("OpenML GET request failed");
	}
	else if (response.statusCode() != 200)
	{
		// request failed, report error
		std::string msg = "OpenML GET request failed with status code " + boost::lexical_cast<std::string>(response.statusCode());
		if (! response.body().empty()) msg += " and error message " + response.body();
		throw SHARKEXCEPTION(msg);
	}
	else
	{
		// request successful, return JSON reply
		detail::Json json;
		json.parse(response.body());
		return json;
	}
}

detail::HttpResponse Connection::postHTTP(std::string const& request, ParamType const& parameters)
{
	SYNCHRONIZE

	detail::HttpResponse response;

	if (! m_socket.connected()) m_socket.connect(m_host, m_port);
	if (! m_socket.connected()) return response;

	std::string msg;

	// check for file uploads
	bool upload = false;
	for (std::size_t i=0; i<parameters.size(); i++)
	{
		if (parameters[i].first.find('|') != std::string::npos) upload = true;
	}

	if (upload)
	{
		// there are file uploads, use multipart/form-data
		std::string url = m_prefix + request;

		std::string boundary = "--------------------------------";
		for (std::size_t i=0; i<boundary.size(); i++) boundary[i] = 48 + (rand() % 10);

		std::string body;
		for (ParamType::const_iterator it = parameters.begin(); it != parameters.end(); ++it)
		{
			std::string name = it->first;
			std::string mime, filename;
			std::size_t pos = it->first.find('|');
			if (pos != std::string::npos)
			{
				std::size_t pos2 = it->first.find('|', pos+1);
				if (pos2 == std::string::npos)
				{
					name = it->first.substr(0, pos);
					mime = it->first.substr(pos+1);
					filename = name;
				}
				else
				{
					name = it->first.substr(0, pos);
					mime = it->first.substr(pos+1, pos2-pos-1);
					filename = it->first.substr(pos2+1);
				}
			}

			body += "--";
			body += boundary;
			body += "\r\nContent-Disposition: form-data; name=\"";
			body += name;
			body += "\"";
			if (! mime.empty())
			{
				body += "; filename=\"" + filename + "\"\r\nContent-Type: ";
				body += mime;
				body += "\r\n";
			}
			body += "\r\n";
			body += it->second;
			body += "\r\n";
		}
		if (! m_key.empty())
		{
			body += "--";
			body += boundary;
			body += "\r\nContent-Disposition: form-data; name=\"api_key\"\r\n\r\n";
			body += m_key;
			body += "\r\n";
		}
		body += "--";
		body += boundary;
		body += "--\r\n";

		msg = "POST " + url + " HTTP/1.1\r\n"
				"host: " + m_host + "\r\n"
				"accept-encoding: gzip\r\n"
				"content-length: " + boost::lexical_cast<std::string>(body.size()) + "\r\n"
				"content-type: multipart/form-data; boundary=" + boundary + "\r\n"
				"\r\n"
				+ body;
	}
	else
	{
		// there are no file uploads, use application/x-www-form-urlencoded
		std::string url = m_prefix + request;

		std::string body;
		for (std::size_t i=0; i<parameters.size(); i++)
		{
			body += detail::urlencode(parameters[i].first);
			body += "=";
			body += detail::urlencode(parameters[i].second);
			body += "&";
		}
		body += "api_key=" + m_key;

		msg = "POST " + url + " HTTP/1.1\r\n"
				"host: " + m_host + "\r\n"
				"content-length: " + boost::lexical_cast<std::string>(body.size()) + "\r\n"
				"content-type: application/x-www-form-urlencoded\r\n"
				"\r\n"
				+ body;
	}

	if (! m_socket.writeAll(msg.c_str(), msg.size()))
	{
		m_socket.close();
		return response;
	}
	receiveResponse(response);
	return response;
};

detail::Json Connection::post(std::string const& request, ParamType const& parameters)
{
	detail::HttpResponse response = postHTTP(request, parameters);
	if (response.statusCode() == 0)
	{
		// general communication or protocol violation error
		throw SHARKEXCEPTION("OpenML POST request failed");
	}
	else if (response.statusCode() != 200)
	{
		// request failed, report error
		std::string msg = "OpenML POST request failed; " + boost::lexical_cast<std::string>(response.statusCode()) + " " + response.returnPhrase();
		if (! response.body().empty()) msg += "; " + response.body();
		throw SHARKEXCEPTION(msg);
	}
	else
	{
		// request successful, return JSON reply
		detail::Json json;
		json.parse(response.body());
		return json;
	}
}

std::size_t Connection::read()
{
	char buffer[4096];
	std::size_t n = m_socket.read(buffer, 4096);
	m_readbuffer.append(buffer, n);
	return n;
}

bool Connection::receiveResponse(detail::HttpResponse& response)
{
	// clear the response object
	response.m_statusCode = 0;
	response.m_returnPhrase.clear();
	response.m_header.clear();
	response.m_body.clear();

	// read header lines
	std::vector<std::string> lines;
	std::size_t pos = 0;
	while (true)
	{
		std::size_t endline;
		while (true)
		{
			endline = m_readbuffer.find("\r\n", pos);
			if (endline != std::string::npos) break;
			if (! read()) { m_socket.close(); return false; }
		}
		std::string line = m_readbuffer.substr(pos, endline - pos);
		pos = endline + 2;
		if (line.empty()) break;
		lines.push_back(line);
	}
	if (lines.empty()) { m_socket.close(); return false; }

	// parse the status line
	std::string const& statusline = lines[0];
	std::size_t space1 = statusline.find(' ');
	if (space1 == std::string::npos) { m_socket.close(); return false; }
	std::size_t space2 = statusline.find(' ', space1 + 1);
	if (space2 == std::string::npos) { m_socket.close(); return false; }
	if (statusline.substr(0, space1) != "HTTP/1.0" && statusline.substr(0, space1) != "HTTP/1.1") { m_socket.close(); return false; }
	response.m_statusCode = boost::lexical_cast<unsigned int>(statusline.substr(space1+1, space2-space1-1));
	response.m_returnPhrase = statusline.substr(space2 + 1);

	// parse header lines
	for (std::size_t i=1; i<lines.size(); i++)
	{
		std::string& line = lines[i];
		std::size_t colon = line.find(':');
		if (colon == std::string::npos) { m_socket.close(); return false; }
		std::size_t value = colon + 1;
		while (line[value] == ' ') value++;
		detail::ASCIItoLowerCase(line);
		response.m_header[line.substr(0, colon)] = line.substr(value);
	}

	// remove already parsed data from the buffer
	m_readbuffer.erase(0, pos);

	// read and parse message body
	std::string contentlength = response.header("content-length");
	std::string transferencoding = response.header("transfer-encoding");
	if (! transferencoding.empty())
	{
		if (transferencoding  == "chunked")
		{
			while (true)
			{
				// read one chunk
				std::size_t endline;
				while (true)
				{
					endline = m_readbuffer.find("\r\n", 0);
					if (endline != std::string::npos) break;
					if (! read()) { m_socket.close(); return false; }
				}
				unsigned long length = std::stoul(m_readbuffer.substr(0, endline), 0, 16);
				if (length == 0) break;
				while (m_readbuffer.size() < endline + length + 2)
				{
					if (! read()) { m_socket.close(); return false; }
				}
				response.m_body += m_readbuffer.substr(endline, length);
				if (m_readbuffer.substr(endline + length, 2) != "\r\n") { m_socket.close(); return false; }
				m_readbuffer.erase(0, endline + length + 2);
			}
			while (m_readbuffer.size() < 2)
			{
				if (! read()) { m_socket.close(); return false; }
			}
			if (m_readbuffer.substr(0, 2) != "\r\n") { m_socket.close(); return false; }
			m_readbuffer.erase(0, 2);
		}
		else { m_socket.close(); return false; }
	}
	else if (! contentlength.empty())
	{
		std::size_t length = boost::lexical_cast<std::size_t>(contentlength);
		while (m_readbuffer.size() < length)
		{
			if (read() == 0) { m_socket.close(); return false; }
		}
		response.m_body = m_readbuffer.substr(0, length);
		m_readbuffer.erase(0, length);
	}

	// recompress content if appropriate
	auto it = response.m_header.find("content-encoding");
	if (it != response.m_header.end())
	{
		if (it->second == "gzip")
		{
			try
			{
				response.m_body = detail::unzip(response.m_body);
			}
			catch (...)
			{
				return false;
			}
		}
	}

	return true;
}


Connection connection;                 ///< \brief The global OpenML connection object.


};  // namespace openML
};  // namespace shark
