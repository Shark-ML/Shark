/*!
 * \brief       Implementation of the Cross Validation methods
 * 
 * \author      O. Krause
 * \date        2015
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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
#ifndef SHARK_DATA_IMPL_DOWNLOADER_HPP
#define SHARK_DATA_IMPL_DOWNLOADER_HPP

#include <string>
#include <map>
#include <stdexcept>
#include <cstring>
// #include <cerrno>
// #include <csignal>

#ifdef _WIN32
	#include <winsock2.h>
	// typedef int socklen_t;

	#ifndef MSG_NOSIGNAL
		#define MSG_NOSIGNAL 0
	#endif

	#pragma comment(lib, "ws2_32.lib")
#else
	#include <netinet/tcp.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
	#include <sys/socket.h>
	#include <sys/time.h>
	#include <netdb.h>
	#include <unistd.h>
	#ifndef MSG_NOSIGNAL
		#define MSG_NOSIGNAL SO_NOSIGPIPE
	#endif
#endif

namespace shark {

namespace detail {

/// \brief Simple TCP/IP socket abstraction.
class Socket
{
public:
	/// \brief Connect to a remote host.
	Socket(std::string const& remoteHost, unsigned short remotePort)
	: m_handle(0)
	{
#ifdef _WIN32
		// initialize Windows network library
		WSADATA info;
		int result = WSAStartup(2 /* API version 2.0 */, &info);
#endif

		m_handle = ::socket(AF_INET, SOCK_STREAM, 0);
		if (m_handle <= 0) return;

		hostent *host = gethostbyname(remoteHost.c_str());
		if (! host)
		{
			close();
			return;
		}

		sockaddr_in addr;
		addr.sin_family = AF_INET;
		addr.sin_port = htons(remotePort);
		addr.sin_addr.s_addr = ((in_addr *)host->h_addr)->s_addr;
		memset(&(addr.sin_zero), 0, 8);

		int ret = ::connect(m_handle, (sockaddr*) & addr, sizeof(sockaddr));
		if (ret < 0) close();
	}

	/// \brief Close the connection
	~Socket()
	{ close(); }


	/// \brief Check whether the connection is fine.
	inline bool connected() const
	{ return (m_handle > 0); }

	/// \brief Close the connection (makes connected() return false).
	void close()
	{
#ifdef _WIN32
		if (m_handle > 0) ::closesocket(m_handle);
#else
		if (m_handle > 0) ::close(m_handle);
#endif
		m_handle = 0;
	}

	/// \brief Read data from the socket (blocking).
	///
	/// The operation reads at must #buffersize bytes from the socket
	/// into the buffer. The number of bytes read is returned. A return
	/// value of zero indicates an error, e.g., that the remote socket
	/// was closed.
	std::size_t read(char* buffer, std::size_t buffersize)
	{
		int ret = recv(m_handle, buffer, buffersize, 0);
		if (ret <= 0)
		{
			ret = 0;
			close();
		}
		return ret;
	}

	/// \brief Write data to the socket.
	///
	/// The operation may write only a part of the buffer, the
	/// number of bytes written is returned.
	std::size_t write(const char* buffer, std::size_t buffersize)
	{
		if (buffersize == 0) return 0;
		int ret = send(m_handle, buffer, buffersize, MSG_NOSIGNAL);
		if (ret <= 0)
		{
			ret = 0;
			close();
		}
		return ret;
	}

	/// \brief Write data to the socket.
	///
	/// In contrast to write, the writeAll function writes the
	/// whole buffer to the socket. The operation is blocking.
	bool writeAll(const char* buffer, std::size_t buffersize)
	{
		std::size_t pos = 0;
		while (pos < buffersize)
		{
			std::size_t n = write(buffer + pos, buffersize - pos);
			if (n == 0) return false;
			pos += n;
		}
		return true;
	}

private:
	int m_handle;                      ///< POSIX socket handle
};

/// \brief Read a line terminated by CR-LF from the socket.
std::string readLineFromSocket(Socket& socket)
{
	std::string ret;
	char c;
	while (true)
	{
		socket.read(&c, 1);
		if (c == '\r')
		{
			std::size_t r = socket.read(&c, 1);
			if (r == 0) throw std::runtime_error("[download] socket error");
			if (c != '\n') throw std::runtime_error("[download] http protocol violation");
			return ret;
		}
		else ret += c;
	}
}

/// \brief Read a chunk of pre-specified size from the socket.
std::string readChunkFromSocket(Socket& socket, std::size_t length)
{
	std::string ret(length, ' ');
	if (length == 0) return ret;
	char* p = &ret[0];
	while (length > 0)
	{
		std::size_t n = length; if (n > 4096) n = 4096;
		std::size_t r = socket.read(p, length);
		if (r == 0) throw std::runtime_error("[download] socket error");
		p += r;
		length -= r;
	}
	return ret;
}

} // namespace detail

/// \brief Download a document with the HTTP protocol.
///
/// \param  domain    fomain name or IP address, for example "www.shark-ml.org"
/// \param  resource  resource identifier relative to the domain, for example "/index.html"
/// \param  port      TCP/IP port, defaults to 80
///
/// The function returns the HTTP request body. In case of success this
/// is the requested document. In case of an error the function throws
/// an exception. Note that the class does not perform standard actions
/// of browsers, e.g., execute javascript or even follow http redirects.
/// All HTTP response status codes other than 200 are interpreted as
/// failure to download the document and trigger an exception.
std::string download(std::string const& domain, std::string const& resource, unsigned short port = 80)
{
	detail::Socket socket(domain, port);
	std::string request = "GET " + resource + " HTTP/1.1\r\nhost: " + domain + "\r\n\r\n";
	socket.writeAll(request.c_str(), request.size());

	// http reply data
	std::map<std::string, std::string> headers;
	std::string body;

	// parse http reply line
	std::string reply = detail::readLineFromSocket(socket);
	if (reply.size() < 12) throw std::runtime_error("[download] http protocol violation");
	if (reply.substr(0, 9) != "HTTP/1.0 " && reply.substr(0, 9) != "HTTP/1.1 ") throw std::runtime_error("[download] http protocol violation");
	if (reply.substr(9, 3) != "200") throw std::runtime_error("[download] failed with HTTP status code " + reply.substr(9, 3));

	// parse http headers
	while (true)
	{
		std::string h = detail::readLineFromSocket(socket);
		if (h.empty()) break;
		std::size_t colon = h.find(":");
		if (colon == std::string::npos) throw std::runtime_error("[download] http protocol violation");
		std::string tag = h.substr(0, colon);
		// convert plain ASCII to lower case
		for (std::size_t i=0; i<tag.size(); i++) if (tag[i] >= 65 && tag[i] <= 90) tag[i] += 32;
		std::string value = h.substr(colon + 1);
		while (! value.empty() && value[0] == ' ') value.erase(0, 1);
		while (! value.empty() && value[value.size() - 1] == ' ') value.erase(value.size() - 1);
		// convert plain ASCII to lower case
		for (std::size_t i=0; i<value.size(); i++) if (value[i] >= 65 && value[i] <= 90) value[i] += 32;
		headers[tag] = value;
	}

	// receive http body
	std::string len = headers["content-length"];
	if (! len.empty())
	{
		std::size_t length = strtol(len.c_str(), NULL, 10);
		body = detail::readChunkFromSocket(socket, length);
	}
	else
	{
		if (headers["transfer-encoding"] != "chunked") throw std::runtime_error("[download] transfer encoding not supported");
		while (true)
		{
			std::string len = detail::readLineFromSocket(socket);
			std::size_t length = strtol(len.c_str(), NULL, 16);
			body += detail::readChunkFromSocket(socket, length);
			std::string x = readLineFromSocket(socket);
			if (! x.empty()) throw std::runtime_error("[download] http protocol violation");
			if (length == 0) break;
		}
	}

	return body;
}

} // namespace shark
#endif
