//===========================================================================
/*!
 * 
 *
 * \brief       implementation of data downloading
 * 
 * 
 *
 * \author      T.Glasmachers, O.Krause
 * \date        2017
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
//===========================================================================
#ifdef _WIN32
	#define _WINSOCK_DEPRECATED_NO_WARNINGS
	#include <winsock2.h>

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

#define SHARK_COMPILE_DLL
#include <shark/Data/Download.h>

#include <string>
#include <map>
#include <stdexcept>
#include <cstring>
#include <cctype>

namespace {

/// \brief Simple TCP/IP socket abstraction.
///
/// This socket class encapsulates the most basic functionality of
/// POSIX TCP/IP sockets. It is designed to act as a client, not as a
/// server of a web service. Its functionality is somewhat tailored to
/// what's needed for an HTTP download.
class Socket
{
public:
#ifdef _WIN32
	typedef SOCKET SocketType;
#else
	typedef int SocketType;
#endif

	/// \brief Connect to a remote host.
	Socket(std::string const& remoteHost, unsigned short remotePort)
	: m_handle(0)
	{
#ifdef _WIN32
		// initialize Windows network library
		WSADATA info;
		int result = WSAStartup(2 /* API version 2.0 */, &info);

		m_handle = ::socket(AF_INET, SOCK_STREAM, 0);
		if (m_handle == INVALID_SOCKET)
		{
			m_handle = 0;
			return;
		}
#else
		m_handle = ::socket(AF_INET, SOCK_STREAM, 0);
		if (m_handle <= 0) return;
#endif

		hostent* host = gethostbyname(remoteHost.c_str());
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

		int connectionResult = ::connect(m_handle, (sockaddr*)&addr, sizeof(sockaddr));
		if (connectionResult != 0) close();
	}

	/// \brief Close the connection
	~Socket()
	{ close(); }


	/// \brief Check whether the connection is fine.
	inline bool connected() const
	{ return (m_handle); }

	/// \brief Close the connection (makes connected() return false).
	void close()
	{
#ifdef _WIN32
		if (m_handle) ::closesocket(m_handle);
#else
		if (m_handle) ::close(m_handle);
#endif
		m_handle = 0;
	}

	/// \brief Read data from the socket (blocking).
	///
	/// The operation reads at most #buffersize bytes from the socket
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

	/// \brief Read a CR-LF terminated line by from the socket.
	///
	/// The returned string does not contain the CR-LF code.
	/// An exception is thrown if reading fails or if a CR not
	/// followed by an LF is encountered.
	std::string readLine()
	{
		std::string ret;
		char c;
		while (true)
		{
			if (read(&c, 1) == 0) return ret;
			if (c == '\r')
			{
				if (read(&c, 1) == 0) return ret;
				if (c != '\n') throw std::runtime_error("[Socket::readLine] broken CR-LF");
				return ret;
			}
			else ret += c;
		}
	}

	/// \brief Read a chunk of pre-specified size from the socket.
	///
	/// In contrast to the read function, this function is guaranteed
	/// to return the requested number of bytes. An exception is thrown
	/// if reading fails.
	std::string readChunk(std::size_t size)
	{
		std::string ret(size, ' ');
		if (size == 0) return ret;
		char* p = &ret[0];
		while (size > 0)
		{
			std::size_t r = read(p, size);
			if (r == 0) throw std::runtime_error("[Socket::readChunk] read failed");
			p += r;
			size -= r;
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
	SocketType m_handle;               ///< POSIX socket handle
};

} // namespace <anonymous>


std::pair<std::string, std::string> shark::splitUrl(std::string const & url)
{
	std::size_t start = 0;
	if(url.size() >= 7 && url.substr(0, 7) == "http://")
	{
		start = 7;
	}
	if(url.size() >= 8 && url.substr(0, 8) == "https://")
	{
		start = 8;
	}
	std::size_t slash_idx = url.find('/', start);
	std::string resource;
	if(slash_idx == std::string::npos)
	{
		slash_idx = url.size();
		resource = "/";
	}
	else
	{
		resource = url.substr(slash_idx);
	}
	std::string domain = url.substr(start, slash_idx - start);
	return std::make_pair(domain, resource);
}


std::string shark::download(std::string const& url, unsigned short port){
	// split the URL into domain and resource
	std::string domain, resource;
	std::tie(domain, resource) = splitUrl(url);

	// open a TCP/IP socket connection
	Socket socket(domain, port);
	if(!socket.connected()){
		throw std::runtime_error("[download] can not connect to url");
	}
	std::string request = "GET " + resource + " HTTP/1.1\r\nhost: " + domain + "\r\n\r\n";
	socket.writeAll(request.c_str(), request.size());

	// http reply data
	std::map<std::string, std::string> headers;
	std::string body;

	// parse http reply line
	std::string reply = socket.readLine();
	if (reply.size() < 12) throw std::runtime_error("[download] http protocol violation");
	if (reply.substr(0, 9) != "HTTP/1.0 " && reply.substr(0, 9) != "HTTP/1.1 ") throw std::runtime_error("[download] http protocol violation");
	if (reply.substr(9, 3) != "200") throw std::runtime_error("[download] failed with HTTP status " + reply.substr(9));

	// parse http headers
	while (true)
	{
		std::string h = socket.readLine();
		if (h.empty()) break;
		std::size_t colon = h.find(":");
		if (colon == std::string::npos) throw std::runtime_error("[download] http protocol violation");
		std::string tag = h.substr(0, colon);
		// convert plain ASCII to lower case
		for (std::size_t i=0; i<tag.size(); i++) tag[i] = std::tolower(tag[i]);
		std::string value = h.substr(colon + 1);
		while (! value.empty() && value[0] == ' ') value.erase(0, 1);
		while (! value.empty() && value[value.size() - 1] == ' ') value.erase(value.size() - 1);
		// convert plain ASCII to lower case
		for (std::size_t i=0; i<value.size(); i++) value[i] = std::tolower(value[i]);
		headers[tag] = value;
	}

	// receive http body
	std::string len = headers["content-length"];
	if (! len.empty())
	{
		// a priori known content length
		std::size_t length = strtol(len.c_str(), NULL, 10);
		body = socket.readChunk(length);
	}
	else
	{
		// chunked encoding
		if (headers["transfer-encoding"] != "chunked") throw std::runtime_error("[download] transfer encoding not supported");
		while (true)
		{
			std::string len = socket.readLine();
			std::size_t length = strtol(len.c_str(), NULL, 16);
			body += socket.readChunk(length);
			std::string x = socket.readLine();
			if (! x.empty()) throw std::runtime_error("[download] http protocol violation");
			if (length == 0) break;
		}
	}

	return body;
}