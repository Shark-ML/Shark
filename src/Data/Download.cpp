//===========================================================================
/*!
 * 
 *
 * \brief       implementation of data downloading
 * 
 * 
 *
 * \author      T.Glasmachers, O.Krause
 * \date        2017-2018
 *
 *
 * \par Copyright 1995-2018 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <https://shark-ml.github.io/Shark/>
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

#include <string>
#include <chrono>
#include <thread>
#include <map>
#include <tuple>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cctype>

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
	#include <netinet/in.h>
	#include <unistd.h>
	#ifndef MSG_NOSIGNAL
		#define MSG_NOSIGNAL SO_NOSIGPIPE
	#endif
#endif

#include <openssl/opensslv.h>
#include <openssl/ssl.h>
#include <openssl/rsa.h>
#include <openssl/x509.h>
#include <openssl/evp.h>
#include <openssl/bio.h>
#include <openssl/err.h>

#define SHARK_COMPILE_DLL
#include <shark/Data/Download.h>


namespace {

/// \brief TCP/IP socket abstraction.
///
/// This socket class encapsulates the most basic functionality of
/// POSIX TCP/IP sockets. It is designed to act as a client, not as a
/// server of a m_web service. Its functionality is somewhat tailored to
/// what's needed for an HTTP or HTTPS download.
class Socket
{
public:
#ifdef _WIN32
	typedef SOCKET SocketType;
#else
	typedef int SocketType;
#endif

	/// \brief Connect to a remote host.
	Socket(bool secure, std::string const& remoteHost, unsigned short remotePort = 0)
	: m_secure(secure)
	, m_handle(0)
	, m_ctx(nullptr)
	, m_web(nullptr)
	, m_ssl(nullptr)
	{
		if (remotePort == 0) remotePort = secure ? 443 : 80;

		if (m_secure)
		{
			if (! m_ssl_initialized)
			{
				SSL_library_init();
				OpenSSL_add_all_algorithms();
				ERR_load_crypto_strings();
				SSL_load_error_strings();
				m_ssl_initialized = true;
			}

			int res;

			const SSL_METHOD* method = SSLv23_method();
			if (! method) { close(); return; }

			m_ctx = SSL_CTX_new(method);
			if (! m_ctx) { close(); return; }

			SSL_CTX_set_options(m_ctx, SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 | SSL_OP_NO_COMPRESSION);

			m_web = BIO_new_ssl_connect(m_ctx);
			if (! m_web) { close(); return; }

			BIO_set_nbio(m_web, 1);   // enable non-blocking mode

			std::string s = remoteHost + ":" + std::to_string(remotePort);
			res = BIO_set_conn_hostname(m_web, s.c_str());
			if (res != 1) { close(); return; }

			BIO_get_ssl(m_web, &m_ssl);
			if (! m_ssl) { close(); return; }

			SSL_set_mode(m_ssl, SSL_MODE_AUTO_RETRY);

			res = SSL_set_tlsext_host_name(m_ssl, remoteHost.c_str());
			if (res != 1) { close(); return; }

			while (true)
			{
				res = BIO_do_connect(m_web);
				if (res <= 0)
				{
					if (BIO_should_retry(m_web)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
					else { close(); return; }
				}
				else break;
			}

			res = BIO_do_handshake(m_web);
			if (res != 1) { close(); return; }

			// Step 1: verify a server certificate was presented during the negotiation
			X509* cert = SSL_get_peer_certificate(m_ssl);
			if (cert) X509_free(cert);
			if (! cert) { close(); return; }
		}
		else
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
	}

	/// \brief Close the connection
	~Socket()
	{ close(); }


	/// \brief Check whether the connection is fine.
	inline bool connected() const
	{
		return m_secure ? (m_ctx && m_web && m_ssl) : (m_handle);
	}

	/// \brief Close the connection (makes connected() return false).
	void close()
	{
		if (m_secure)
		{
			if (m_ctx) SSL_CTX_free(m_ctx);
			if (m_web) BIO_free_all(m_web);
			m_ctx = nullptr;
			m_web = nullptr;
			m_ssl = nullptr;
		}
		else
		{
#ifdef _WIN32
			if (m_handle) ::closesocket(m_handle);
#else
			if (m_handle) ::close(m_handle);
#endif
			m_handle = 0;
		}
	}

	/// \brief Read data from the socket (blocking).
	///
	/// The operation reads at most #buffersize bytes from the socket
	/// into the buffer. The number of bytes read is returned. A return
	/// value of zero indicates an error, e.g., that the remote socket
	/// was closed.
	std::size_t read(char* buffer, std::size_t buffersize)
	{
		if (m_secure)
		{
			int result = BIO_read(m_web, buffer, buffersize);
			return result;
//			if (result > 0) return result;
//			if (! BIO_should_retry(m_web)) { close(); return 0; }
		}
		else
		{
			int ret = recv(m_handle, buffer, buffersize, 0);
			if (ret <= 0)
			{
				ret = 0;
				close();
			}
			return ret;
		}
	}

	/// \brief Read a CR-LF terminated line from the socket.
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
		if (m_secure)
		{
			int result = BIO_write(m_web, buffer, buffersize);
			return result;
//			if (result > 0) return result;
//			if (! BIO_should_retry(m_web)) { close(); return 0; }
		}
		else
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
	bool m_secure;                     ///< is this a secure (https) socket?

	SocketType m_handle;               ///< POSIX socket handle

	SSL_CTX* m_ctx;                    ///< openssl data structure
	BIO* m_web;                        ///< openssl data structure
	SSL* m_ssl;                        ///< openssl data structure

	static bool m_ssl_initialized;     ///< is the openssl library initialized?
};

bool Socket::m_ssl_initialized = false;

} // namespace <anonymous>


std::tuple<bool, std::string, std::string> shark::splitUrl(std::string const & url)
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
	return std::make_tuple(start == 8, domain, resource);
}


std::string shark::download(std::string const& url, unsigned short port) {
	// split the URL into domain and resource
	bool https;
	std::string domain, resource;
	std::tie(https, domain, resource) = splitUrl(url);

	// open a TCP/IP socket connection
	Socket socket(https, domain, port);
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