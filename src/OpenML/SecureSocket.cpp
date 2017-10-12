//===========================================================================
/*!
 * 
 *
 * \brief       Socket with TLS encryption, used internally by the OpenML module.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
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

#include <shark/OpenML/detail/SecureSocket.h>
#include <shark/Core/Exception.h>

#include <openssl/opensslv.h>
#include <openssl/ssl.h>
#include <openssl/rsa.h>
#include <openssl/x509.h>
#include <openssl/evp.h>
#include <openssl/bio.h>
#include <openssl/err.h>

#include <boost/lexical_cast.hpp>

#include <thread>
#include <chrono>
#include <csignal>

#ifdef _WIN32
	#include <winsock2.h>
	typedef int socklen_t;

	#pragma comment(lib, "ws2_32.lib")
#else
	#include <netinet/in.h>
#endif


namespace shark {
namespace openML {
namespace detail {


////////////////////////////////////////////////////////////


//
// Polymorphic super class of the different end point types with
// virtual function interface for delegation of specialized jobs.
//
struct SecureSocket::PIMPL
{
	PIMPL()
	{
		if (! initialized)
		{
			// ignore annoying pipe signal
			struct sigaction sa;
			sa.sa_handler = SIG_IGN;
			sigemptyset(&sa.sa_mask);
			sa.sa_flags = 0;
			sigaction(SIGPIPE, &sa, 0);

			// initialize and setup openssl
			SSL_library_init();
			OpenSSL_add_all_algorithms();
			ERR_load_BIO_strings();
			ERR_load_crypto_strings();
			SSL_load_error_strings();

			initialized = true;
		}
	}

	virtual ~PIMPL()
	{ }


	virtual bool connect(std::string const& remoteHost, std::uint16_t remotePort) = 0;

	virtual bool connected() const = 0;

	virtual bool peer(std::vector<std::uint8_t>& ip, std::uint16_t& port) = 0;

	virtual void close() = 0;

	virtual bool hasData(unsigned int timeoutSeconds = 0, unsigned int timeoutUSeconds = 0) = 0;

	virtual std::size_t read(char* buffer, std::size_t buffersize)
	{ throw SHARKEXCEPTION("[shark::openML::SecureSocket::read] internal error"); }

	virtual std::size_t write(const char* buffer, std::size_t buffersize)
	{ throw SHARKEXCEPTION("[shark::openML::SecureSocket::write] internal error"); }

	virtual bool writeAll(const char* buffer, std::size_t buffersize)
	{
		size_t pos = 0;
		while (pos < buffersize)
		{
			size_t n = write(buffer + pos, buffersize - pos);
			if (n == 0) return false;
			pos += n;
		}
		return true;
	}

	virtual int handle() const = 0;

	static bool initialized;
};


// initialize static data
bool SecureSocket::PIMPL::initialized = false;


struct ClientEndpoint : public SecureSocket::PIMPL
{
	SSL_CTX* ctx;
	BIO* web;
	SSL* ssl;

	ClientEndpoint()
	: ctx(nullptr)
	, web(nullptr)
	, ssl(nullptr)
	{ }

	~ClientEndpoint()
	{ close(); }


	bool connect(std::string const& remoteHost, std::uint16_t remotePort) override
	{
		if (connected()) close();

		int res;

		const SSL_METHOD* method = SSLv23_method();
		if (! method) { close(); return false; }

		ctx = SSL_CTX_new(method);
		if (! ctx) { close(); return false; }

		SSL_CTX_set_options(ctx, SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 | SSL_OP_NO_COMPRESSION);

		web = BIO_new_ssl_connect(ctx);
		if (! web) { close(); return false; }

		std::stringstream ss;
		ss << remoteHost << ":" << remotePort;
		res = BIO_set_conn_hostname(web, ss.str().c_str());
		if (res != 1) { close(); return false; }

		BIO_get_ssl(web, &ssl);
		if (! ssl) { close(); return false; }

		SSL_set_mode(ssl, SSL_MODE_AUTO_RETRY);

		res = SSL_set_tlsext_host_name(ssl, remoteHost.c_str());
		if (res != 1) { close(); return false; }

		res = BIO_do_connect(web);
		if (res != 1) { close(); return false; }

		res = BIO_do_handshake(web);
		if (res != 1) { close(); return false; }

		// verify that a server certificate was presented during the negotiation
		X509* cert = SSL_get_peer_certificate(ssl);
		if (cert) X509_free(cert);
		if (! cert) { close(); return false; }

		// Step 2: verify the result of chain verification (according to RFC 4158)
		// skip this, since we won't maintain a list of trusted root authorities

		// report success
		return true;
	}

	bool connected() const override
	{ return (ctx && web && ssl); }

	bool peer(std::vector<std::uint8_t>& ip, std::uint16_t& port) override
	{
		int handle;
		if (BIO_get_fd(web, &handle) == -1) return false;
		if (handle == 0) return false;

		sockaddr_in s;
		socklen_t len = sizeof(sockaddr_in);
		int err = getpeername(handle, (sockaddr*)(&s), &len);
		if (err == 0)
		{
			std::uint32_t ip4 = ntohl(s.sin_addr.s_addr);
			ip.resize(4);
			for (std::size_t i=0; i<4; i++) ip[3-i] = (ip4 >> (8*i)) & 0xff;
			port = ntohs(s.sin_port);
			return true;
		}
		else
		{
			close();
			return false;
		}
	}

	void close() override
	{
		if (web) BIO_free_all(web);
		if (ctx) SSL_CTX_free(ctx);
		ctx = nullptr;
		web = nullptr;
		ssl = nullptr;
	}

	bool hasData(unsigned int timeoutSeconds = 0, unsigned int timeoutUSeconds = 0) override
	{
		int handle;
		if (BIO_get_fd(web, &handle) == -1) return false;
		if (handle == 0) return false;

		timeval timeout;
		timeout.tv_sec = timeoutSeconds;
		timeout.tv_usec = timeoutUSeconds;

		fd_set read_socks;
		FD_ZERO(&read_socks);
		FD_SET(handle, &read_socks);

		int count = select(handle + 1, &read_socks, nullptr, nullptr, &timeout);
		bool ret = (count == 1);

		FD_ZERO(&read_socks);
		if (count == -1)
		{
			close();
			return false;
		}

		return ret;
	}

	std::size_t read(char* buffer, std::size_t buffersize) override
	{
		while (true)
		{
			int result = BIO_read(web, buffer, buffersize);
			// for low-level debugging:
			// if (result > 0) { printf("\n[read=%d]\n", result); for (int i=0; i<result; i++) printf("%c", buffer[i]); };
			if (result > 0) return result;
			if (! BIO_should_retry(web)) { close(); return 0; }
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}

	std::size_t write(const char* buffer, std::size_t buffersize) override
	{
		while (true)
		{
			int result = BIO_write(web, buffer, buffersize);
			// for low-level debugging:
			// if (result > 0) { printf("\n[write=%d]\n", result); for (int i=0; i<result; i++) printf("%c", buffer[i]); };
			if (result > 0) return result;
			if (! BIO_should_retry(web)) { close(); return 0; }
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}

	int handle() const override
	{
		int handle = 0;
		if (BIO_get_fd(web, &handle) == -1) return 0;
		return handle;
	}
};


////////////////////////////////////////////////////////////


SecureSocket::SecureSocket()
: pimpl(new ClientEndpoint())
{ }

SecureSocket::~SecureSocket()
{ pimpl->close(); }


bool SecureSocket::connect(std::string const& remoteHost, std::uint16_t remotePort)
{ return pimpl->connect(remoteHost, remotePort); }

bool SecureSocket::connected() const
{ return pimpl->connected(); }

bool SecureSocket::peer(std::vector<std::uint8_t>& ip, std::uint16_t& port)
{ return pimpl->peer(ip, port); }

void SecureSocket::close()
{ pimpl->close(); }

bool SecureSocket::hasData(unsigned int timeoutSeconds, unsigned int timeoutUSeconds)
{ return pimpl->hasData(timeoutSeconds, timeoutUSeconds); }

std::size_t SecureSocket::read(char* buffer, std::size_t buffersize)
{ return pimpl->read(buffer, buffersize); }

std::size_t SecureSocket::write(const char* buffer, std::size_t buffersize)
{ return pimpl->write(buffer, buffersize); }

bool SecureSocket::writeAll(const char* buffer, std::size_t buffersize)
{ return pimpl->writeAll(buffer, buffersize); }

int SecureSocket::handle() const
{ return pimpl->handle(); }


};  // namespace detail
};  // namespace openML
};  // namespace shark
