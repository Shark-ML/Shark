
#include <shark/Network/Connection.h>
#include <shark/Network/Tools.h>

#include <sstream>
#include <algorithm>
#include <cstring>
#include <ctime>


#define PACKAGE_SIZE               1492     // safe TCP/IP package size
#define MAX_MEMORY_OVERHEAD     1048576     // don't accumulate more than 1 MB of wasted memory per socket


namespace shark {
namespace http {


Connection::Connection(Socket* socket)
: m_socket(socket)
, m_parser(m_request)
, m_readUsed(0)
, m_sendOffset(0)
{ }

Connection::~Connection()
{
	delete m_socket;
}


bool Connection::processRead()
{
	std::size_t n = m_socket->read(m_readBuffer + m_readUsed, 4096 - m_readUsed);
	if (n == 0)
	{
		m_socket->close();
		return false;
	}

	m_readUsed += n;

	// parse data
	std::size_t parsed;
	if (! m_parser.processData(m_readBuffer, m_readUsed, parsed))
	{
		m_socket->close();
		return false;
	}

	// update the buffer
	if (parsed > 0 && parsed < m_readUsed) memcpy(m_readBuffer, m_readBuffer + parsed, m_readUsed - parsed);
	m_readUsed -= parsed;

	return true;
}

bool Connection::processWrite()
{
	std::size_t sz = m_sendBuffer.size();
	std::size_t diff = sz - m_sendOffset;
	if (diff == 0)
	{
		if (sz != 0)
		{
			m_sendBuffer.clear();
			m_sendOffset = 0;
		}
		return true;
	}
	if (diff <= PACKAGE_SIZE)
	{
		if (m_socket->write(m_sendBuffer.c_str() + m_sendOffset, diff) < diff)
		{
			m_socket->close();
			return false;
		}
		else
		{
			m_sendBuffer.clear();
			m_sendOffset = 0;
			return true;
		}
	}
	else
	{
		if (m_socket->write(m_sendBuffer.c_str() + m_sendOffset, PACKAGE_SIZE) < PACKAGE_SIZE)
		{
			m_socket->close();
			return false;
		}
		else
		{
			m_sendOffset += PACKAGE_SIZE;
			return true;
		}
	}
}

void Connection::sendRawData(std::string content)
{
	if (m_sendOffset >= MAX_MEMORY_OVERHEAD)
	{
		m_sendBuffer.erase(0, m_sendOffset);
		m_sendOffset = 0;
	}
	m_sendBuffer += content;
}

void Connection::sendDocument(std::string const& content, std::string mime, unsigned int expires, std::string additionalHeaders)
{
	std::string mimeHeader, cacheHeaders;
	if (mime.empty()) mime = "text/html";      // just guessing!
	mimeHeader = "content-type: " + mime + "\r\n";
	if (expires == 0) cacheHeaders = "cache-control: no-cache\r\n";

	char timestring[100];
	std::time_t t1 = time(NULL);
	struct tm tt1 = *gmtime(&t1);
	strftime(timestring, 100, "%H:%M:%S %d.%m.%y", &tt1);
	cacheHeaders += "date: " + std::string(timestring) + "\r\n";

	tt1.tm_sec += expires;
	std::time_t t2 = mktime(&tt1);
	struct tm tt2 = *gmtime(&t2);
	strftime(timestring, 100, "%H:%M:%S %d.%m.%y", &tt2);
	cacheHeaders += "expires: " + std::string(timestring) + "\r\n";

	std::stringstream ss;
	ss << "HTTP/1.1 200 OK\r\ncontent-length: " << content.size() << "\r\n"
			<< cacheHeaders
			<< mimeHeader
			<< additionalHeaders << "\r\n" << content;
	sendRawData(ss.str());
}

void Connection::sendError(unsigned int code, std::string msg)
{
	std::stringstream ss; ss << code;
	std::string content = 
		"<!DOCTYPE html>\r\n"
		"<html>\r\n"
		"<head><title>" + ss.str() + " " + msg + "</title></head>\r\n"
		"<body><h1>" + ss.str() + " " + msg + "</h1></body>\r\n"
		"</html>\r\n";
	std::string s = "HTTP/1.1 " + ss.str() + " " + msg + "\r\n";
	std::stringstream ss2; ss2 << content.size();
	s += "CONTENT-TYPE: text/html; charset=iso-8859-1\r\n"
		"CONTENT-LENGTH: " + ss2.str() + "\r\n\r\n" + content;
	sendRawData(s);
}

void Connection::close()
{
	m_socket->close();
}


}}
