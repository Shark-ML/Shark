
#include <shark/Network/Requestparser.h>
#include <shark/Network/Tools.h>

#include <fstream>
#include <sstream>
#include <cstdlib>


// #define error(msg) throw std::string(msg)
#define error(msg) return false


namespace shark {
namespace http {


RequestParser::RequestParser(Request& request)
: m_request(request)
, m_status(esRequestLine)
, m_bodytype(ebNone)
, m_current(0)
, m_bodylength(0)
{ }

RequestParser::~RequestParser()
{ }


void RequestParser::reset()
{
	m_status = esRequestLine;
	m_bodytype = ebNone;
	m_current = 0;
	m_bodylength = 0;
	m_request.m_method.clear();
	m_request.m_resource.clear();
	m_uri.clear();
	m_header.clear();
	m_body.clear();
	m_request.m_parameter.clear();
	m_request.m_filename.clear();
}

bool RequestParser::processData(const char* buffer, std::size_t size, std::size_t& used)
{
	used = 0;
	std::size_t content, length;
	while (true)
	{
		if (m_status == esRequestLine)
		{
			if (! extractLine(buffer, size, content, length)) return true;

			std::string line = std::string(buffer, buffer + content);
			std::size_t pos1 = line.find(' ');
			if (pos1 == std::string::npos)
			{
				error("invalid request line");
			}
			std::size_t pos2 = line.find(' ', pos1 + 1);
			if (pos2 == std::string::npos)
			{
				error("invalid request line");
			}

			m_request.m_method = line.substr(0, pos1);
			if (! decode(line.substr(pos1 + 1, pos2 - pos1 - 1), m_uri, false))
			{
				error("URL encoding error");
			}
			if (m_request.m_method != "GET" && m_request.m_method != "HEAD" && m_request.m_method != "POST")
			{
				error("http method not supported: " + m_request.m_method);
			}
			if (line.substr(pos2 + 1, 5) != "HTTP/")
			{
				error("invalid request line");
			}

			used += length;
			size -= length;
			buffer += length;
			m_status = esHeaderLine;
		}
		else if (m_status == esHeaderLine)
		{
			if (! extractLine(buffer, size, content, length)) return true;

			if (content == 0)
			{
				if (m_request.m_method == "HEAD") m_bodytype = ebNone;
				m_status = esBody;
			}
			else
			{
				std::string line = std::string(buffer, buffer + content);
				std::size_t pos1 = line.find(':');
				if (pos1 == std::string::npos)
				{
					error("invalid header line");
				}
				std::size_t pos2 = pos1 + 1;
				while (isspace(line[pos2])) pos2++;
				std::string tag = line.substr(0, pos1);
				std::string value = line.substr(pos2);
				for (std::size_t i=0; i<tag.size(); i++) if (tag[i] >= 'A' && tag[i] <= 'Z') tag[i] += 32;
				m_header[tag] = value;

				if (tag == "content-length")
				{
					m_bodytype = ebLength;
					m_bodylength = atoi(value.c_str());
				}
				if (tag == "transfer-encoding" && tag == "chunked")
				{
					m_bodytype = ebChunked;
				}
			}

			used += length;
			size -= length;
			buffer += length;
		}
		else if (m_status == esBody)
		{
			if (m_bodytype == ebNone)
			{
				m_status = esInterpret;
			}
			else if (m_bodytype == ebLength)
			{
				if (size < m_bodylength - m_current)
				{
					// read all data
					m_body += std::string(buffer, buffer + size);
					m_current += size;
					used += size;
					size = 0;
					return true;
				}
				else
				{
					// finish the body
					std::size_t l = m_bodylength - m_current;
					m_body += std::string(buffer, buffer + l);
					m_current += l;
					used += l;
					size -= l;
					m_status = esInterpret;
				}
			}
			else if (m_bodytype == ebChunked)
			{
				if (m_bodylength == 0)
				{
					// read a new chunk header
					if (! extractLine(buffer, size, content, length)) return true;
					char* endptr;
					m_bodylength = strtol(buffer, &endptr, 16);
					m_current = 0;
					if (endptr != buffer + content)
					{
						error("chunked encoding broken");
					}
					used += length;
					size -= length;
					buffer += length;
					if (m_bodylength == 0) m_status = esInterpret;
				}
				else
				{
					// read data
					if (size <= m_bodylength - m_current + 2)
					{
						// read all data
						std::size_t l = std::min(size, m_bodylength - m_current - 1);
						m_body += std::string(buffer, buffer + l);
						used += l;
						return true;
					}
					else
					{
						// finish the chunk
						std::size_t l = m_bodylength - m_current;
						m_body += std::string(buffer, buffer + l);
						if (buffer[l] != '\r' || buffer[l+1] != '\n')
						{
							error("chunked encoding broken");
						}
						l += 2;
						used += l;
						size -= l;
						buffer += l;
					}
				}
			}
			else
			{
				error("[RequestParser::processData] internal error");
			}
		}
		else if (m_status == esInterpret)
		{
			// extract parameters from the URI and potentially from POST data
			std::size_t q = m_uri.find("?");
			if (q == std::string::npos)
			{
				m_request.m_resource = m_uri;
			}
			else
			{
				m_request.m_resource = m_uri.substr(0, q);
				extractParams(m_uri.substr(q + 1));
			}
			if (! m_body.empty())
			{
				std::string ct = m_header["content-type"];
				if (ct.substr(0, 19) == "multipart/form-data")
				{
					if (! handleMultipartFormData())
					{
						error("multipart data broken");
					}
				}
				else extractParams(m_body);
			}
			m_status = esReady;
			break;
		}
		else
		{
			break;
		}
	}
	return true;
}

bool RequestParser::extractLine(const char* buffer, std::size_t size, std::size_t& content, std::size_t& length)
{
	length = 0;
	content = 0;
	for (std::size_t i=0; i<size; i++)
	{
		if (buffer[i] == '\r')
		{
			i++;
			if (i >= size) return false;
			if (buffer[i] == '\n')
			{
				content = i - 1;
				length = i + 1;
				return true;
			}
		}
		else if (buffer[i] == '\n')
		{
			content = i;
			length = i + 1;
			return true;
		}
	}
	return false;
}

int RequestParser::tohex(char c)
{
	if (c >= '0' && c <= '9') return c - '0';
	if (c >= 'A' && c <= 'F') return c - 'A' + 10;
	if (c >= 'a' && c <= 'f') return c - 'A' + 10;
	return -1;
}

bool RequestParser::decode(std::string str, std::string& ret, bool decodeplus)
{
	ret.clear();
	for (std::size_t i=0; i<str.size(); i++)
	{
		if (decodeplus && str[i] == '+') ret += ' ';
		else if (str[i] == '%')
		{
			char c;
			int hex;
			i++; hex = tohex(str[i]);
			if (hex < 0) return false;
			c = 16 * hex;
			i++; hex = tohex(str[i]);
			if (hex < 0) return false;
			c += hex;
			ret += c;
		}
		else ret += str[i];
	}
	return true;
}

bool RequestParser::extractParam(std::string str)
{
	std::size_t eq = str.find('=');
	if (eq == std::string::npos) return false;
	std::string tag, value;
	if (! decode(str.substr(0, eq), tag)) return false;
	if (! decode(str.substr(eq+1), value)) return false;
	m_request.m_parameter[tag] = value;
	return true;
}

bool RequestParser::extractParams(std::string str)
{
	std::size_t start = 0;
	while (true)
	{
		std::size_t amp = str.find('&', start);
		if (amp == std::string::npos) return extractParam(str.substr(start));
		else
		{
			if (! extractParam(str.substr(start, amp - start))) return false;
			start = amp + 1;
		}
	}
}

bool RequestParser::handleMultipartFormData()
{
	std::size_t content, length;
	const char* body = m_body.c_str();
	std::size_t size = m_body.size();
	std::size_t used = 0;

	// read the separator
	if (! extractLine(body + used, size, content, length)) return false;
	std::string separator = "\r\n" + m_body.substr(0, content);
	used += length;

	// read parts
	while (true)
	{
		// read additional header fields and extract the file name
		std::string name;
		while (true)
		{
			if (! extractLine(body + used, size, content, length)) return false;
			std::string line = m_body.substr(used, content);
			used += length;
			if (content == 0) break;

			std::size_t colon = line.find(':');
			if (colon == std::string::npos) return false;
			for (std::size_t i=0; i<colon; i++) if (line[i] >= 'A' && line[i] <= 'Z') line[i] += 32;
			if (line.substr(0, colon) == "content-disposition")
			{
				std::size_t start = line.find("name=\"");
				if (start == std::string::npos) return false;
				start += 6;
				std::size_t end = line.find('\"', start);
				if (end == std::string::npos) return false;
				name = line.substr(start, end - start);

				start = line.find("filename=\"", end);
				if (start != std::string::npos)
				{
					start += 10;
					std::size_t end = line.find('\"', start);
					if (end == std::string::npos) return false;
					std::string filename = line.substr(start, end - start);
					m_request.m_filename[name] = filename;
				}
			}
		}
		if (name == "") return false;

		// find the separator
		std::size_t end = m_body.find(separator, used);
		if (end == std::string::npos) return false;
		m_request.m_parameter[name] = m_body.substr(used, end - used);
		used = end + separator.size();
		if (m_body[used] == '-' && m_body[used+1] == '-')
		{
			used += 2;
			if (m_body[used] != '\r' || m_body[used+1] != '\n') return false;
			used += 2;
			if (used != m_body.size()) return false;
			break;
		}
		else
		{
			if (m_body[used] != '\r' || m_body[used+1] != '\n') return false;
			used += 2;
		}
	}

	return true;
}


}}
