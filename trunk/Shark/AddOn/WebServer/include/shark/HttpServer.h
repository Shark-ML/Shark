/**
 *
 *  \brief Implements an http server extendable with custom request handlers.
 *
 *  \author  T. Voss, T. Glasmachers
 *  \date    2011, 2013
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
#ifndef SHARK_NETWORK_HTTPSERVER_H
#define SHARK_NETWORK_HTTPSERVER_H

#include <shark/Network/Server.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace shark {


/// \brief Implements an http-server with arbitrary handlers.
class HttpServer : public http::Server
{
public:
	/// \brief Destructor.
	~HttpServer()
	{
		for (std::size_t i=0; i<m_handler.size(); i++) delete m_handler[i];
	}


	///
	/// \brief Models a handler for HTTP requests.
	///
	/// \par
	/// An AbstractRequestHandler is an abstraction of a request handler
	/// specialized in certain types of requests. The class models the
	/// type of request handling through its virtual handleRequest
	/// function, and the object models the requests by defining
	/// resource patterns. For example, a handler for html files should
	/// add the pattern "*.html" to its pattern list m_pattern, while
	/// a specialized handler for the single resource "/About" will
	/// register the static pattern "/About". The registration is
	/// usually done by the constructor of a sub class.
	///
	class AbstractRequestHandler {
	public:
		/// \brief Default constructor
		AbstractRequestHandler()
		{ }

		/// \brief Constructor defining the name
		AbstractRequestHandler(std::string name)
		: m_name(name)
		{ }

		/// \brief Virtual d'tor.
		virtual ~AbstractRequestHandler()
		{ }

		/// \brief Accesses the name of the handler.
		std::string const& name() const
		{ return m_name; }

		/// \brief Check whether the handler claims responsibility for handling a resource.
		bool matchesResource(std::string const& resource) const
		{
			for (std::size_t i=0; i<m_pattern.size(); i++) if (matchesMask(resource, m_pattern[i])) return true;
			return false;
		}

		/// \brief shorthand for handleRequest
		void operator() (http::Connection& connection)
		{ handleRequest(connection); }

		/// \brief Called by the server, needs to be implemented by custom request handlers. */
		virtual void handleRequest(http::Connection& connection) = 0;

	protected:
		/// \brief Check whether a string matches a mask with wildcards.
		///
		/// \par
		/// The special characters '*' and '?' in the mask string have
		/// the same meaning as in file system queries. The star matches
		/// any sequence (including the empty one), and the question
		/// mark matches any single character. Currently there is no
		/// escaping mechanism - and it should not really be necessary.
		static bool matchesMask(std::string const& s, std::string const& mask)
		{
			std::size_t starpos = mask.find('*');
			if (starpos == std::string::npos)
			{
				if (s.size() != mask.size()) return false;
				for (std::size_t r=0; r<s.size(); r++)
				{
					if (mask[r] == '?') continue;
					if (mask[r] != s[r]) return false;
				}
				return true;
			}
			else
			{
				if (s.size() < starpos) return false;
				for (std::size_t r=0; r<starpos; r++)
				{
					if (mask[r] == '?') continue;
					if (mask[r] != s[r]) return false;
				}
				std::string m = mask.substr(starpos + 1);
				std::string sub = s.substr(starpos);
				while (true)
				{
					if (matchesMask(sub, m)) return true;
					if (sub.empty()) break;
					sub.erase(0, 1);
				}
				return false;
			}
		}

		std::string m_name;                     ///< name of the request handler
		std::vector<std::string> m_pattern;     ///< resource patterns to handle
	};

	///
	/// \brief Register a new resource handler.
	///
	/// \par
	/// The function takes over the ownership of the object.
	/// The handler will be deleted by the server destructor.
	///
	/// \par
	/// The handler is invoked based on the pattern it defines.
	/// The first handler that matches a request gets to handle it.
	/// This is why the most generic handler should be registered last
	/// (often a FileHandler that tries to deliver a file corresponding
	/// to the resource string).
	///
	void registerHandler(AbstractRequestHandler* handler)
	{ m_handler.push_back(handler); }

	/// \brief Delegate a server request to a handler.
	void handleRequest(http::Connection& connection)
	{
		for (std::size_t i=0; i<m_handler.size(); i++)
		{
			if (m_handler[i]->matchesResource(connection.request().resource()))
			{
				m_handler[i]->handleRequest(connection);
				return;
			}
		}

		// no handler found -- send an error reply
		connection.sendError(404, "resource not found");
	}

protected:
	std::vector<AbstractRequestHandler*> m_handler;   ///< list of request handlers
};


}
#endif
