/**
 *
 *  \brief Implements a file handler for the shark HTTP server.
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
#ifndef SHARK_NETWORK_HANDLERS_FILE_HANDLER_H
#define SHARK_NETWORK_HANDLERS_FILE_HANDLER_H

#include <shark/Network/HttpServer.h>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

/*
#include <boost/bimap.hpp>
#include <boost/filesystem.hpp>
*/

namespace shark {


/** \brief Implements a file handler for the shark HTTP server.
 *
 *  \par
 *  The file handler interprets any incoming resource as a request for
 *  a file below its document root directory. It handles mine types of
 *  the most prominent file types and sends all other files as binary.
 *  Note that this handler may may pose a serious security risk when
 *  being accessed remotely by an attacker.
 */
class FileHandler : public HttpServer::AbstractRequestHandler {
public:

	/**
	 * \brief C'tor, initializes the document root of this file handler.
	 * 
	 * Requests are served relative to the document root.
	 */
	FileHandler( const boost::filesystem::path & documentRoot )
	: HttpServer::AbstractRequestHandler("SharkAboutHandler")
	, m_documentRoot( documentRoot )
	{
		m_pattern.push_back("*");
	}

	/**
	 * \brief Handles the supplied request and writes to the 
	 * supplied connection.
	 * 
	 * Requests are served relative to the document root.
	 */
	void handleRequest(http::Connection& connection) {
		http::Request& request = connection.request();

	    boost::filesystem::path p = m_documentRoot / request.resource();

	    if( !boost::filesystem::exists( p ) ) {
	    	connection.sendError(404, "file not found");
	    	return;
	    }

	    boost::filesystem::file_status status = boost::filesystem::status( p );
	    switch( status.type() ) {
	    case boost::filesystem::regular_file:
			handleRegularFile( p, request, connection );
			break;
	    case boost::filesystem::directory_file:
			handleDirectory( p, request, connection );
			break;
		default:
	    	connection.sendError(404, "file not found");
			return;
			break;
	    }
	}

	/** 
	 * \brief Generates html directory listings. 
	 */
	void handleDirectory(
			const boost::filesystem::path & p, 
			http::Request& request, 
			http::Connection& connection)
	{
	    std::stringstream ss;
	    ss << "<html>";
	    ss << "<body>";
	    ss << "<ul class=\"listing\">";

	    std::string dir = request.resource();
	    if (dir.empty() || dir[dir.size() - 1] != '/') dir += "/";
	    boost::filesystem::directory_iterator it( p );
	    boost::filesystem::directory_iterator itE;
	    for( ; it != itE; ++it ) {
		boost::format f( "<li><a href=\"%1%\">%2%</a></li>" );
		f = f % 
			( dir + it->path().filename().string() ) %
		    it->path().filename();
		ss << f.str();
	    }

	    ss << "</ul>";
	    ss << "</body>";
	    ss << "</html>";

		connection.sendDocument(ss.str(), "text/html", 600);
	}

	/** 
	 * \brief Sends out the file content over the supplied connection. 
	 */
	void handleRegularFile( 
			const boost::filesystem::path & p, 
			http::Request& request, 
			http::Connection& connection)
	{
		std::string content = http::readFile(p.string());

		std::string resource = request.resource();
		std::string mime;
		if (matchesMask(resource, "*.html"))       mime = "text/html";
		else if (matchesMask(resource, "*.html"))  mime = "text/html";
		else if (matchesMask(resource, "*.xhtml")) mime = "text/html";
		else if (matchesMask(resource, "*.txt"))   mime = "text/plain";
		else if (matchesMask(resource, "*.css"))   mime = "text/css";
		else if (matchesMask(resource, "*.js"))    mime = "text/javascript";
		else if (matchesMask(resource, "*.png"))   mime = "image/png";
		else if (matchesMask(resource, "*.jpg"))   mime = "image/jpeg";
		else if (matchesMask(resource, "*.jpeg"))  mime = "image/jpeg";
		else if (matchesMask(resource, "*.json"))  mime = "application/json";
		else                                       mime = "x-application/octet-stream";

		connection.sendDocument(content, mime, 86400);
	}

	boost::filesystem::path m_documentRoot; ///< Stores the document root relevant to the local file system.
};


}
#endif
