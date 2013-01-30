/**
 *
 *  \brief Implements a file handler for the shark HTTP server.
 *
 *  \author  T. Voss
 *  \date    2011
 *
 *  \par Copyright (c) 2007-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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

#include <boost/bimap.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/noncopyable.hpp>

namespace shark {

    /** \brief Implements a file handler for the shark HTTP server. */
    class FileHandler : public HttpServer::AbstractRequestHandler {
    public:

	/**
	 * \brief C'tor, initializes the document root of this file handler.
	 * 
	 * Requests are served relative to the document root.
	 */
    FileHandler( const boost::filesystem::path & documentRoot ) : m_documentRoot( documentRoot ) {
	}

	/**
	 * \brief Handles the supplied request and writes to the 
	 * supplied connection.
	 * 
	 * Requests are served relative to the document root.
	 */
	void handle( const HttpServer::request_type & request, HttpServer::connection_ptr_type connection ) {

	    SHARK_LOG_DEBUG( HttpServer::LOGGER, "Enter: FileHandler::handle()", "Enter: FileHandler::handle()" );

	    HttpServer::async_server_type::response_header headers[] = {
		{"Connection","close"},
		{"Content-Type", "x-application/octet-stream"},
		{"Content-Length", "0"},
	    };

	    boost::filesystem::path p = m_documentRoot / request.destination;

	    SHARK_LOG_DEBUG( 
			    HttpServer::LOGGER, 
			    "FileHandler::handle(): " + request.destination + " maps to: " + p.string(), 
			    "FileHandler::handle()" 
			     );

	    if( !boost::filesystem::exists( p ) ) {
		SHARK_LOG_WARNING( 
				  HttpServer::LOGGER, 
				  "FileHandler::handle(): " + p.string() + " does not exist.",
				  "FileHandler::handle()" 
				   );
		connection->set_status( HttpServer::connection_type::not_found );
		connection->set_headers( boost::make_iterator_range( headers, headers + 2 ) );
		return;
	    }

	    boost::filesystem::file_status status = boost::filesystem::status( p );

	    switch( status.type() ) {
	    case boost::filesystem::file_not_found:
	    case boost::filesystem::fifo_file:
	    case boost::filesystem::socket_file:
	    case boost::filesystem::type_unknown:
		connection->set_status( HttpServer::connection_type::not_found );
		connection->set_headers( boost::make_iterator_range( headers, headers + 2 ) );
		return;
		break;
	    case boost::filesystem::regular_file:
		handleRegularFile( p, request, connection );
		break;
	    case boost::filesystem::directory_file:
		handleDirectory( p, request, connection );
		break;
	    case boost::filesystem::symlink_file:
	    case boost::filesystem::block_file:
	    case boost::filesystem::character_file:
		connection->set_status( HttpServer::connection_type::not_found );
		connection->set_headers( boost::make_iterator_range( headers, headers + 2 ) );
		return;
		break;
	    }

	    SHARK_LOG_DEBUG( HttpServer::LOGGER, "Leave: FileHandler::handle()", "Leave: FileHandler::handle()" );
	}

	/** 
	 * \brief Generates html directory listings. 
	 */
	void handleDirectory( 
			     const boost::filesystem::path & p, 
			     const HttpServer::request_type & request, 
			     HttpServer::connection_ptr_type connection ) {
			 
	    HttpServer::async_server_type::response_header headers[] = {
		{ "Content-Type", "text/html" },
		{ "Content-Length", "0" },
	    };

	    std::stringstream ss;
	    ss << "<html>";
	    ss << "<body>";
	    ss << "<ul class=\"listing\">";
			
	    std::string dir = request.destination + ( boost::algorithm::ends_with( request.destination, "/" ) ? "" : "/" );
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

	    headers[ 1 ].value = boost::lexical_cast< std::string >( ss.str().size() );

	    connection->set_status( HttpServer::connection_type::ok );
	    connection->set_headers( boost::make_iterator_range( headers, headers + 2 ) );
	    connection->write( ss.str() );
	}

	/** 
	 * \brief Sends out the file content over the supplied connection. 
	 */
	void handleRegularFile( 
			       const boost::filesystem::path & p, 
			       const HttpServer::request_type & request,
			       HttpServer::connection_ptr_type connection ) {
	    HttpServer::async_server_type::response_header headers[] = {
		{"Content-Type", "x-application/octet-stream"},
		{"Content-Length", "0"},
	    };

	    boost::interprocess::file_mapping fm( p.string().c_str(), boost::interprocess::read_only );
	    boost::interprocess::mapped_region mr( fm, boost::interprocess::read_only );
			
	    headers[ 0 ].value = MimeType::mimeTypeForFile( p ).string();
	    headers[ 1 ].value = boost::lexical_cast< std::string >( mr.get_size() );

	    connection->set_status( HttpServer::connection_type::ok );
	    connection->set_headers( boost::make_iterator_range( headers, headers + 2 ) );

	    unsigned char * data = static_cast< unsigned char * >( mr.get_address() );
	    connection->write( 
			      boost::make_iterator_range( data, data + mr.get_size() ) );

	    // boost::interprocess::file_mapping::remove( p.string().c_str() );
	}

	boost::filesystem::path m_documentRoot; ///< Stores the document root relevant to the local file system.
    };

}

#endif
