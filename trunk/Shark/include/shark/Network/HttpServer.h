/**
 *
 *  \brief Implements an http server extendable with custom request handlers.
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
#ifndef SHARK_NETWORK_HTTP_SERVER_H
#define SHARK_NETWORK_HTTP_SERVER_H

#include <shark/Core/Shark.h>

#include <boost/bimap.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/noncopyable.hpp>

#include <boost/network/protocol/http/server.hpp>

#include <fstream>
#include <iostream>
#include <map>
#include <string>

namespace shark {
    
    /**
     * \brief Mime-type registry to ease mapping filenames to mime types.
     */
    class MimeType {
    public:
        
	/**
         * \brief Models the actual mime type as a major and a minor type.
         */
	struct Type {
            
	Type( const std::string & majorType = std::string(), const std::string & minorType = std::string() ) : m_majorType( majorType ),
		m_minorType( minorType ) {}
            
	    std::string string() const {
		return( m_majorType + "/" + m_minorType );
	    }
            
	    std::string m_majorType;
	    std::string m_minorType;
	};
        
	/**
         * \brief Models the type of the registry for storing known mime types.
         */
	typedef std::map< std::string, Type > registry_type;
        
	/**
         * \brief Queries the registry for the mime type of the supplied file.
         */
	static Type mimeTypeForFile( const boost::filesystem::path & p ) {
            
	    registry_type::iterator it = m_lut.find( boost::filesystem::extension( p ) );
	    if( it != m_lut.end() )
		return( it->second );
            
	    return( Type( "text", "plain" ) );
	}
        
    protected:
	static registry_type m_lut;
    };
    
    /**
     * \brief Initializes the mime-type registry for common file types.
     */
    MimeType::registry_type MimeType::m_lut = boost::assign::map_list_of
	( ".html",	MimeType::Type( "text",			"html"			) )
	( ".xhtml",	MimeType::Type( "text",			"html"			) )
	( ".txt",	MimeType::Type( "text",			"plain"			) )
	( ".css",	MimeType::Type( "text",			"css"			) )
	( ".js",	MimeType::Type( "text",			"javascript"	) )
	( ".png",	MimeType::Type( "image",		"png"			) )
	( ".jpeg",	MimeType::Type( "image",		"jpeg"			) )
	( ".jpg",	MimeType::Type( "image",		"jpeg"			) )
	( ".json",	MimeType::Type( "application",	"json"			) );
    
    /**
     * \brief Implements an asynchronous http-server with arbitrary handlers.
     */
    class HttpServer {
    public:
        
	/**
         * \brief Default logger of all http server instances.
         */
	static boost::shared_ptr< shark::Logger > LOGGER;
        
	// Forward Declaration
	struct Handler;
        
	/** \brief Synchronous server type. */
	typedef boost::network::http::server< 
	    Handler 
	    > sync_server_type;
        
	/** \brief Asynchronous server type. */
	typedef boost::network::http::async_server< 
	    Handler  
	    > async_server_type;
        
	/** \brief HTTP request type */
	typedef async_server_type::request request_type;
        
	/** \brief HTTP response type */
	typedef async_server_type::response response_type;
        
	/** \brief Asynchronous connection type */
	typedef async_server_type::connection connection_type;
        
	/** \brief Managed ptr for asynchronous connections */
	typedef async_server_type::connection_ptr connection_ptr_type;
        
	/** \brief Models a handler for HTTP requests.*/
	class AbstractRequestHandler {
	public:
	    /** \brief Virtual d'tor. */
	    virtual ~AbstractRequestHandler() {}
            
	    /** \brief Accesses the name of the handler. */
	    virtual const std::string & name() const {
		return( m_name );
	    }
            
	    /**
             * \brief Frontend method, calls the virtual function handle with the supplied arguments.
             */
	    void operator()( const request_type & request, connection_ptr_type connection ) {
		handle( request, connection );
	    }
            
	    /** \brief Called from the server, needs to be implemented by custom request handlers. */
	    virtual void handle( const request_type & request, connection_ptr_type connection ) = 0;
            
	protected:
	    std::string m_name; ///< Stores the name of the handler.
	};
        
	/**
         * \brief Constructs the server for the supplied host and port.
         * \param [in] host Network interface to bind to, default: 0.0.0.0.
         * \param [in] port Port to bind the server to.
         */
    HttpServer( const std::string & host, const std::string & port ) : m_threadPool( 100 ),
	    m_handler( *this ) {
            
	    mp_server.reset( 
                            new async_server_type( 
                                                  host, 
                                                  port, 
                                                  m_handler, 
                                                  m_threadPool, 
                                                  m_ioService 
						   ) 
			     );
            
	    SHARK_LOG_INFO( LOGGER, (boost::format( "Initialized server at: %1%:%2%" ) % host % port).str(), "shark::HttpServer" );
	}
        
	/**
         * \brief Starts the server.
         */
	void start() {
	    SHARK_LOG_DEBUG( LOGGER, "Starting server thread", "shark::HttpServer" );
	    mp_server->run();
	}
        
	/**
         * \brief Stops the server.
         */
	void stop() {
	    mp_server->stop();
	    SHARK_LOG_DEBUG( LOGGER, "Stopped server thread", "shark::HttpServer" );
	}
        
	/**
         * \brief Registers the supplied handler with the supplied uri(resource).
         */
	void registerHandler( const std::string & uri, boost::shared_ptr< AbstractRequestHandler > handler ) {
	    m_handler.m_handlerRegistry.insert( registry_type::value_type( uri, handler ) );
	}
        
	/**
         * \brief Unregisters the handler for the supplied uri(resource).
         */
	void unregisterHandler( const std::string & uri ) {
	    //boost::lock_guard< boost::mutex > lg( m_handlerRegistryGuard );
	    registry_type::left_iterator it = m_handler.m_handlerRegistry.left.find( uri );
	    if( it != m_handler.m_handlerRegistry.left.end() )
		m_handler.m_handlerRegistry.left.erase( it );
	}
        
	/**
         * \brief Removes the supplied handler from the handler registry.
         */
	void unregisterHandler( boost::shared_ptr< AbstractRequestHandler > handler ) {
	    //boost::lock_guard< boost::mutex > lg( m_handlerRegistryGuard );
	    registry_type::right_iterator it = m_handler.m_handlerRegistry.right.find( handler );
	    if( it != m_handler.m_handlerRegistry.right.end() )
		m_handler.m_handlerRegistry.right.erase( it );
	}
        
	/**
         * \brief Accesses the fallback handler of this server.
         *
         * The fallback handler is called for every request that could not be handled
         * by any other handler.
         */ 
	const boost::shared_ptr< 
	    AbstractRequestHandler 
	    > & fallbackHandler() const {
	    return( m_fallbackHandler );
	}
        
	/**
         * \brief Sets the fallback handler of this server.
         */ 
	void setFallbackHandler( const boost::shared_ptr< AbstractRequestHandler > & handler ) {
	    m_fallbackHandler = handler;
	}
        
	/** \brief Models the registry type for associating resources with handlers. */
	typedef boost::bimap< 
	    std::string, 
	    boost::shared_ptr< 
	    AbstractRequestHandler 
	    > 
	    > registry_type;
        
	/**
         * \brief Internal handler type.
         */
	struct Handler {
            
	Handler( HttpServer & server ) : m_httpServer( server ) {}
            
	    void operator()( const request_type & request, connection_ptr_type connection ) {
		boost::shared_ptr<
		    AbstractRequestHandler
		    > handler;
                
		SHARK_LOG_DEBUG( 
                                HttpServer::LOGGER, 
                                ( boost::format( "Request received from %1% for resource %2% and verb %3%" ) % request.source % request.destination % request.method ).str(), 
                                "shark::HttpServer::Handler::operator()" 
				 );
                
		std::string path( request.destination );
		path.erase( std::find( path.begin(), path.end(), '?' ), path.end() );
		registry_type::left_iterator it = m_handlerRegistry.left.find( 
                                                                              path
									       );
                
		if( it != m_handlerRegistry.left.end() )
		    handler = it->second;
                
		if( handler ) {
		    SHARK_LOG_DEBUG( HttpServer::LOGGER, "Calling handler: " + handler->name(), "shark::HttpServer::Handler::operator()"  );
		    handler->handle( request, connection );
		}
		else {
		    SHARK_LOG_DEBUG( HttpServer::LOGGER, "Dispatching request to fallback handler.", "shark::HttpServer::Handler::operator()" );
		    if( m_httpServer.fallbackHandler() )
			m_httpServer.fallbackHandler()->handle( request, connection );
		    else {
			SHARK_LOG_DEBUG( HttpServer::LOGGER, "No such resource: " + request.destination, "shark::HttpServer::Handler::operator()" );
			connection->set_status( connection_type::not_found );
		    }
		}
	    }
            
        protected:
            
            friend class HttpServer;
            
	    template<typename T>
	    void log( const T & t ) {
		(void) t;
	    }
            
	    registry_type m_handlerRegistry;
	    HttpServer & m_httpServer;
	};
        
	Handler m_handler; ///< Top-level request handler for dispatching request to appropriate abstract request handlers.
        
	boost::asio::io_service m_ioService; ///< IO service for networking purposes.
	boost::network::utils::thread_pool m_threadPool; ///< Thread pool for dispatching requests to handlers and running them in separate threads.
	boost::shared_ptr< async_server_type > mp_server; ///< Instance of the underyling server type.
        
	registry_type m_handlerRegistry; ///< Instance of the handler registry.
	boost::shared_ptr< 
	    AbstractRequestHandler 
	    > m_fallbackHandler; ///< Fallback handler, might be empty.
        
    };
    
    /** \brief Registers the default http-server logger with the Shark logger pool. */
    boost::shared_ptr< shark::Logger > HttpServer::LOGGER = shark::LoggerPool::instance().registerLogger( "edu.rub.ini.shark.http.server" );
    
}

#endif //SHARK_NETWORK_HTTP_SERVER_H 
