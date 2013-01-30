#include <shark/Core/LogFormatters/PrintfLogFormatter.h>
#include <shark/Core/LogHandlers/StreamHandler.h>

#include <shark/Network/HttpServer.h>

#include <shark/Network/Handlers/AboutHandler.h>
#include <shark/Network/Handlers/FileHandler.h>
#include <shark/Network/Handlers/RestHandler.h>

#include <shark/Rng/GlobalRng.h>

// Connect to the server by:
// http://localhost:9090/groupgrid.html

namespace shark {
    struct Producer : public ProbeProvider< Producer > {

	Producer() : ProbeProvider<Producer>( "Producer" ) {
	    for( unsigned int i = 0; i < 30; i++ )
		m_probes.push_back( 
				   registerProbe( 
						 ( boost::format( "Probe_%1%" ) % i ).str(), 
						 "An example probe" 
						  ) 
				    );
	}

	void start() {
	    m_thread = boost::move( boost::thread( boost::bind( &Producer::run, this ) ) );
	}

	void stop() {
	    m_thread.join();
	}

	void run() {
			
	    while( true ) {
		m_probes[ shark::Rng::discrete( 0, m_probes.size()-1 ) ]->setValue( shark::Rng::gauss() );

		boost::this_thread::sleep( boost::posix_time::milliseconds( 100 ) );
	    }

	}

	std::vector< 
	    boost::shared_ptr< 
		shark::Probe
		>
	    > m_probes;

    boost::thread m_thread;
};
}

int main( int argc, char ** argv ) {

    boost::shared_ptr< 
	shark::Logger::AbstractHandler 
    > logHandler( shark::LogHandlerFactory::instance()[ "CoutLogHandler" ] );
    logHandler->setFormatter( 
			     boost::shared_ptr< 
				 shark::Logger::AbstractFormatter 
			     >( shark::LogFormatterFactory::instance()[ "PlainTextLogFormatter" ] )
			      );
    shark::HttpServer::LOGGER->registerHandler( logHandler );
    shark::HttpServer::LOGGER->setLogLevel( shark::Logger::DEBUG_LEVEL );

    shark::Producer producer;
    producer.start();

    shark::HttpServer httpServer( "0.0.0.0", "9090" );
    httpServer.registerHandler( 
			       "/About", 
			       boost::shared_ptr< shark::HttpServer::AbstractRequestHandler >( new shark::SharkAboutHandler() ) 
				);
    httpServer.registerHandler( 
			       "/ProbeManager", 
			       boost::shared_ptr< shark::HttpServer::AbstractRequestHandler >( new shark::RestHandler() ) 
				);
    httpServer.setFallbackHandler(
				  boost::shared_ptr< shark::HttpServer::AbstractRequestHandler >( new shark::FileHandler( boost::filesystem::initial_path() ) ) 
				  );

    httpServer.start();

    return( EXIT_SUCCESS );
}
