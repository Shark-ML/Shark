#include <shark/Core/Shark.h>

#include <shark/Core/Logger.h>
#include <shark/Core/LogFormatters/PrintfLogFormatter.h>
#include <shark/Core/LogHandlers/StreamHandler.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <fstream>


// Worker thread for generating log messages with the specified log level.
void run( shark::Logger::Level level, boost::shared_ptr< shark::Logger > logger ) {
	boost::uuids::uuid uuid = boost::uuids::random_generator()();
	std::stringstream ss; ss << uuid;
	std::string id = ss.str();
	for( unsigned int i = 0; i < 10; i++ ) {
		*logger << shark::Logger::Record( level, "Test message.", "Logger example::worker thread @ " + id );
	}
}

int main( int argc, char ** argv ) {

	// Instantiate a default plain text formatter.
	boost::shared_ptr< shark::Logger::AbstractFormatter > formatter( new shark::PlainTextLogFormatter() );

	// Stream global shark logger to file Shark.txt.
	std::ofstream sharkOut( "Shark.txt" );
	boost::shared_ptr< 
		shark::Logger::AbstractHandler
	> sharkFileHandler( new shark::StlStreamHandler( sharkOut ) );
	sharkFileHandler->setFormatter( formatter );
	// Demonstrates configuration of log level.
	boost::property_tree::ptree root;
	root.put( "LogLevel", "Debug" );

	shark::Shark::logger()->configure( root );
	shark::Shark::logger()->registerHandler( sharkFileHandler );
	shark::Shark::init( argc, argv );

	SHARK_LOG_DEBUG( shark::Shark::logger(), "Debug message", "Logging example main" );
	SHARK_LOG_INFO( shark::Shark::logger(), "Info message", "Logging example main" );
	SHARK_LOG_WARNING( shark::Shark::logger(), "Warning message", "Logging example main" );
	SHARK_LOG_ERROR( shark::Shark::logger(), "Error message", "Logging example main" );

	// Register a shark global log level.
	boost::shared_ptr<
		shark::Logger
	> sharkLogger = shark::LoggerPool::instance().registerLogger( "just.a.string.that.is.not.parsed" );
	sharkLogger->configure( root );

	// List contents of the log handler factory.
	shark::LogHandlerFactory::instance().print( std::cout );

	// setup output files.
	std::ofstream outPlainText( "PlainTextLog.txt" );
	std::ofstream outXml( "XmlLog.txt" );
	std::ofstream outJson( "JsonLog.txt" );

	// Define a plain text file handler and a corresponding formatter.
	boost::shared_ptr< 
		shark::Logger::AbstractHandler
	> plainTextFileHandler( new shark::StlStreamHandler( outPlainText ) );
	plainTextFileHandler->setFormatter( 
		boost::shared_ptr< 
			shark::Logger::AbstractFormatter 
		>( new shark::PlainTextLogFormatter() ) 
	);

	// Define an xml file handler and a corresponding formatter.
	boost::shared_ptr< 
		shark::Logger::AbstractHandler
	> xmlFileHandler( new shark::StlStreamHandler( outXml ) );
	xmlFileHandler->setFormatter( 
		boost::shared_ptr< 
			shark::Logger::AbstractFormatter 
		>( new shark::XmlLogFormatter() ) 
		);

	// Define a json file handler and a corresponding formatter.
	boost::shared_ptr< 
		shark::Logger::AbstractHandler
	> jsonFileHandler( new shark::StlStreamHandler( outJson ) );
	jsonFileHandler->setFormatter( 
		boost::shared_ptr< 
			shark::Logger::AbstractFormatter 
		>( new shark::JsonLogFormatter() ) 
	);

	// Register handlers with the default shark logger defined before.
	sharkLogger->registerHandler( plainTextFileHandler );
	sharkLogger->registerHandler( xmlFileHandler );
	sharkLogger->registerHandler( jsonFileHandler );

	// Iterator the log handler factory, instantiate every registered log handler type,
	// setup its formatter and register it with the shark logger.
	shark::LogHandlerFactory::const_iterator it = shark::LogHandlerFactory::instance().begin();
	while( it != shark::LogHandlerFactory::instance().end() ) {
		boost::shared_ptr< shark::Logger::AbstractHandler > handler( it->second->create() );
		handler->setFormatter( formatter );
		sharkLogger->registerHandler( handler );
		
		++it;
	}

	// Create 4 threads that generate log messages and put them to the shark logger.
	boost::thread_group tg;
	tg.create_thread( boost::bind( run, shark::Logger::DEBUG_LEVEL, sharkLogger ) );
	tg.create_thread( boost::bind( run, shark::Logger::INFO_LEVEL, sharkLogger ) );
	tg.create_thread( boost::bind( run, shark::Logger::WARNING_LEVEL, sharkLogger ) );
	tg.create_thread( boost::bind( run, shark::Logger::ERROR_LEVEL, sharkLogger ) );

	// Wait for all threads to complete.
	tg.join_all();

	return( EXIT_SUCCESS );
}
