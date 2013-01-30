#include <shark/Core/Shark.h>

#include <iostream>

int main( int argc, char ** argv ) {

	shark::Shark::init( argc, argv );
	shark::Shark::info( std::cout );

}