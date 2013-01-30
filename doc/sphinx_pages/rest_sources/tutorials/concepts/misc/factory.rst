Factory Pattern Implementation
==============================

The Shark machine learning library provides an implementation of the
well-known factory-method-pattern for runtime-type resolution and
object instantiation. The implementation aims for as much flexibility
as possible by abstracting both the class type to be constructed as
well as the index or tag type. 

This tutorial illustrates the definition of factories for user-defined
types and explains the numerous convenience macros and functions that
are part of the implementation.

For the remainder of this tutorial, we assume a custom class with a
public default c'tor much like given in the following code snippet: ::

  class Oracle {
  public:

    virtual ~Oracle() {}

    virtual unsigned int answer() const = 0;
  };

First, we need to include the following header files: ::

  #include <shark/Core/Factory.h>
  
  #include <string>

Next, we define some subclasses of the abstract Oracle specified before: ::

  class OracleOfDelphi : public Oracle {
  public:
    unsigned int answer() const { return( 41 ); } // Slightly off, but don't worry :-)
  };

And one oracle giving the right answer: ::

  class HitchhikersGuideToTheGalaxy : public Oracle {
  public:
    unsigned int answer() const { return( 42 ); } // Much better :-)
  };

Now, we define our custom Factory for associating names of oracles
with the respective factories: ::

  typedef Factory< Oracle, std::string > OracleFactory;

And finally, we are able to populate our factory of oracles: ::

  OracleFactory::instance().registerType( 
    "Delphi", 
    new TypeErasedAbstractFactory< OracleOfDelphi, OracleFactory >() 
  );
  
  OracleFactory::instance().registerType( 
    "HitchhikersGuideToTheGalaxy", 
    new TypeErasedAbstractFactory< HitchhikersGuideToTheGalaxy, OracleFactory >() 
  );

As you see, every factory type specialized for an object type and a
tag type is modelled as a singleton.

Instantiating oracles and querying the answer to life and everything
else is then carried out as illustrated in the following source code
snippet: ::

  Oracle * o1 = OracleFactory::instance()[ "Delphi" ];
  if( !o1 ) 
    std::cout << "Delphi says: " << o1->answer() << std::endl;
  Oracle * o2 = OracleFactory::instance()[ "HitchhikersGuideToTheGalaxy" ];
  if( !o2 ) 
    std::cout << "HitchhikersGuideToTheGalaxy says: " << o2->answer() << std::endl;

Solved? Solved!

Compile Time Type Registration and Convenience Macros
=====================================================

Considering the complete example presented before once again: ::

  #include <shark/Core/Factory.h>
  
  #include <string>

  int main( int argc, char ** argv ) {
    OracleFactory::instance().registerType( 
      "Delphi", 
      new TypeErasedAbstractFactory< OracleOfDelphi, OracleFactory >() 
    );
  
    OracleFactory::instance().registerType( 
      "HitchhikersGuideToTheGalaxy", 
      new TypeErasedAbstractFactory< HitchhikersGuideToTheGalaxy, OracleFactory >() 
    );

    Oracle * o1 = OracleFactory::instance()[ "Delphi" ];
    if( o1 ) 
      std::cout << "Delphi says: " << o1->answer() << std::endl;
    Oracle * o2 = OracleFactory::instance()[ "HitchhikersGuideToTheGalaxy" ];
    if( o2 ) 
      std::cout << "HitchhikersGuideToTheGalaxy says: " << o2->answer() << std::endl;    

    return( EXIT_SUCCESS );
  }

Note that registering types is carried out manually up until now. In
an ideal world, we would like to delegate the registry of types to the
compiler or to the runtime environment and a basic idea arises: all we
need is a method that is automatically executed at program startup
and which registers the respective types with the correct
factory. Fortunately, this task can be accomplished by means of the
following class: ::

  namespace shark {
    ...

    	template<typename FactoryType>
	struct FactoryRegisterer {

		/**
		* \brief C'tor.
		*/
		FactoryRegisterer( const typename FactoryType::tag_type & tag, typename FactoryType::AbstractFactory * factory ) {
			FactoryType::instance().registerType( tag, factory );
		}
	};

    ...
  }

Of course, we do not want to require every class to have a static
member of type FactoryRegisterer. For this reason, we propose the
usage of convenience macros of the following form: ::

  #define ANNOUNCE_ORACLE( Oracle, Factory ) \
    namespace Oracle ## _detail {\
      typedef TypeErasedAbstractFactory< Oracle, Factory > abstract_factory_type;\
      typedef FactoryRegisterer< Factory > factory_registerer_type;\
      static factory_registerer_type FACTORY_REGISTERER = factory_registerer_type( #Oracle, new abstract_factory_type() );\
  }\

The macro defines a type specific namespace and sets up a static
variable FACTORY_REGISTERER. On program startup, before the execution
of main, the object is instantiated and registered with the respective factory.
Thus, the source code for the tutorial example can be altered to: ::

  #include <shark/Core/Factory.h>
  
  #include <iostream>
  #include <string>

  ANNOUNCE_ORACLE( OracleOfDelphi, OracleFactory );
  ANNOUNCE_ORACLE( HitchhikersGuideToTheGalaxy, OracleFactory );

  int main( int argc, char ** argv ) {
    Oracle * o1 = OracleFactory::instance()[ "Delphi" ];
    if( o1 ) 
      std::cout << "Delphi says: " << o1->answer() << std::endl;
    Oracle * o2 = OracleFactory::instance()[ "HitchhikersGuideToTheGalaxy" ];
    if( o2 ) 
      std::cout << "HitchhikersGuideToTheGalaxy says: " << o2->answer() << std::endl;    

    return( EXIT_SUCCESS );
  }

.. warning::

  Compile time type registration might be affected by the so-called
  static initialization fiasco (see
  http://www.parashift.com/c++-faq-lite/ctors.html#faq-10.14). Thus,
  accessing the factory singleton instance is only save when entering
  main.
