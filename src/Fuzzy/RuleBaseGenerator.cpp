
/**
 * \file RuleBaseGenerator.cpp
 *
 * \brief Reads and writes rule bases from and to XML-files.
 *
 * \author Marc Nunkesser
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */

/* $log$ */

#include <Fuzzy/RuleBaseGenerator.h>

#include <Fuzzy/RuleBase.h>
#include <Fuzzy/LinguisticVariable.h>
#include <Fuzzy/LinguisticTerm.h>
#include <Fuzzy/MamdaniIM.h>
#include <Fuzzy/RCPtr.h>
#include <Fuzzy/SugenoIM.h>
#include <Fuzzy/BellLT.h>
#include <Fuzzy/GeneralizedBellLT.h>
#include <Fuzzy/SigmoidalLT.h>
#include <Fuzzy/TriangularLT.h>
#include <Fuzzy/TrapezoidLT.h>

// #include <Fuzzy/xmlParser.h>

#include <sstream>
#include <string>

/*template<typename T>
static T convert( const std::string & s ) {
	std::stringstream ss( s );
	T t; ss >> t;
	return( t );
}

static std::string check_for_null( const char * data ) {
	std::string toReturn( data == 0 ? "" : data );
	return( toReturn );
}

static LinguisticTerm * process_linguistic_term( XMLNode node, RCPtr<LinguisticVariable> & parent );

RuleBase build_rule_base_from_xml( const std::string & descriptionFile ) {
	RuleBase result;
	
	XMLNode mainNode = XMLNode::openFileHelper( descriptionFile.c_str(), "FuzzyController" );
	XMLNode node;
	int noLinguisticVariables = mainNode.nChildNode( "LinguisticVariable" ); 
	
	for( int i = 0; i < noLinguisticVariables; i++ ) {
		node = mainNode.getChildNode( "LinguisticVariable", i ); 
		if( !node.isAttributeSet( "name" ) ) {
			std::cout << "(EE) Every linguistic variable needs to be named." << std::endl;
			continue;
		}
		std::string name = check_for_null( node.getAttribute( "name" ) );
		
		std::string type = node.isAttributeSet( "type" ) ? node.getAttribute( "type" ) : "Input";
		RCPtr<LinguisticVariable> lv( new LinguisticVariable( name ) );
		
		double min = node.getChildNode( "Range" ).isAttributeSet( "minimum" ) ? convert<double>( node.getChildNode( "Range" ).getAttribute( "minimum" ) ) : -std::numeric_limits<double>::max();
		double max = node.getChildNode( "Range" ).isAttributeSet( "maximum" ) ? convert<double>( node.getChildNode( "Range" ).getAttribute( "maximum" ) ) : std::numeric_limits<double>::max();
		
		lv->setBounds( min, max );
		
		int noLinguisticTerms = node.nChildNode( "LinguisticTerm" );
		for( int j = 0; j < noLinguisticTerms; j++ ) {
			XMLNode ltNode = node.getChildNode( "LinguisticTerm", j );
			LinguisticTerm * lt = process_linguistic_term( ltNode, lv );
			if( lt != 0 )
				lv->addTerm( lt );
			else
				std::cout << "(EE) Processing Term failed." << std::endl; 
		}
		
		if( type == "Input" )
			result.addToInputFormat( lv );
		else if( type == "Output" )
			result.addToOutputFormat( lv );
	}
	
	XMLNode ruleBaseNode = mainNode.getChildNode( "RuleBase" );
	
	std::string connectionType = ruleBaseNode.isAttributeSet("Connection") ? ruleBaseNode.getAttribute( "Connection" ) : "AND";
	int connection = AND;
	if( connectionType == "AND" ) {
		connection = AND;
	} else if( connectionType == "OR" ) {
		connection = OR;
	}  else if( connectionType == "PRODUCT" ) {
		connection = PROD;
	}  else if( connectionType == "PROBOR" ) {
		connection = PROBOR;
	}
	
	int noRules = ruleBaseNode.nChildNode( "Rule" );
	for( int i = 0; i < noRules; i++ ) {
		std::string rule = check_for_null( ruleBaseNode.getChildNode( "Rule", i ).getText() );
		for( std::string::iterator it = rule.begin(); it != rule.end(); ) 
			if( *it == '(' || *it == ')' )
				it = rule.erase( it );
			else
				++it;
		
		double weight = ruleBaseNode.getChildNode( "Rule", i ).isAttributeSet( "weight" ) ? convert<double>( ruleBaseNode.getChildNode( "Rule", i ).getAttribute( "weight" ) ) : 1.0;
		result.addRule( RCPtr<Rule>( new Rule( rule, &result, weight ) ) );
	}
	
	return( result );
}

static LinguisticTerm * process_linguistic_term( XMLNode node, RCPtr<LinguisticVariable> & parent ) {
	std::string name = check_for_null( node.getAttribute( "name" ) );
	std::string type = check_for_null( node.getAttribute( "type" ) );
	
	LinguisticTerm * result = 0;
	
	std::cout << "(II) name: " << name << " type: " << type << std::endl;
	
	if( type == "Triangle" ) {
		int n = node.nChildNode( "Point" );
		std::vector<double> v( 3 );
		for( int i = 0; i < n; i++ ) {
			v[i] = convert<double>( check_for_null( node.getChildNode( "Point", i ).getText() ) );
		}
		result = new TriangularLT( name, parent, v[0], v[1], v[2] );
	} else if( type == "Trapezoid" ) {
		assert( node.nChildNode( "Point" ) == 4 );
		result = new TrapezoidLT( name, 
								  parent, 
								  convert<double>( node.getChildNode( "Point", 0 ).getText() ),
								  convert<double>( node.getChildNode( "Point", 1 ).getText() ),
								  convert<double>( node.getChildNode( "Point", 2 ).getText() ),
								  convert<double>( node.getChildNode( "Point", 3 ).getText() )
								  );
	} else if( type == "Sigmoidal") {
		double steigung = convert<double>( node.getChildNode( "Steigung" ).getText() );
		//double steigung = node.isAttributeSet( "Steigung" ) ? convert<double>(node.getAttribute("Steigung")) : 1;
		double offset = convert<double>( node.getChildNode( "Offset" ).getText() );
		// double offset = node.isAttributeSet( "Offset" ) ? convert<double>(node.getAttribute("Offset")) : 0;
		result = new SigmoidalLT( name, parent, steigung, offset );
	} else if( type == "Bell" ) {
		double sigma = convert<double>( node.getChildNode( "Sigma" ).getText() );
		// double sigma = node.isAttributeSet( "Sigma" ) ? convert<double>(node.getAttribute("Sigma")) : 1;
		double offset = convert<double>( node.getChildNode("Offset").getText() );
		//double offset = node.isAttributeSet( "Offset" ) ? convert<double>(node.getAttribute("Offset")) : 0;
		double scale = convert<double>( node.getChildNode("Scale").getText() );
		// double scale = node.isAttributeSet( "Scale" ) ? convert<double>(node.getAttribute("Scale")) : 1;
		result = new BellLT( name, 
							 parent, 
							 sigma,
							 offset,
							 scale
							 );
	} else if( type == "GeneralizedBell" ) {
		double slope = convert<double>( node.getChildNode("Slope").getText() );
		double center = convert<double>( node.getChildNode("Center").getText() );
		double width = convert<double>( node.getChildNode("Width").getText() );
		result = new GeneralizedBellLT( name, parent, slope, center, width );
	}
	
	if( result != 0 ) {
		char fn[256];
		sprintf( fn, "%s.dat", name.c_str() );
		if( name == "VG" )
			result->makeGNUPlotData(fn, 1000, 0, 0.4);
		if( name == "VK" )
			result->makeGNUPlotData(fn, 1000, 1, 2);
		else
			result->makeGNUPlotData(fn, 1000);
	}
	
	return( result );
}

static std::string linguistic_variable_to_xml( const RCPtr<LinguisticVariable> & lv );
static std::string rule_to_xml( const RCPtr<Rule> & rule );

bool save_rule_base_to_xml( const std::string & descriptionFile, RuleBase & rb ) {
	std::ofstream ofs( descriptionFile.c_str() );
	
	if( !ofs.is_open() )
		return( false );
	
	ofs << "<FuzzyController>" << std::endl;
	
	RuleBase::FormatIterator fIt = rb.getFirstFormatIterator();
	RuleBase::FormatIterator fItE = rb.getLastFormatIterator();
	
	while( fIt != fItE ) {
		ofs << linguistic_variable_to_xml( *fIt ) << std::endl;
		++fIt;
	}
	
	fIt = rb.getFirstConclIt();
	fItE = rb.getLastConclIt();
	
	while( fIt != fItE ) {
		ofs << linguistic_variable_to_xml(*fIt) << std::endl;
		++fIt;
	}
	
	RuleBase::BaseIterator rIt = rb.getFirstIterator();
	RuleBase::BaseIterator rItE = rb.getLastIterator();
	
	while( rIt != rItE ) {
		ofs << rule_to_xml(*rIt) << std::endl;
		++rIt;
	}
	
	return( true );
}

static std::string linguistic_variable_to_xml( const RCPtr<LinguisticVariable> & lv ) {
	return( std::string( "(EE) linguistic_variable_to_xml(...) not yet implemented." ) );
}

static std::string rule_to_xml( const RCPtr<Rule> & rule ) {
	return( std::string( "(EE) rule_to_xml(...) not yet implemented." ) );
}
*/
//RuleBase build_rule_base_from_fcl(const std::string & uri) {
//	std::cout << "(EE) Parser for fcl not implemented yet." << std::endl;
//	return( RuleBase() );
//}

/*
static bool starts_with( const std::string & s, const std::string & prefix ) {
	return( s.find( prefix ) == 0 );
}

static void strip_white_space( std::string & s ) {
	int i;
	
	for( i = 0; i < s.size(); i++ ) {
		if( !isspace( s[i] ) )
			break;
	}
	s.erase( 0, i );
	
	for( i = s.size() - 1; i >= 0; i-- )
		if( !isspace( s[i] ) )
			break;
	s.erase( i + 1 );
}

void process_input_variable( ParseResult & rb, const std::string & line );
void process_output_variable( ParseResult & rb, const std::string & line );
void process_term( ParseResult & rb, RCPtr<LinguisticVariable> & lv, const std::string & line );
void process_rule( ParseResult & rb, const std::string & line );

static bool operator==( const RCPtr<LinguisticVariable> & lv, const std::string & name ) {
	return( lv->getName() == name );
}

static ParseResult parse_fcl_data_source( std::istream & in ) {
	ParseResult pr;
	
	ParserState state = Default;
	
	std::string line; RCPtr<LinguisticVariable> currentLV;
	
	while( in ) {
		
		std::getline( in, line );
		
		strip_white_space( line );
		
		if( line.size() == 0 )
			continue;
		
		if( line == "VAR_INPUT" ) {
			state = InputVariable;
			continue;
		} else if( line == "VAR_OUTPUT" ) {
			state = OutputVariable;
			continue;
		} else if( line == "END_VAR" ) {
			state = Default;
			continue;
		}
		else if( starts_with( line, "FUZZIFY" ) ) {
			state = Term; std::string name;
			for( int i = line.size() - 1; i >= 0; i-- ) {
				if( isspace( line[i] ) )
					break;
				
				name = line[i] + name;
			}
			std::vector<RCPtr<LinguisticVariable> >::iterator it = std::find( pr.m_inputVariables.begin(), pr.m_inputVariables.end(), name );
			if( it == pr.m_inputVariables.end() ) {
				it = std::find( pr.m_outputVariables.begin(), pr.m_outputVariables.end(), name );
				if( it != pr.m_outputVariables.end() )
					currentLV = *it;
			} else
				currentLV = *it;
				
			continue;
		}
		else if( line == "END_FUZZIFY" ) {
			state = Default;
			currentLV = RCPtr<LinguisticVariable>( 0 );
			continue;
		}
		else if( starts_with( line, "RULE" ) ) {
			state = Rule;
		}

		// State Management finished, handling actual input
		switch( state ) {
			case Default:
				break;
			case InputVariable:
				process_input_variable( pr, line );
				break;
			case OutputVariable:
				process_output_variable( pr, line );
				break;
			case Term:
				process_term( pr, currentLV, line );
				break;
			case Rule:
				process_rule( pr, line );
				state = Default;
				break;
		}
	}
	
	return( pr );
}

static void process_input_variable( ParseResult & rb, const std::string & line ) {
	std::stringstream s( line ); std::string name;
	s >> name;
	
	rb.m_inputVariables.push_back( RCPtr<LinguisticVariable>( new LinguisticVariable( name ) ) );
}

static void process_output_variable( ParseResult & rb, const std::string & line ) {
	std::stringstream s( line ); std::string name;
	s >> name;
	
	rb.m_outputVariables.push_back( RCPtr<LinguisticVariable>( new LinguisticVariable( name ) ) );
}

typedef enum {
	DefaultTermParserState,
	TermName,
	Point
} TermParserState;

static void process_term( ParseResult & rb, RCPtr<LinguisticVariable> & lv, const std::string & line ) {
	std::string name, point; std::vector<double> points;
	
	TermParserState state = DefaultTermParserState;
	
	std::string::const_iterator it, itE;
	it = line.begin();
	itE = line.end();
	
	while( it != itE ) {
		
		std::cout << *it;
		
		if( isspace( *it ) && name.size() == 0 ) {
			state = TermName;
			++it;
			continue;
		} else if( isspace( *it ) && name.size() > 0 ) {
			state = DefaultTermParserState;
			++it;
			continue;
		}
		else if( *it == '(' ) {
			state = Point;
			++it;
			continue;
		} else if( *it == ')' ) {
			state = DefaultTermParserState;
			// We need to store the "extracted" so far
			unsigned idx = point.find( "," );
			std::cout << "std::string( point.begin(), point.begin() + idx ): " << std::string( point.begin(), point.begin() + idx ) << std::endl;
			std::stringstream ss( std::string( point.begin(), point.begin() + idx ) ); 
			double x; ss >> x;
			points.push_back( x );
			point.clear();
			++it;
			continue;
		}
		
		switch( state ) {
			case TermName:
				name.push_back( *it );
				break;
			case Point:
				point.push_back( *it );
				break;
			case DefaultTermParserState:
			default:
				break;
		}
		
		++it;
	}
	
	// Handle Multiple Points MF Later. Plain Triangular LTs up until now.
	if( !lv ) 
		std::cout << "lv is null" << std::endl;
	else {
		printf( "(II) %f, %f, %f \n", points[0], points[1], points[2] );
		lv->addTerm( new TriangularLT( name, lv, points[0], points[1], points[2] ) );
	}
}

static void process_rule( ParseResult & rb, const std::string & line ) {
	std::string::const_iterator it,itE; std::string rule;
	
	it = line.begin();
	itE = line.end();
	
	int state = 0;
	
	while( it != itE ) {
		
		std::cout << *it;
		
		if( *it == ':' ) {
			state = 1;
			++it;
			continue;
		} else if( *it == ';' ) {
			state = 0;
		}
		
		if( state == 0 ) {
			++it;
			continue;
		} else if( state == 1 && *it != '(' && *it != ')' ) {
			rule.push_back( *it );
		}
		
		++it;
	}
	std::cout << std::endl;
	
	std::cout << "(II) " << rule << std::endl;
	
	if( rule.size() > 0 )
		rb.m_rules.push_back( rule );
}
*/
