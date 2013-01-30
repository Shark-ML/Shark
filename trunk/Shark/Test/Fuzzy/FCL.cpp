// Enable for debugging purposes
#define BOOST_SPIRIT_DEBUG

#define BOOST_TEST_MODULE Fuzzy_Control_Language_Parser
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Fuzzy/FCL/FuzzyControlLanguageParser.h>

#include <fstream>

const char fcl[] = "FUNCTION_BLOCK tipper\n"
"VAR_INPUT\n"
"\t\tservice : REAL;\n"
"\t\tfood : REAL;\n"
"END_VAR\n"
"\n"
"VAR_OUTPUT\n"
"\t\ttip : REAL;\n"
"END_VAR\n"
"\n"
"FUZZIFY service\n"
"\t\tTERM poor := (0, 1) (4, 0) ; \n"
"\t\tTERM good := (1, 0) (4,1) (6,1) (9,0);\n"
"\t\tTERM excellent := (6, 0) (9, 1);\n"
"END_FUZZIFY\n"
"\n"
"FUZZIFY food\n"
"\t\tTERM rancid := (0, 1) (1, 1) (3,0) ;\n"
"\t\tTERM delicious := (7,0) (9,1);\n"
"END_FUZZIFY\n"
"\n"
"DEFUZZIFY tip\n"
"\t\tTERM cheap := (0,0) (5,1) (10,0);\n"
"\t\tTERM average := (10,0) (15,1) (20,0);\n"
"\t\tTERM generous := (20,0) (25,1) (30,0);\n"
"\t\tMETHOD : COG;\n"
"\t\tDEFAULT := 0;\n"
"END_DEFUZZIFY\n"
"\n"
"RULEBLOCK No1\n"
"\t\tAND : MIN;\n"
"\t\tACT : MIN;\n"
"\t\tACCU : MAX;\n"
"\n"
"\t\tRULE 1 : IF service IS poor OR food IS rancid \n"
"\t\t\t\t\t\t\t\tTHEN tip IS cheap;\n"
"\n"
"\t\tRULE 2 : IF service IS good \n"
"\t\t\t\t\t\t\t\tTHEN tip IS average; \n"
"\n"
"\t\tRULE 3 : IF service IS excellent AND food IS delicious \n"
"\t\t\t\t\t\t\t\tTHEN tip IS generous;\n"
"END_RULEBLOCK\n"
"\n"
"END_FUNCTION_BLOCK\n";

void print( const shark::FunctionBlockDeclaration & block ) {

    
    std::cout << "Function Block Name: " << block.m_name << std::endl;

    std::cout << "# input variable declarations: " << block.m_ioDeclarations.m_inputDeclarations.size() << std::endl;
    for( std::size_t i = 0; i < block.m_ioDeclarations.m_inputDeclarations.size(); i++ )
	std::cout << "Input variable: " << block.m_ioDeclarations.m_inputDeclarations[ i ].m_name << std::endl;
    std::cout << "# output variable declarations: " << block.m_ioDeclarations.m_outputDeclarations.size() << std::endl;
    for( std::size_t i = 0; i < block.m_ioDeclarations.m_outputDeclarations.size(); i++ )
    std::cout << "Output variable: " << block.m_ioDeclarations.m_outputDeclarations[ i ].m_name << std::endl;
    std::cout << "# fuzzify blocks: " << block.m_functionBlockBody.m_fuzzifyBlocks.size() << std::endl;
    std::cout << "# defuzzify blocks: " << block.m_functionBlockBody.m_defuzzifyBlocks.size() << std::endl;
    std::cout << "# rule blocks: " << block.m_functionBlockBody.m_ruleBlocks.size() << std::endl;  
    
}

BOOST_AUTO_TEST_CASE( Fuzzy_Control_Language_Parser ) {

    std::string s( fcl );
    std::istringstream iss( fcl );
    iss.unsetf( std::ios::skipws );
    // wrap istream into iterator
    boost::spirit::istream_iterator begin(iss);
    boost::spirit::istream_iterator end;
    
    shark::FuzzyControlLanguageParser< boost::spirit::istream_iterator > fclParser;

    shark::FunctionBlockDeclaration block;

    BOOST_CHECK( boost::spirit::qi::phrase_parse(
					begin, 
					end, 
					fclParser,
					boost::spirit::ascii::space,
					block
					 )
    );

    print( block );
}

/*BOOST_AUTO_TEST_CASE( Fuzzy_Control_Language_Parser ) {



}*/
