//===========================================================================
/*!
 * 
 *
 * \brief       Parser for the Fuzzy Control Language (IEC 61131-7).
 * 
 * The EBNF is given in http://www.fuzzytech.com/binaries/ieccd1.pdf.
 * 
 * 
 *
 * \author      -
 * \date        -
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#ifndef SHARK_FUZZY_CONTROL_LANGUAGE_PARSER_H
#define SHARK_FUZZY_CONTROL_LANGUAGE_PARSER_H

#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/support_ascii.hpp>

#include <boost/optional.hpp>
#include <boost/variant.hpp>

#include <string>
#include <vector>

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace phoenix = boost::phoenix;

namespace shark {

	namespace tag {
		/** \brief Tags an input variable to a fuzzy control system. */
		struct Input {};
		/** \brief Tags an output variable to a fuzzy control system. */
		struct Output {};
	}

	/** \cond */
	namespace detail {

		enum VariableType {
			REAL
		};

		enum ConnectorType {
			AND,
			OR
		};

		enum OrOperatorType {
			OR_MAX,
			OR_ASUM,
			OR_BSUM
		};

		enum AndOperatorType {
			AND_MIN,
			AND_PROD,
			AND_BDIF
		};

		enum ActivationType {
			PROD_ACTIVATION,
			MIN_ACTIVATION
		};

		enum AccumulationType {
			MAX_ACCUMULATION,
			BSUM_ACCUMULATION,
			NSUM_ACCUMULATION
		};

		enum DefuzzificationType {
			COG,
			COGS,
			COA,
			LM,
			RM
		};
	}

	/** \brief Type declaration of a fuzzy term name. */
	typedef std::string TermName;

	/** \brief Type declaration of a fuzzy variable name. */
	typedef std::string VariableName;


	/** \brief Type declaration of a numeric literal. */
	typedef boost::variant<int,double> NumericLiteral;

	/** \brief Models a fuzzy variable. */
	template<typename Tag>
	struct Variable {
		typedef Tag tag_type;
		typedef Variable< tag_type > this_type;

		VariableName m_name; ///< Stores the name of the variable.
		detail::VariableType m_type; ///< Stores the type of the variable.
	};

	/** \brief Type declaration for a set of input variables. */
	typedef std::vector< Variable< tag::Input > > InputDeclarations;

	/** \brief Type declaration for a set of output variables. */
	typedef std::vector< Variable< tag::Output > > OutputDeclarations;

	/** \brief Bundles together input and output variable declarations. */
	struct InputOutputDeclarations {
		InputDeclarations m_inputDeclarations;
		OutputDeclarations m_outputDeclarations;
	};

	/** \brief Models an interval. */
        typedef std::vector< NumericLiteral > NumericRange;

	/** \brief Models a point (x,y). */
	struct Point {

		/** \brief A coordinate is either a numeric literal or a variable name. */
		typedef boost::variant<
			NumericLiteral,
			VariableName
		> Coordinate;

		Coordinate m_x; ///< Stores the x coordinate of the point.
		NumericLiteral m_y; ///< Stores the y coordinate of the point.
	};

	/** \brief Models a set of points. */
	typedef std::vector< Point > Points;

	/** \brief Models a sub condition. */
	struct SubCondition {
		boost::optional< std::string > m_variablePrefix;
		VariableName m_variable;
		boost::optional< std::string > m_termPrefix;
		VariableName m_term;
	};

	struct ConditionClause {
		typedef boost::variant<
			SubCondition,
			VariableName
		> VariantType;

		VariantType m_t;
		detail::ConnectorType m_connector;
	};

	struct Condition {
		boost::variant<
			SubCondition,
			VariableName
		> m_firstClause;

		std::vector< ConditionClause > m_tailClauses;
	};

	struct TermedConclusion {
		VariableName m_variable;
		TermName m_term;
	};

	struct Conclusion {
		VariableName m_var;
		boost::optional< VariableName > m_term;
	};

	typedef boost::variant<
		VariableName,
		NumericLiteral
	> WeightingFactor;

	typedef boost::variant<
		NumericLiteral,
		std::string
	> DefaultValue;

	struct Rule {
		int m_id;
		Condition m_condition;
		Conclusion m_conclusion;
		boost::optional< WeightingFactor > m_weightingFactor;
	};

	typedef detail::AccumulationType AccumulationMethod;
	typedef detail::ActivationType ActivationMethod;
	typedef detail::DefuzzificationType DefuzzificationMethod;

	typedef boost::variant<
		NumericLiteral,
		std::string
	> Singleton;

	typedef boost::variant<
		Singleton,
		std::vector< Point >
	> MemberShipFunction;

        struct LinguisticTermDescription {
		TermName m_name;
		MemberShipFunction m_msf;
	};

	typedef boost::variant<
		detail::OrOperatorType,
		detail::AndOperatorType
	> OperatorDefinition;

	struct RuleBlock {

		VariableName m_name;
		OperatorDefinition m_operatorDefinition;
		boost::optional< ActivationMethod > m_activationMethod;
		AccumulationMethod m_accumulationMethod;
		std::vector< Rule > m_rules;

	};

	struct DefuzzifyBlock {
		VariableName m_name;
                std::vector< LinguisticTermDescription > m_terms;
		DefuzzificationMethod m_defuzzificationMethod;
		DefaultValue m_defaultValue;
                boost::optional< NumericRange > m_range;
	};

	struct FuzzifyBlock {
		VariableName m_variableName;
                std::vector< LinguisticTermDescription > m_terms;
	};

	struct FunctionBlockBody {
		std::vector< FuzzifyBlock > m_fuzzifyBlocks;
		std::vector< DefuzzifyBlock > m_defuzzifyBlocks;
		std::vector< RuleBlock > m_ruleBlocks;
	};

	struct FunctionBlockDeclaration {
		VariableName m_name;
		InputOutputDeclarations m_ioDeclarations;
		FunctionBlockBody m_functionBlockBody;
	};
	/** \endcond */
}

#include <shark/Fuzzy/FCL/FuzzyControlLanguageParserFusionBindings.h>

namespace shark {

	/**
	 * \brief LL-Parser for the Fuzzy-Control-Language (see IEC 61131-7).
	 *
	 * Implemented in terms of boost::spirit, the EBNF is given in http://www.fuzzytech.com/binaries/ieccd1.pdf.
	 */
	template<
	typename Iterator
	> class FuzzyControlLanguageParser : public qi::grammar<Iterator, FunctionBlockDeclaration(), ascii::space_type> {
	public:

		/**
		 * \brief Default c'tor, initializes atomic and compound rules.
		 */
		FuzzyControlLanguageParser() : FuzzyControlLanguageParser::base_type( m_functionBlockDeclaration, "FCL Grammar" ) {

			using namespace qi::labels;

			m_identifier = // +( ascii::char_( "a-zA-Z1-9" ) ) -
					+( ascii::char_( "a-zA-Z1-9" ) ) -
					ascii::string( "VAR_INPUT" ) -
					ascii::string( "VAR_OUTPUT" ) -
					ascii::string( "END_VAR" ) -
					ascii::string( "FUZZIFY" ) -
					ascii::string( "END_FUZZIFY" ) -
					ascii::string( "METHOD" ) -
					ascii::string( "END_DEFUZZIFY" ) -
					ascii::string( "RULEBLOCK" ) -
					ascii::string( "RULE" ) -
					ascii::string( "END_RULEBLOCK" ) -
					ascii::string( "END_FUNCTION_BLOCK" ) -
					ascii::string( "TERM" ) -
					ascii::string( "METHOD" ) -
					ascii::string( "AND" ) -
					ascii::string( "OR" ) -
					ascii::string( "MIN" ) -
					ascii::string( "MAX" ) -
					ascii::string( "NOT" ) -
					ascii::string( "IF" ) -
					ascii::string( "IS" ) -
					ascii::string( "THEN" );

			//m_identifier %= qi::attr_cast( m_identifier );

			m_variableType.add
			( "REAL", detail::REAL );

			m_connectorType.add
			( "AND", detail::AND )
			( "OR", detail::OR );

			m_orOperatorType.add
			( "MAX", detail::OR_MAX )
			( "ASUM", detail::OR_ASUM )
			( "BSUM", detail::OR_BSUM );

			m_andOperatorType.add
			( "MIN", detail::AND_MIN )
			( "PROD", detail::AND_PROD )
			( "BDIF", detail::AND_BDIF );

			m_activationType.add
			( "PROD", detail::PROD_ACTIVATION)
			( "MIN", detail::MIN_ACTIVATION);

			m_accumulationType.add
			( "MAX", detail::MAX_ACCUMULATION )
			( "BSUM", detail::BSUM_ACCUMULATION )
			( "NSUM", detail::NSUM_ACCUMULATION );

			m_defuzzificationType.add
			( "COG", detail::COG )
			( "COGS", detail::COGS )
			( "COA", detail::COA )
			( "LM", detail::LM )
			( "RM", detail::RM );

			m_numericLiteral %= qi::int_ | qi::double_;

			m_weightingFactor %= ( m_identifier | m_numericLiteral );

			m_conclusion %= m_identifier >> -( qi::lit( "IS" ) >> m_identifier );

			m_subcondition %=
					-( ascii::string( "NOT" ) ) >>
					-( qi::lit( "(" ) ) >>
					m_identifier >>
					qi::lit( "IS" ) >>
					-( ascii::string( "NOT" ) ) >>
					m_identifier >>
					-( qi::lit( ")" ) );

			m_condition %=
					(m_subcondition | m_identifier) >>
					*(
							m_connectorType >>
							( m_subcondition | m_identifier )
					);
			m_rule %=
					qi::lit( "RULE" ) >>
					qi::int_ >>
					qi::lit( ":" ) >>
					qi::lit( "IF" ) >>
					m_condition >>
					qi::lit( "THEN" ) >>
					m_conclusion >>
					-( qi::lit( "WITH" ) >>
							m_weightingFactor
					) >>
					qi::lit( ";" );

			m_accumulationMethod %=
					qi::lit( "ACCU" ) >>
					qi::lit( ":" ) >>
					m_accumulationType >>
					qi::lit( ";" );

			m_activationMethod %=
					qi::lit( "ACT" ) >>
					qi::lit( ":" ) >>
					m_activationType >>
					qi::lit( ";" );
			m_operatorDefinition %=
					( qi::lit( "OR" ) >>
							qi::lit( ":" ) >>
							m_orOperatorType
							|
							qi::lit( "AND" ) >>
							qi::lit( ":" ) >>
							m_andOperatorType
					) >>
					qi::lit( ";" );

                        m_range = qi::lit( "RANGE" ) >>
					qi::lit( ":=" ) >>
					qi::lit( "(" ) >>
					m_numericLiteral >>
					qi::lit( ".." ) >>
					m_numericLiteral >>
					qi::lit( ")" );

                        m_defaultValue = qi::lit( "DEFAULT" ) >>
					qi::lit( ":=" ) >>
                                        ( m_numericLiteral | ascii::string( "NC" ) ) >>
					qi::lit( ";" );

			m_defuzzificationMethod %= qi::lit( "METHOD" ) >>
					qi::lit( ":" ) >>
					m_defuzzificationType >>
					qi::lit( ";" );

			m_points %= *(
					qi::lit( "(" ) >>
					(m_numericLiteral | m_identifier) >>
					qi::lit( "," ) >>
					m_numericLiteral >>
					qi::lit( ")" )
			);

                        m_singleton = m_numericLiteral | ascii::string( "NC" );

			m_membershipFunction = m_points | m_singleton;

			m_linguisticTerm =
					qi::lit( "TERM" ) >>
					m_identifier >>
					qi::lit( ":=" ) >>
					m_membershipFunction >>
					qi::lit( ";" );

			m_ruleBlock %=
					qi::lit( "RULEBLOCK" ) >>
					m_identifier >>
					m_operatorDefinition >>
					-( m_activationMethod ) >>
					m_accumulationMethod >>
					*( m_rule ) >>
					qi::lit( "END_RULEBLOCK" );

			m_defuzzifyBlock %=
					qi::lit( "DEFUZZIFY" ) >>
					m_identifier >>
					*( m_linguisticTerm ) >>
					m_defuzzificationMethod >>
					m_defaultValue >>
					-( m_range ) >>
					qi::lit( "END_DEFUZZIFY" );

			m_fuzzifyBlock %=
					qi::lit( "FUZZIFY" ) >>
					m_identifier >>
					*( m_linguisticTerm ) >>
					qi::lit( "END_FUZZIFY" );

			m_functionBlockBody =
					*( m_fuzzifyBlock ) >>
					*( m_defuzzifyBlock ) >>
					*( m_ruleBlock );

			m_inputVariableDeclaration %= qi::lit( "VAR_INPUT" ) >>
					*( m_identifier >>
							qi::lit( ":" ) >>
							m_variableType >> ";" ) >>
							qi::lit( "END_VAR" );

			m_outputVariableDeclaration %=  qi::lit( "VAR_OUTPUT" ) >>
					*( m_identifier >>
							qi::lit( ":" ) >>
							m_variableType >> ";" ) >>
							( qi::lit( "END_VAR" ) );

			m_inputOutputDeclarations %= m_inputVariableDeclaration >>
					m_outputVariableDeclaration;

			m_functionBlockDeclaration %= qi::lit( "FUNCTION_BLOCK" ) >>
					m_identifier >>
					m_inputOutputDeclarations >>
					m_functionBlockBody >>
					qi::lit( "END_FUNCTION_BLOCK" );

			// BOOST_SPIRIT_DEBUG_NODE( connectorType );
                        /* Commented out due to issues with boost 1.44.
                        BOOST_SPIRIT_DEBUG_NODE( m_identifier );
			BOOST_SPIRIT_DEBUG_NODE( m_numericLiteral );
			BOOST_SPIRIT_DEBUG_NODE( m_functionBlockDeclaration );
			BOOST_SPIRIT_DEBUG_NODE( m_inputOutputDeclarations );
			BOOST_SPIRIT_DEBUG_NODE( m_inputVariableDeclaration );
			BOOST_SPIRIT_DEBUG_NODE( m_outputVariableDeclaration );
			BOOST_SPIRIT_DEBUG_NODE( m_fuzzifyBlock );
			BOOST_SPIRIT_DEBUG_NODE( m_defuzzifyBlock );
			BOOST_SPIRIT_DEBUG_NODE( m_linguisticTerm );
			BOOST_SPIRIT_DEBUG_NODE( m_singleton );
			BOOST_SPIRIT_DEBUG_NODE( m_points );
			BOOST_SPIRIT_DEBUG_NODE( m_defuzzificationMethod );
			BOOST_SPIRIT_DEBUG_NODE( m_defaultValue );
			BOOST_SPIRIT_DEBUG_NODE( m_ruleBlock );
			BOOST_SPIRIT_DEBUG_NODE( m_operatorDefinition );
			BOOST_SPIRIT_DEBUG_NODE( m_activationMethod );
			BOOST_SPIRIT_DEBUG_NODE( m_accumulationMethod );
			BOOST_SPIRIT_DEBUG_NODE( m_rule );
			BOOST_SPIRIT_DEBUG_NODE( m_condition );
			BOOST_SPIRIT_DEBUG_NODE( m_subcondition );
                        BOOST_SPIRIT_DEBUG_NODE( m_conclusion );*/

			qi::on_error< qi::fail >
			(
					m_functionBlockDeclaration
					, std::cout
					<< phoenix::val("Error! Expecting ")
			<< _4                               // what failed?
			<< phoenix::val(" here: \"")
			<< phoenix::construct<std::string>(_3, _2)   // iterators to error-pos, end
			<< phoenix::val("\"")
			<< std::endl
			);

			qi::on_error< qi::fail >
			(
					m_inputOutputDeclarations,
					std::cout
					<< phoenix::val("Error! Expecting ")
			<< _4                               // what failed?
			<< phoenix::val(" here: \"")
			<< phoenix::construct<std::string>(_3, _2)   // iterators to error-pos, end
			<< phoenix::val("\"")
			<< std::endl
			);
			qi::on_error< qi::fail >
			(
					m_inputVariableDeclaration,
					std::cout
					<< phoenix::val("Error! Expecting ")
			<< _4                               // what failed?
			<< phoenix::val(" here: \"")
			<< phoenix::construct<std::string>(_3, _2)   // iterators to error-pos, end
			<< phoenix::val("\"")
			<< std::endl
			);
		}

		/** \cond */
		qi::rule<Iterator, VariableName()> m_identifier;
		qi::rule<Iterator, NumericLiteral(), ascii::space_type> m_numericLiteral;

		qi::rule<Iterator, WeightingFactor(), ascii::space_type> m_weightingFactor;
		qi::rule<Iterator, Conclusion(), ascii::space_type> m_conclusion;
		qi::rule<Iterator, Condition(), ascii::space_type> m_condition;
		qi::rule<Iterator, SubCondition(), ascii::space_type> m_subcondition;
		qi::rule<Iterator, Rule(), ascii::space_type> m_rule;
		qi::rule<Iterator, AccumulationMethod(), ascii::space_type> m_accumulationMethod;
		qi::rule<Iterator, ActivationMethod(), ascii::space_type> m_activationMethod;
		qi::rule<Iterator, OperatorDefinition(), ascii::space_type> m_operatorDefinition;
                qi::rule<Iterator, NumericRange(), ascii::space_type > m_range;
		qi::rule<Iterator, DefaultValue(), ascii::space_type > m_defaultValue;
		qi::rule<Iterator, DefuzzificationMethod(), ascii::space_type > m_defuzzificationMethod;
		qi::rule<Iterator, Points(), ascii::space_type > m_points;
		qi::rule<Iterator, Singleton(), ascii::space_type > m_singleton;
		qi::rule<Iterator, MemberShipFunction(), ascii::space_type > m_membershipFunction;
                qi::rule<Iterator, LinguisticTermDescription(), ascii::space_type > m_linguisticTerm;
		qi::rule<Iterator, RuleBlock(), ascii::space_type > m_ruleBlock;
		qi::rule<Iterator, DefuzzifyBlock(), ascii::space_type > m_defuzzifyBlock;
		qi::rule<Iterator, FuzzifyBlock(), ascii::space_type > m_fuzzifyBlock;
		qi::rule<Iterator, FunctionBlockBody(), ascii::space_type > m_functionBlockBody;
		qi::rule<Iterator, std::vector< Variable< tag::Input > >(), ascii::space_type > m_inputVariableDeclaration;
		qi::rule<Iterator, std::vector< Variable< tag::Output > >(), ascii::space_type > m_outputVariableDeclaration;
		qi::rule<Iterator, InputOutputDeclarations(), ascii::space_type > m_inputOutputDeclarations;
		qi::rule<Iterator, FunctionBlockDeclaration(), ascii::space_type > m_functionBlockDeclaration;

		qi::symbols< char, detail::VariableType > m_variableType;
		qi::symbols< char, detail::ConnectorType > m_connectorType;
		qi::symbols< char, detail::OrOperatorType > m_orOperatorType;
		qi::symbols< char, detail::AndOperatorType > m_andOperatorType;
		qi::symbols< char, detail::ActivationType > m_activationType;
		qi::symbols< char, detail::AccumulationType > m_accumulationType;
		qi::symbols< char, detail::DefuzzificationType > m_defuzzificationType;
		/** \endcond */
	};


}

#endif
