#ifndef SHARK_FUZZY_CONTROL_LANGUAGE_PARSER_FUSION_BINDINGS_H
#define SHARK_FUZZY_CONTROL_LANGUAGE_PARSER_FUSION_BINDINGS_H

#include <shark/Fuzzy/FCL/FuzzyControlLanguageParser.h>

#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/io.hpp>

#include <boost/variant.hpp>

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    BOOST_FUSION_ADAPT_STRUCT(
                  shark::Variable< shark::tag::Input >,
                  ( shark::VariableName, m_name )
                  ( shark::detail::VariableType, m_type )
                   );

    BOOST_FUSION_ADAPT_STRUCT(
                  shark::Variable< shark::tag::Output >,
                  ( shark::VariableName, m_name )
                  ( shark::detail::VariableType, m_type )
                   );

    BOOST_FUSION_ADAPT_STRUCT (
                   shark::InputOutputDeclarations,
                   ( shark::InputDeclarations, m_inputDeclarations )
                   ( shark::OutputDeclarations, m_outputDeclarations )
                   );

    BOOST_FUSION_ADAPT_STRUCT(
                  shark::Point,
                  ( shark::Point::Coordinate, m_x )
                  ( shark::NumericLiteral, m_y )
                   )

    BOOST_FUSION_ADAPT_STRUCT(
                  shark::Conclusion,
                  ( shark::VariableName, m_var )
                  ( boost::optional< shark::VariableName >, m_term )
                   )

    BOOST_FUSION_ADAPT_STRUCT(
                  shark::Rule,
                  ( int, m_id )
                  ( shark::Condition, m_condition )
                  ( shark::Conclusion, m_conclusion )
                  ( boost::optional< shark::WeightingFactor >, m_weightingFactor )
                   )

    BOOST_FUSION_ADAPT_STRUCT(
                  shark::SubCondition,
                  ( boost::optional< std::string >, m_variablePrefix )
                  ( shark::VariableName, m_variable )
                  ( boost::optional< std::string >, m_termPrefix )
                  ( shark::VariableName, m_term )
                   )

    BOOST_FUSION_ADAPT_STRUCT(
                  shark::ConditionClause,
                  ( shark::detail::ConnectorType, m_connector )
                  ( shark::ConditionClause::VariantType, m_t )
                   )
    BOOST_FUSION_ADAPT_STRUCT(
                  shark::Condition,
                  ( shark::ConditionClause::VariantType, m_firstClause )
                  ( std::vector< shark::ConditionClause >, m_tailClauses )
                   )
    /*BOOST_FUSION_ADAPT_STRUCT(
                  shark::AccumulationMethod,
                  ( std::string, m_type )
                   )

    BOOST_FUSION_ADAPT_STRUCT(
                  shark::ActivationMethod,
                  ( std::string, m_type )
                   )

    BOOST_FUSION_ADAPT_STRUCT(
                  shark::DefuzzificationMethod,
                  ( std::string, m_type )
                  )*/

    BOOST_FUSION_ADAPT_STRUCT(
                              shark::LinguisticTermDescription,
                  ( shark::TermName, m_name )
                  ( shark::MemberShipFunction, m_msf )
                   )

    BOOST_FUSION_ADAPT_STRUCT(
                  shark::RuleBlock,
                  ( shark::VariableName, m_name )
                  ( shark::OperatorDefinition, m_operatorDefinition )
                  ( boost::optional< shark::ActivationMethod >, m_activationMethod )
                  ( shark::AccumulationMethod, m_accumulationMethod )
                  ( std::vector< shark::Rule >, m_rules )
                   )

    BOOST_FUSION_ADAPT_STRUCT(
                  shark::DefuzzifyBlock,
                  ( shark::VariableName, m_name )
                              ( std::vector< shark::LinguisticTermDescription >, m_terms )
                  ( shark::DefuzzificationMethod, m_defuzzificationMethod )
                  ( shark::DefaultValue, m_defaultValue )
                              ( boost::optional< shark::NumericRange >, m_range )
                   )

    BOOST_FUSION_ADAPT_STRUCT(
                  shark::FuzzifyBlock,
                  ( shark::VariableName, m_variableName )
                              ( std::vector< shark::LinguisticTermDescription >, m_terms )
                   )

    BOOST_FUSION_ADAPT_STRUCT(
                  shark::FunctionBlockBody,
                  ( std::vector< shark::FuzzifyBlock >, m_fuzzifyBlocks )
                  ( std::vector< shark::DefuzzifyBlock >, m_defuzzifyBlocks )
                  ( std::vector< shark::RuleBlock >, m_ruleBlocks )
                   )

    BOOST_FUSION_ADAPT_STRUCT(
                  shark::FunctionBlockDeclaration,
                  ( shark::VariableName, m_name )
                  ( shark::InputOutputDeclarations, m_ioDeclarations )
                  //( std::vector< shark::InputDeclarations >, m_inputDeclarations )
                  //( std::vector< shark::OutputDeclarations >, m_outputDeclarations )
                  ( shark::FunctionBlockBody, m_functionBlockBody )
                   )

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

#endif
