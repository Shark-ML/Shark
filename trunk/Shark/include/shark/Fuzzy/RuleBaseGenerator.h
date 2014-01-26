
/**
 *
 * \brief Reads and writes rule bases from and to XML-files.
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */

#ifndef __RULEBASEGENERATOR_H__
#define __RULEBASEGENERATOR_H__

#include <shark/Fuzzy/RuleBase.h>

/**
 * \brief Reads a RuleBase from XML file
 * 
 * @param descriptionFile the XML file containing the RuleBase
 * @return the read RuleBase
 */
RuleBase build_rule_base_from_xml( const std::string & descriptionFile );

/**
 * \brief Writes a RuleBase into a XML file
 * 
 * @param descriptionFile the output file
 * @param rb the RuleBase to write
 * @return true, in case of success
 */
bool save_rule_base_to_xml( const std::string & descriptionFile, RuleBase & rb );

// Building a rule-base from fcl.
//RuleBase build_rule_base_from_fcl( const std::string & descriptionFile );
#endif
