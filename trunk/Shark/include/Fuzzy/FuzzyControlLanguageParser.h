//===========================================================================
/*!
 *  \file FuzzyControlLanguageParser.h
 *
 *  \brief Parser for the Fuzzy Control Language (see http://en.wikipedia.org/wiki/Fuzzy_Control_Language)
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
//===========================================================================

#ifndef FUZZYCONTROLLANGUAGEPARSER_H
#define FUZZYCONTROLLANGUAGEPARSER_H

/*! \brief Parser for the Fuzzy Control Language (see http://en.wikipedia.org/wiki/Fuzzy_Control_Language) */
template<typename CharType>
struct FuzzyControlLanguageParserBase {
	
	FuzzyControlLanguageParserBase() : mep_currentState( 0 ),
	m_defaultState( *this ),
	m_functionBlockState( *this ),
	m_inputVariableState( *this ),
	m_outputVariableState( *this ),
	m_fuzzifyState( *this ),
	m_defuzzifyState( *this ),
	m_ruleBlockState( *this ),
	m_ruleState( *this ),
	m_operatorState( *this ),
	m_termState( *this ) {
		m_defaultState.activate();
	}
	
	/*! \brief Base class for parser states. */
	struct State {
		State( FuzzyControlLanguageParserBase<CharType> & parent ) : m_parent( parent ) {}
		virtual ~State() {}
		
		virtual void activate() {
			m_parent.mep_currentState = this;
		}
		
		virtual bool operator()( const CharType & c ) = 0;		
		
		FuzzyControlLanguageParserBase<CharType> & m_parent;
	};
	
	template<typename IteratorType>
	bool operator()( IteratorType it, IteratorType itE ) {
		while( it != itE ) {
			if( !(*mep_currentState)( *it ) )
				return( false );
			++it;
		}
		
		return( true );
	}
	
	State * mep_currentState;
	std::string m_buffer;
	
	
	/*! \brief The default parser state. */
	struct DefaultState : public State {
		DefaultState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		bool operator()( const CharType & c ) {
			
			m_buffer.push_back( c );
			
			if( m_buffer == "FUNCTION_BLOCK" ) {
				m_buffer.clear();
				m_parent.m_functionBlockState.activate();
			}
			
			return( true );
		}
		
	} m_defaultState;
	
	/*! \brief Parser state that handles function block definitions. */
	struct FunctionBlockState : public State {
		FunctionBlockState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		bool operator()( const CharType & c ) {
			
			if( m_parent.m_buffer == "VAR_INPUT" )
				m_parent.m_inputVariableState.activate();
			else if( m_parent.m_buffer == "FUZZIFY" )
				m_parent.m_fuzzifyState.activate();
			else if( m_parent.m_buffer == "DEFUZZIFY" )
				m_parent.m_defuzzifyState.activate();
			else if( m_parent.m_buffer == "RULEBLOCK" )
				m_parent.m_ruleBlockState.activate();
			else
				m_parent.m_buffer.push_back( c );
			
			return( true );
		}
	} m_functionBlockState;
	
	/*! \brief Parser state that handles input variable definitions. */
	struct InputVariableState : public State {
		InputVariableState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		void activate() {
			m_parent.mep_currentState = this;
			m_parent.m_buffer.clear();
		}
		
		bool operator()( const CharType & c ) {
			
			if( c == ':' ) {
				m_variableName = m_parent.m_buffer;
				m_parent.m_buffer.clear();
			} else if( c == ';' ) {
				
				// Add variable here
				m_parent.m_buffer.clear();
				m_variableName.clear();
				
			} else if( m_buffer == "END_VAR" ) 
				m_parent.m_functionBlockState.activate();
			else
				m_parent.m_buffer.push_back( c );
				
			return( true );
		}
		
		std::string m_variableName;
	} m_inputVariableState;
	
	/*! \brief Parser state that handles output variable definitions. */
	struct OutputVariableState : public State {
		OutputVariableState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		void activate() {
			m_parent.mep_currentState = this;
			m_parent.m_buffer.clear();
		}
		
		bool operator()( const CharType & c ) {
			
			if( c == ':' ) {
				m_variableName = m_parent.m_buffer;
				m_parent.m_buffer.clear();
			} else if( c == ';' ) {
				
				// Add variable here
				m_parent.m_buffer.clear();
				m_variableName.clear();
				
			} else if( m_buffer == "END_VAR" ) 
				m_parent.m_functionBlockState.activate();
			else
				m_parent.m_buffer.push_back( c );
			
			return( true );
		}
	} m_outputVariableState;
	
	/*! \brief Parser state that handles fuzzify-blocks. */
	struct FuzzyifyState : public State {
		FuzzyifyState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		void activate() {
			m_parent.m_buffer.clear();
			m_variableName.clear();
			m_termName.clear();
		}
		
		bool operator()( const CharType & c ) {
			
			if( c == '\n' ) {
				m_variableName = m_parent.m_buffer;
			} else if( c == ':' ) {
				m_termName = m_parent.m_buffer;
				m_parent.m_buffer.clear();
			} else if( c == ';' ) {
				m_variableName.clear();
				m_termName.clear();
			}else if( m_parent.m_buffer == "TERM" ) {
				m_parent.m_buffer.clear();
			} 
			
			return( true );
		}
		
		std::string m_variableName;
		std::string m_termName;
	} m_fuzzifyState;
	
	/*! \brief Parser state that handles defuzzify states. */
	struct DefuzzyifyState : public State {
		DefuzzyifyState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		bool operator()( const CharType & c ) {
			return( true );
		}
	} m_defuzzifyState;
	
	/*! \brief Parser state that handles rule blocks. */
	struct RuleBlockState : public State {
		RuleBlockState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		bool operator()( const CharType & c ) {
			return( true );
		}
	} m_ruleBlockState;
	
	/*! \brief Parser state that handles rule definitions. */
	struct RuleState : public State {
		RuleState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		bool operator()( const CharType & c ) {
			return( true );
		}
	} m_ruleState;
	
	/*! \brief Parser state that handles operator definitions. */
	struct OperatorState : public State {
		OperatorState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		bool operator()( const CharType & c ) {
			return( true );
		}
	} m_operatorState;
	
	/*! \brief Parser state that handles term definitions. */
	struct TermState : public State {
		TermState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		bool operator()( const CharType & c ) {
			return( true );
		}
	} m_termState;
};

typedef FuzzyControlLanguageParserBase<char> FuzzyControlLanguageParser;

#endif // FUZZYCONTROLLANGUAGEPARSER_H