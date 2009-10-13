#ifndef FUZZYCONTROLLANGUAGEPARSER_H
#define FUZZYCONTROLLANGUAGEPARSER_H

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
	
	struct DefuzzyifyState : public State {
		DefuzzyifyState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		bool operator()( const CharType & c ) {
			return( true );
		}
	} m_defuzzifyState;
	
	struct RuleBlockState : public State {
		RuleBlockState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		bool operator()( const CharType & c ) {
			return( true );
		}
	} m_ruleBlockState;
	
	struct RuleState : public State {
		RuleState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		bool operator()( const CharType & c ) {
			return( true );
		}
	} m_ruleState;
	
	struct OperatorState : public State {
		OperatorState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		bool operator()( const CharType & c ) {
			return( true );
		}
	} m_operatorState;
	
	struct TermState : public State {
		TermState( FuzzyControlLanguageParserBase<CharType> & parent ) : State( parent ) {
		};
		
		bool operator()( const CharType & c ) {
			return( true );
		}
	} m_termState;
};

typedef FuzzyControlLanguageParserBase<char> FuzzyControlLanguageParser;

int main( int argc, char ** argv ) {
	FuzzyControlLanguageParser parser;
}

#endif // FUZZYCONTROLLANGUAGEPARSER_H