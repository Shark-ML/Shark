
/**
*
* \brief Fuzzy inference machine.
* 
* \authors Marc Nunkesser
*/

/* $log$ */

#ifndef INFERENCEMACHINE_H
#define INFERENCEMACHINE_H

#include <shark/LinAlg/Base.h>

#include <shark/Fuzzy/RuleBase.h>
#include <shark/Fuzzy/FuzzySet.h>
#include <shark/Fuzzy/FuzzySets/TrapezoidFS.h>

#include <boost/shared_ptr.hpp>

#include <iostream>
#include <fstream>
#include <vector>

namespace shark {

/**
 * \brief An inference machine.
 *
 * A virtual basis class for the different inference machines.
 */
class InferenceMachine {
public:
    typedef std::vector< boost::shared_ptr< FuzzySet > > OutputType;
    typedef RealVector InputType;

    /**
  * \brief Constructor.
  *
  * @param rb the associated rulebase.
  */
    InferenceMachine( const boost::shared_ptr<RuleBase> & rb = boost::shared_ptr<RuleBase>() ) : mep_ruleBase( rb ) {}

    /**
  * \brief Destructor
  */
    virtual ~InferenceMachine() {}

    /**
  * \brief Computes the inference.
  *
  *  @param it a vector of crisp values (an InputType)
  *  @return a vector of fuzzy sets
  */
    virtual OutputType computeInference( const InputType & it ) const {
        boost::shared_ptr< TrapezoidFS > trap;
        RuleBase::rule_set_iterator first = mep_ruleBase->ruleSetBegin();
        OutputType Result= buildTreeFast( first,
                                          mep_ruleBase->numberOfRules()-1,
                                          ((*first)->conclusion()).size(),
                                          it);
        unsigned int i = 0;
        for( RuleBase::output_type_iterator f= mep_ruleBase->conclusionsBegin(); f != mep_ruleBase->conclusionsEnd(); ++f, ++i ) {
            //std::cerr << "assess result " << i << " " << Result.size() << " no rules:" <<  mep_ruleBase->numberOfRules() << " " << ((*first)->conclusion()).size() << std::endl;
            //if (!Result.size()) continue; // if commented out, no segmentation fault occurs, but this cannot be the solution
            if (!(Result[i])) continue;
            trap.reset(new TrapezoidFS((*f)->lowerBound(),
                                       (*f)->lowerBound(),
                                       (*f)->upperBound(),
                                       (*f)->upperBound()));
            Result[i].reset(
                        new ComposedFS(
                            ComposedFS::MIN,
                            Result[i],
                            trap
                            )
                        );
        };
        return(Result);
    }

    /**
  * \brief Computes the inference.
  *
  *  @param a a crisp value
  *  @return a vector of fuzzy sets
  */
    virtual OutputType computeInference(double a)  const {
        RealVector v( 1 );
        v(0) = a;
        return( computeInference( v ) );
    }

    /**
  * \brief Computes the inference.
  *
  *  @param a first crisp value
  *  @param b second crisp value
  *  @return a vector of fuzzy sets
  */
    virtual OutputType computeInference(double a, double b)  const {
        RealVector v( 2 );
        v[0] = a;
        v[1] = b;
        return( computeInference( v ) );
    }

    /**
  * \brief Computes the inference.
  *
  *  @param a first crisp value
  *  @param b second crisp value
  *  @param c thrid crisp value
  *  @return a vector of fuzzy sets
  */
    virtual OutputType computeInference(double a, double b, double c)  const {
        RealVector v( 3 );
        v(0) = a;
        v(1) = b;
        v(2) = c;
        return( computeInference( v ) );
    }

    /**
  * \brief Computes the inference.
  *
  *  @param a first crisp value
  *  @param b second crisp value
  *  @param c thrid crisp value
  *  @param d forth crisp value
  *  @return a vector of fuzzy sets
  */
    virtual OutputType computeInference(double a, double b, double c, double d)  const {
        RealVector v( 4 );

        v(0) = a;
        v(1) = b;
        v(2) = c;
        v(3) = d;

        return( computeInference( v ) );
    }

    /**
  * \brief Set the associated rule base.
  *
  * @param rb the rule base
  */
    inline void setRuleBase( const boost::shared_ptr<RuleBase> & rb ) {
        mep_ruleBase = rb;
    }

    /**
  * \brief Plots the defuzzification results for the whole input range into a
  * gnuplot-suited file.
  *
  * For evaluation the borders of the linguistic variables are taken into
  * consideration.
  *
  * @param fileName name of the file where the data of characteristic curve is saved
  * @param resolution the resolution of the characterestic curve
  */
    void characteristicCurve( const std::string fileName = "curve.dat", long int resolution = 50 ) const {

        double lowerI,upperI,lowerJ,upperJ;
        double stepI,stepJ;
        double i,j;
        RuleBase::input_type_iterator fi;

        std::ofstream dataFile( fileName.c_str(),std::ios::out );
        if (!dataFile)
            throw( SHARKEXCEPTION( "Cannot write to disk" ) );

        switch( mep_ruleBase->numberOfInputs() )
        {
        case 1:
            //std::cerr << "1D input" << std::endl;
            fi = mep_ruleBase->formatBegin();
            lowerI=(*fi)->lowerBound();
            upperI=(*fi)->upperBound();
            assert(lowerI<upperI);
            stepI = (upperI-lowerI)/resolution;
            for (i = lowerI;i<= upperI;i+=stepI) {
                addToFile(i,dataFile);
            };
            break;
        case 2:
            //std::cerr << "2D input" << std::endl;
            fi = mep_ruleBase->formatBegin();
            lowerI=(*fi)->lowerBound();
            upperI=(*fi)->upperBound();
            assert(lowerI<upperI);
            stepI = (upperI-lowerI)/resolution;
            fi++;
            lowerJ=(*fi)->lowerBound();
            upperJ=(*fi)->upperBound();
            stepJ = (upperJ-lowerJ)/resolution;
            assert((stepJ>0)&&(stepI>0));
            for (i = lowerI;i<= upperI;i+=stepI) {
                for (j= lowerJ; j<= upperJ;j+=stepJ) {
                    //std::cerr << i << " " << j << std::endl;
                    addToFile(i,j,dataFile);
                };
                dataFile << std::endl;
            };
            break;
        default:
            throw( SHARKEXCEPTION( "Dimension of rules make visualization impossible" ) );
            break;
        }

    }
protected:
    boost::shared_ptr<RuleBase> mep_ruleBase;

    virtual OutputType buildTreeFast( RuleBase::rule_set_iterator & actual, unsigned int remainingRules, int conclusionNumber, const InputType in) const = 0;
private:

    virtual void addToFile(double i, std::ofstream & dataFile ) const {
        RealVector v( 1 );
        v(0) = i;
        dataFile << i << " " << (computeInference(v)[0])->defuzzify() << std::endl;
    }

    virtual void addToFile(double i, double j, std::ofstream & dataFile ) const {
        RealVector v( 2 );

        v(0) = i;
        v(1) = j;

        //std::cerr << "compute inference" << std::endl;
        OutputType fs = computeInference(v);

        if(!fs.size()) return; // Because of error on example - this is not a solution!!!!
        //std::cerr << "output size: " << std::endl;
        //std::cerr << fs.size() << std::endl;
        //std::cerr << "defuzzyfy" << std::endl;

        double d = (fs[ 0 ])->defuzzify();

        //std::cerr << "write" << std::endl;

        dataFile << i << " " << j << " " << d << std::endl;
    }
};
}
#endif
