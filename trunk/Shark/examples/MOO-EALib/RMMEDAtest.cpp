#include <ctime>
#include <cmath>
#include <fstream>
#include <vector>
#include <MOO-EALib/PopulationMOO.h>
#include <Rng/GlobalRng.h>
#include <Array/Array.h>
#include <Array/ArrayIo.h>
#include <MOO-EALib/RMMEDA.h>

// parameters
// unsigned int PopSize,
// 			 DimX,
// 			 DimF;



void computeFitnessRoot(PopulationMOO & X){
  double g = 0.0, f1,f2;
  unsigned int i;

  for (unsigned int p=0;p<X.size();p++){
  g=0.0;
  for(i=2; i<X[0][0].size(); i+=2){
    g +=  (dynamic_cast<ChromosomeT<double>&>(X[p][0])[i]*dynamic_cast<ChromosomeT<double>&>(X[p][0])[i] 
	   - dynamic_cast<ChromosomeT<double>&>(X[p][0])[0])*
      (dynamic_cast<ChromosomeT<double>&>(X[p][0])[i]*dynamic_cast<ChromosomeT<double>&>(X[p][0])[i]
       - dynamic_cast<ChromosomeT<double>&>(X[p][0])[0]);//(X[i] - X[0])*(X[i] - X[0]);
  }
  g = 1.0 + 9.0*g/(X[0][0].size() - 1.0);
  f1 = dynamic_cast<ChromosomeT<double>&>(X[p][0])[0] + g - 1.0;
  f2 = 1.0 - sqrt(dynamic_cast<ChromosomeT<double>&>(X[p][0])[0]) + g - 1.0;
  X[p].setMOOFitnessValues(f1,f2);
  }
}

void computeFitnessSquared(PopulationMOO & X){
  double g = 0.0, f1,f2;
  unsigned int i;
  for (unsigned p=0;p<X.size();p++){
    g=0.0;
    for(i=2; i<X[0][0].size(); i+=2)
      g +=  (dynamic_cast<ChromosomeT<double>&>(X[p][0])[i]*dynamic_cast<ChromosomeT<double>&>(X[p][0])[i] 
	     - dynamic_cast<ChromosomeT<double>&>(X[p][0])[0])
	*(dynamic_cast<ChromosomeT<double>&>(X[p][0])[i]*dynamic_cast<ChromosomeT<double>&>(X[p][0])[i]
	  - dynamic_cast<ChromosomeT<double>&>(X[p][0])[0]);//(X[i] - X[0])*(X[i] - X[0]);
    g = 1.0 + 9.0*g/(X[0][0].size() - 1.0);
    f1 = dynamic_cast<ChromosomeT<double>&>(X[p][0])[0] + g - 1.0;
    f2 = 1.0 - dynamic_cast<ChromosomeT<double>&>(X[p][0])[0]*dynamic_cast<ChromosomeT<double>&>(X[p][0])[0] + g - 1.0;
    X[p].setMOOFitnessValues(f1,f2);
  }
}


int main()
{
	unsigned int i,s;
	unsigned int Dim     =    30; // dimension of chromosomes
	unsigned int PopSize =    20; // population size
	unsigned int fitness =     2; // 1 for root and 2 for squared version
	Rng::seed(0);
	Array<Array<double> > SearchBound(2);
	PopSize = 20; 
	PopulationMOO parents(PopSize, ChromosomeT< double >(Dim));
	PopulationMOO offsprings(PopSize, ChromosomeT< double >(Dim));
	parents.setAscending(1);
	offsprings.setAscending(1);
	parents.setNoOfObj( 2 );
	offsprings.setNoOfObj( 2 );
	SearchBound(0).resize(Dim); SearchBound(1).resize(Dim);
	SearchBound(0) = 0.0; SearchBound(1) = 1.0;
	for(i=0; i<PopSize; i++) {
	  for(s=0;s<Dim;s++) {
	    dynamic_cast< ChromosomeT< double >& >(parents[i][0])[s] = Rng::uni(SearchBound(0)(s), SearchBound(1)(s)); 
	  }
	}
	if (fitness==1)
	  computeFitnessRoot(parents);
	if (fitness==2)
	  computeFitnessSquared(parents);

	unsigned int clu;
	az::mea::gen::mod::RM rmx;
	az::mea::sel::NDS sel;
	for(unsigned int gen=0; gen<100; gen++)
	{
		clu = 2;// +(unsigned int)(gen/50.0);
		rmx.Generate(1, clu, 50, 0.15, SearchBound(0), SearchBound(1), PopSize, offsprings, parents);
		  if (fitness==1)
		      computeFitnessRoot(offsprings);
		  if (fitness==2)
		      computeFitnessSquared(offsprings);
		  //		  parents.selectRMMuPlusLambda(parents,offsprings);
		  parents.selectRMMuPlusLambda(offsprings);
		  std::cout<<gen<<" "<<offsprings[0].getMOOFitness(0)<<" "<<offsprings[0].getMOOFitness(1)<<std::endl;
	}
	//	SavePop();
	return 0;
}	
