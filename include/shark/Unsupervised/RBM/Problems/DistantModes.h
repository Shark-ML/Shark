#ifndef UNSUPERVISED_RBM_PROBLEMS_DISTANTMODES_H
#define UNSUPERVISED_RBM_PROBLEMS_DISTANTMODES_H

#include  <shark/Data/Dataset.h>
#include  <shark/LinAlg/Base.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Models/AbstractModel.h>

namespace shark{


///\brief Creates a set of pattern (each later representing the "center" of a mode)
/// which than are randomly perturbed to create the data set.
///\todo add reference 
///
///The higher the perturbation is the harder it is to classify, 
///but the closer are the modes and thus the easier the data distribution is to learn.
class DistantModes{
private:
	UnlabeledData<RealVector> m_data;
	
	double m_p;
	unsigned m_dim;
	unsigned m_modes;
	unsigned m_copies;
	std::size_t m_batchSize;
	
	//Genererates a basic pattern representing the "center" of a mode.
	void modePrototype(RealVector& pattern, unsigned mode) const {
		for (std::size_t i = 0; i != pattern.size(); ++i){
			pattern(i) = (mode % 2) ^ (i * (mode / 2 + 1) / pattern.size()) % 2;
		}
	}


	///Perturbates the pattern by randomly flipping pixels
	///@param pattern the pattern
	///@param p the flipping probability
	void perturbate(RealVector& pattern, double p)const{
		for (std::size_t i = 0; i < pattern.size(); ++i){
			if (Rng::uni(0,1) > p){
				pattern(i) = !pattern(i);
			}
		}
	}
	
	void init() {
		std::vector<RealVector> data(m_modes * m_copies,RealVector(m_dim));
		for (std::size_t i = 0; i != data.size(); ++i) {
			RealVector& element=data[i];
			unsigned mode = i % m_modes;
			modePrototype(element, mode);
			perturbate(element, m_p);
		}
		m_data = createDataFromRange(data, m_batchSize);
	}
	
public:
	///generates the DistantModes distribution. It can later be changed using configure
	///
	///\param p the probability of changing a inputneuron
	///\param dim the dimensionality of the data.
	///\param modes the number of modes, should be a multiple of 2
	///\param copies the number of disturbed copies for each mode
	///\param batchSize the size of the batches in which the genereated data set is organized
	DistantModes(double p = 0, unsigned dim = 16, unsigned modes=4, unsigned copies =2500, size_t batchSize=0)
		:m_p(p), m_dim(dim), m_modes(modes), m_copies(copies), m_batchSize(batchSize) {
		init();
	}
	
	///configure needs the following 4 properties:
	///p: the pro pability of disturbance
	///dimension: input dimension of the data. default is 16
	///modes: number of basic mdoes. Default is 4
	///copies: number of disturbed copies of a mode. default is 2500
	void configure( const PropertyTree & node ) {
		m_p = node.get<double>("p");
		m_dim = node.get("dimension",16);
		m_modes = node.get("modes",4);
		m_copies = node.get("copies",2500);
		init();
	}
    
	///returns the generated dataset
	UnlabeledData<RealVector> data() const{
		return m_data;
	};
	
	///returns the dimensionality of the data
	std::size_t inputDimension() const {
		return m_dim;
	}
};

}
#endif
