
//!
//! A cover tree is a data structure forstering the fast
//! computation of nearest neighbor queries. The algorithm
//! is described in:
//!
//! Alina Beygelzimer, Sham M. Kakade, and John Langford.
//! Cover Trees for Nearest Neighbor. ICML 2006.
//!


#ifndef _CoverTree_H_
#define _CoverTree_H_


#include <Array/Array.h>
#include <ReClaM/KernelFunction.h>


//!
//! \brief Cover tree data structure for fast nearest neighbors
//!
//! \par
//! A cover tree is a data structure for fast processing
//! of nearest neighbor queries. The algorithm is
//! described in:
//!
//! \par
//! Alina Beygelzimer, Sham M. Kakade, and John Langford.
//! <i>Cover Trees for Nearest Neighbor</i>. ICML 2006.
//!
class CoverTree
{
public:
	//! Construct a cover tree for the given set of points.
	CoverTree(const Array<double>& points);

	//! Destructor
	virtual ~CoverTree();


	//! This method computes the nearest neighbor of an arbitrary
	//! query point (one that was not used to construct the tree).
	//! Returns the index of the nearest neighbor.
	unsigned int OneNN(const Array<double>& query) const;

	//! This method computes the nearest neighbor of a point that
	//! was used to construct the tree, but not the point itself.
	//! Returns the index of the nearest neighbor.
	unsigned int SecondNN(unsigned int query) const;

	//! This method computes the k nearest neighbors of an arbitrary
	//! query point (one that was not used to construct the tree).
	//! Returns the induces of the nearest neighbors.
	//!
	//! \param  query      query point
	//! \param  k          number of neighbors
	//! \param  neighbors  list of neighbor indices
	void KNN(const Array<double>& query, unsigned int k, std::vector<unsigned int>& neighbors) const;

	//! This method returns the point corresponding to an
	//! index as returned by the different query methods.
	inline const ArrayReference<double> Point(unsigned int index) const
	{
		return (*points)[index];
	}

	//! \brief compute the squared distance
	//!
	//! \par
	//! This method computes the squared distance
	//! between p1 and p2. Override this method to
	//! change the distance computation.
	virtual double distance2(const Array<double>& p1, const Array<double>& p2) const;

	inline double distance(const Array<double>& p1, const Array<double>& p2) const
	{
		return sqrt(distance2(p1, p2));
	}

protected:
	struct Node
	{
		unsigned int index;
		Node* firstchild;
		Node* nextsibling;
		int scale;
	};

	struct tNode
	{
		Node* node;
		double dist;
	};

	inline Node* freeNode()
	{
		Node* ret = &mem[usedmem];
		usedmem++;
		return ret;
	}

	void Insert(unsigned int index);

	const Array<double>* points;
	Node* root;
	std::vector<Node> mem;
	unsigned int usedmem;

	friend bool cmp(const tNode& n1, const tNode& n2);
};


//!
//! \brief Cover tree with kernel distances
//!
class KernelCoverTree : public CoverTree
{
public:
	//! Construct a cover tree for the given set of points
	//! using the kernel induced distance measure.
	KernelCoverTree(KernelFunction* kernel, const Array<double>& points);

	//! Destructor
	~KernelCoverTree();


	//! distance computation based on the kernel function
	double distance2(const Array<double>& p1, const Array<double>& p2) const;

protected:
	KernelFunction* kernel;
};


#endif
