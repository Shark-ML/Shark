
#include <ReClaM/CoverTree.h>


#define POINTSCALE -1022


union IntWiseDouble
{
	double d;
	int i[2];
};

// return 2^(exponent+1)
double i_pow2(int exponent)
{
	IntWiseDouble ret;
	ret.d = 2.0;
	ret.i[1] += (exponent << 20);
	return ret.d;
// 	return pow(2.0, exponent + 1.0);
}

int i_log2(double dist2)
{
	IntWiseDouble value;
	value.d = dist2;
	return (((value.i[1] & 0x7ff00000) >> 20) - 1023) >> 1;
// 	return (int)floor(0.5 * log(dist2) / log(2.0));
}


////////////////////////////////////////////////////////////


CoverTree::CoverTree(const Array<double>& points)
{
	this->points = &points;

	unsigned int i, ic = points.dim(0);
	mem.resize(2*ic);			// upper bound
	usedmem = 0;

	// create a minimal tree of one element
	root = freeNode();
	root->index = 0;
	root->firstchild = NULL;
	root->nextsibling = NULL;
	root->scale = POINTSCALE;

	// insert the remaining points
	for (i=1; i<ic; i++) Insert(i);
}

CoverTree::~CoverTree()
{
}


unsigned int CoverTree::OneNN(const Array<double>& query) const
{
	tNode n;
	std::vector<tNode> list1;
	std::vector<tNode> list2;
	std::vector<tNode>* cand = &list1;
	std::vector<tNode>* refine = &list2;
	double best_d = distance((*points)[root->index], query);
	n.node = root;
	n.dist = best_d;
	cand->push_back(n);
	int s = root->scale;

	// iteratively refine the set
	while (true)
	{
		// compute shortest distance to all children
		int i, ic = cand->size();

		// refine to next scale
		bool children = false;
		for (i=0; i<ic; i++)
		{
			tNode& nr = (*cand)[i];
			if (nr.node->scale >= s - 1)
			{
				// loop through sub-nodes
				Node* child = nr.node->firstchild;
				while (child != NULL)
				{
					double d = distance((*points)[child->index], query);
					if (d <= best_d + i_pow2(child->scale))
					{
						n.node = child;
						n.dist = d;
						refine->push_back(n);
						children = children || (child->firstchild != NULL);

						if (d < best_d) best_d = d;
					}
					child = child->nextsibling;
				}
			}
			else
			{
				if (nr.dist <= best_d + i_pow2(nr.node->scale))
				{
					refine->push_back(nr);
					children = children || (nr.node->firstchild != NULL);
				}
			}
		}

		// stopping condition
		if (! children) break;

		cand->clear();
		std::swap(refine, cand);
		s--;
	}

	return (*refine)[0].node->index;
}

unsigned int CoverTree::SecondNN(unsigned int query) const
{
	const ArrayReference<double> pt = (*points)[query];
	tNode n;
	std::vector<tNode> list1;
	std::vector<tNode> list2;
	std::vector<tNode>* cand = &list1;
	std::vector<tNode>* refine = &list2;
	double best_d = 1e100;
	n.node = root;
	n.dist = best_d;
	cand->push_back(n);
	int s = root->scale;

	// iteratively refine the set
	while (true)
	{
		// compute shortest distance to all children
		int i, ic = cand->size();

		// refine to next scale
		bool children = false;
		for (i=0; i<ic; i++)
		{
			tNode& nr = (*cand)[i];
			if (nr.node->scale >= s - 1)
			{
				// loop through sub-nodes
				Node* child = nr.node->firstchild;
				while (child != NULL)
				{
					double d = distance((*points)[child->index], pt);
					if (d <= best_d + i_pow2(child->scale))
					{
						n.node = child;
						n.dist = d;
						refine->push_back(n);
						children = children || (child->firstchild != NULL);

						if (d < best_d && child->index != query)
						{
							best_d = d;
						}
					}
					child = child->nextsibling;
				}
			}
			else
			{
				if (nr.dist <= best_d + i_pow2(nr.node->scale))
				{
					refine->push_back(nr);
					children = children || (nr.node->firstchild != NULL);
				}
			}
		}

		// stopping condition
		if (! children) break;

		cand->clear();
		std::swap(refine, cand);
		s--;
	}

	if (refine->size() < 2)
	{
		throw SHARKEXCEPTION("[CoverTree::SecondNN] internal error");
	}
	else if (refine->size() == 2)
	{
		if ((*refine)[0].node->index == query) return (*refine)[1].node->index;
		else return (*refine)[0].node->index;
	}
	else
	{
		double best_d2 = 1e100;
		unsigned int ret = 0;
		int i, ic = refine->size();
		for (i=0; i<ic; i++)
		{
			unsigned int index = (*refine)[i].node->index;
			if (index == query) continue;
			double d2 = distance2((*points)[index], pt);
			if (d2 < best_d2)
			{
				best_d2 = d2;
				ret = index;
			}
		}
		return ret;
	}
}

bool cmp(const CoverTree::tNode& n1, const CoverTree::tNode& n2)
{
	return (n1.dist < n2.dist);
}

void CoverTree::KNN(const Array<double>& query, unsigned int k, std::vector<unsigned int>& neighbors) const
{
	tNode n;
	std::vector<tNode> list1;
	std::vector<tNode> list2;
	std::vector<tNode>* cand = &list1;
	std::vector<tNode>* refine = &list2;
	double kth_best_d = 1e100;
	n.node = root;
	n.dist = kth_best_d;
	cand->push_back(n);
	int s = root->scale;

	// iteratively refine the set
	while (true)
	{
		// compute shortest distance to all children
		int i, ic = cand->size();

		// refine to next scale
		bool children = false;
		for (i=0; i<ic; i++)
		{
			tNode& nr = (*cand)[i];
			if (nr.node->scale >= s - 1)
			{
				// loop through sub-nodes
				Node* child = nr.node->firstchild;
				while (child != NULL)
				{
					double d = distance((*points)[child->index], query);
					if (d <= kth_best_d + i_pow2(child->scale))
					{
						n.node = child;
						n.dist = d;
						refine->push_back(n);
						children = children || (child->firstchild != NULL);
					}
					child = child->nextsibling;
				}
			}
			else
			{
				if (nr.dist <= kth_best_d + i_pow2(nr.node->scale))
				{
					refine->push_back(nr);
					children = children || (nr.node->firstchild != NULL);
				}
			}
		}

		// compute k-th best distance
		std::sort(refine->begin(), refine->end(), cmp);
		if (refine->size() >= k) kth_best_d = (*refine)[k - 1].dist;
		else kth_best_d = 1e100;

		// stopping condition
		if (! children) break;

		cand->clear();
		std::swap(refine, cand);
		s--;
	}

	neighbors.resize(k);
	unsigned int i;
	for (i=0; i<k; i++) neighbors[i] = (*refine)[i].node->index;
}

double CoverTree::distance2(const Array<double>& p1, const Array<double>& p2) const
{
	unsigned int d, dim = p1.dim(0);
	double dist2 = 0.0;
	for (d=0; d<dim; d++) dist2 += Shark::sqr(p2(d) - p1(d));
	return dist2;
}

void CoverTree::Insert(unsigned int index)
{
	const Array<double> point = (*points)[index];
	Node* parent = root;
	double pd2 = distance2((*points)[parent->index], point);
	int ps = i_log2(pd2);

	// check if the current scale is sufficient
	if (ps > parent->scale)
	{
		// create new top level
		parent = freeNode();
		parent->index = root->index;
		parent->scale = ps;
		parent->firstchild = root;
		parent->nextsibling = NULL;

		Node* node = freeNode();
		node->index = index;
		node->firstchild = NULL;
		node->nextsibling = NULL;
		node->scale = POINTSCALE;

		root->nextsibling = node;
		root = parent;

		return;
	}

	// find the parent
	while (true)
	{
		// find the closest child
		Node* closest = parent->firstchild;
		if (closest == NULL)
		{
			throw SHARKEXCEPTION("[CoverTree::Insert] internal error");
		}

		double cd2 = distance2((*points)[closest->index], point);
		Node* child = closest->nextsibling;
		while (child != NULL)
		{
			double d2 = distance2((*points)[child->index], point);
			if (d2 < cd2)
			{
				closest = child;
				cd2 = d2;
			}
			child = child->nextsibling;
		}
		int cs = i_log2(cd2);

		// check whether we are already at the right level
		if (cs == parent->scale)
		{
			// insert into existing sibling structure
			Node* node = freeNode();
			node->index = index;
			node->scale = POINTSCALE;
			node->firstchild = NULL;
			node->nextsibling = parent->firstchild;
			parent->firstchild = node;

			return;
		}

		if (cs > closest->scale)
		{
			if (closest->scale == POINTSCALE)
			{
				// create new bottom level
				Node* copy = freeNode();
				Node* node = freeNode();

				copy->index = closest->index;
				copy->scale = POINTSCALE;
				copy->firstchild = NULL;
				copy->nextsibling = node;

				node->index = index;
				node->scale = POINTSCALE;
				node->firstchild = NULL;
				node->nextsibling = NULL;

				closest->firstchild = copy;
				closest->scale = cs;

				return;
			}
			else
			{
				// insert a level
				Node* copy = freeNode();
				Node* node = freeNode();

				copy->index = closest->index;
				copy->scale = closest->scale;
				copy->firstchild = closest->firstchild;
				copy->nextsibling = node;

				closest->scale = cs;
				closest->firstchild = copy;

				node->index = index;
				node->scale = POINTSCALE;
				node->firstchild = NULL;
				node->nextsibling = NULL;

				return;
			}
		}

		// otherwise update parent
		parent = closest;
		pd2 = cd2;
		ps = cs;
	}
}


////////////////////////////////////////////////////////////


KernelCoverTree::KernelCoverTree(KernelFunction* kernel, const Array<double>& points)
: CoverTree(points)
{
	this->kernel = kernel;
}

KernelCoverTree::~KernelCoverTree()
{
}


double KernelCoverTree::distance2(const Array<double>& p1, const Array<double>& p2) const
{
	return (*kernel)(p1, p1) + (*kernel)(p2, p2) - 2.0 * (*kernel)(p1, p2);
}
