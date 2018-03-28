//===========================================================================
/*!
 * 
 *
 * \brief       Random Forest Implementation files
 * 
 * 
 *
 * \author      O. Krause, F.Gieseke
 * \date        2017
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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


#ifndef SHARK_ALGORITHMS_TRAINERS_IMPL_CART_H
#define SHARK_ALGORITHMS_TRAINERS_IMPL_CART_H

#include <shark/Core/Random.h>
#include <shark/Core/utility/KeyValuePair.h>
#include <shark/LinAlg/Base.h>
#include <shark/Statistics/Distributions/MultiNomialDistribution.h>

namespace shark {namespace CART{
	
template<class DataSet, class LabelSet>
struct Bootstrap{
	DataSet const& data; ///< constant reference to the dataset. only the susbet indexed by indices is used.
	std::vector<std::size_t> indices; ///< stores the index of the ith element in data
	std::vector<std::size_t> complement; ///<Complement of indices used for OOB error
	LabelSet labels; ///< the label of the ith data point. 
	std::size_t labelDim; ///< dimensionality of a vector to hold the labels in expanded form 
	std::vector<unsigned int> weights; /// number of times the ith point got picked.
	
	///\brief Creates a random bootstrap from the provided dataset.
	Bootstrap(random::rng_type& rng, DataSet const& data, LabelSet const& labels, RealVector const& sample_weights):data(data){
		// sample bootstrap indices (with replacement)
		// we use the sample weights as sample probabilities (they are correctly normalized to 1)
		MultiNomialDistribution dist(sample_weights);
		unsigned int ell = data.size1();
		std::vector<unsigned int> bootstrapIndices(ell,0);
		for (unsigned int i = 0; i < ell; i++) {
			bootstrapIndices[dist(rng)] += 1;
		}
		//compress bootstrap sample to actually sampled points
		for (unsigned int i = 0; i < ell; i++) {
			if (bootstrapIndices[i] == 0){
				complement.push_back(i);
			}else{
				indices.push_back(i);
				weights.push_back(bootstrapIndices[i]);
			}
		}
		setLabels(labels);
	}
	
	// Partition the bootstrap (indices,weights,labels) in the range start,end
	// so that all points with the value of the feature smaller than the theshold
	// are on the left side and all values larger are on the right side
	void partition(
		std::size_t start, std::size_t end,
		double threshold, unsigned feature
	) {
		std::size_t pos = start;
		while (pos < end) {
			double xf = data(indices[pos], feature);

			if (xf <= threshold) {
				pos++;
			} else {
				end--;
				std::swap(indices[pos],indices[end]);
				std::swap(weights[pos],weights[end]);
				swapLabel(labels, pos, end);
			}
		}
	}
private:
	void setLabels(RealMatrix const& labels){
		labelDim = labels.size2();
		this->labels.resize(indices.size(), labelDim);
		for (std::size_t i = 0; i != indices.size(); ++i) {
			noalias(row(this->labels,i)) = row(labels,indices[i]);
		}
	}
	
	void setLabels(UIntVector const& labels){
		labelDim = max(labels) + 1;
		for (std::size_t i = 0; i != indices.size(); ++i) {
			auto index = indices[i];
			this->labels.push_back(labels(index));
		}
	}
	
	void swapLabel(RealMatrix& labels, std::size_t i, std::size_t j){
		for(std::size_t k = 0; k != labels.size2(); ++k){
			std::swap(labels(i,k),labels(j,k));
		}
	}
	void swapLabel(UIntVector& labels, std::size_t i, std::size_t j){
		std::swap(labels(i),labels(j));
	}
};
	
struct MSECriterion{
	struct CriterionRecord {
		std::size_t current_pos;
		double impurity;
		double impurity_left;
		double impurity_right;
		double improvement;

		double weight_left;
		double weight_right;

		RealVector sum_left;
		RealVector sum_right;
		RealVector sq_sum_left;
		RealVector sq_sum_right;
	};
	
	static RealVector leafLabel(CriterionRecord const& record){
		return record.sum_right/record.weight_right;
	}
	
	static double leafLoss(CriterionRecord const& record){
		return record.impurity;
	}
	
	static CriterionRecord initCriterion(
		RealMatrix const& labels, std::vector<unsigned int> const& weights, std::size_t labelDim
	) {
		CriterionRecord critRecord;
		critRecord.current_pos = 0;

		// set right sum/weights
		critRecord.sum_right =blas::repeat(0.0,labelDim);
		critRecord.sq_sum_right = blas::repeat(0.0,labelDim);

		//compute impurity for leftmost split
		double sum_weights = 0;
		for (std::size_t i = 0; i < labels.size1(); i++) {
			double weight = weights[i];
			auto const& label = row(labels,i);
			noalias(critRecord.sum_right) += weight * label;
			noalias(critRecord.sq_sum_right) += weight*sqr(label);
			sum_weights += weight;
		}

		// impurity of node
		critRecord.impurity = sum(critRecord.sq_sum_right) / sum_weights - norm_sqr(critRecord.sum_right)  / (sum_weights * sum_weights);
		critRecord.impurity_left = 0.0;
		critRecord.impurity_right = critRecord.impurity;
		critRecord.weight_left = 0;
		critRecord.weight_right = sum_weights;

		// improvement
		critRecord.improvement = 0;
		return critRecord;
	}
	
	static void updateCriterion(
		CriterionRecord& critRecord,
		std::size_t new_pos,
		std::vector<KeyValuePair<double,unsigned int> >& XF,
		RealMatrix const& labels,
		std::vector<unsigned int> const& weights
	) {
		
		if(critRecord.sum_left.empty()){
			critRecord.sum_left = blas::repeat(0.0, critRecord.sum_right.size());
			critRecord.sq_sum_left = blas::repeat(0.0, critRecord.sum_right.size());
		}
		for (std::size_t k = critRecord.current_pos; k < new_pos; k++) {
			double weight = weights[XF[k].value];
			auto const& label = row(labels,XF[k].value);
			noalias(critRecord.sum_left) += weight*label;
			noalias(critRecord.sq_sum_left) += weight * sqr(label);
			noalias(critRecord.sum_right) -= weight * label;
			noalias(critRecord.sq_sum_right) -= weight * sqr(label);
			critRecord.weight_left += weight;
			critRecord.weight_right -= weight;
		}
		critRecord.current_pos = new_pos;

		// left and right impurity
		critRecord.impurity_left = sum(critRecord.sq_sum_left) / critRecord.weight_left;
		critRecord.impurity_left -= norm_sqr(critRecord.sum_left) / (critRecord.weight_left * critRecord.weight_left);
		critRecord.impurity_right = sum(critRecord.sq_sum_right) / critRecord.weight_right;
		critRecord.impurity_right -= norm_sqr(critRecord.sum_right) / (critRecord.weight_right * critRecord.weight_right);

		double weight_all = critRecord.weight_left + critRecord.weight_right;
		double fraction_left = critRecord.weight_left / weight_all;
		double fraction_right = critRecord.weight_right / weight_all;

		// improvement
		critRecord.improvement = critRecord.impurity - fraction_left * critRecord.impurity_left - fraction_right * critRecord.impurity_right;
	}
	
	static void split(CriterionRecord critRecord, CriterionRecord& left, CriterionRecord& right){
		left.current_pos = 0;
		left.impurity = critRecord.impurity_left;
		left.impurity_left = 0;
		left.impurity_right = critRecord.impurity_left;
		left.improvement = 0;
		left.weight_left = 0;
		left.weight_right = critRecord.weight_left;
		left.sum_right = std::move(critRecord.sum_left);
		left.sq_sum_right = std::move(critRecord.sq_sum_left);
		
		right.current_pos = 0;
		right.impurity = critRecord.impurity_right;
		right.impurity_left = 0;
		right.impurity_right = critRecord.impurity_right;
		right.improvement = 0;
		right.weight_left = 0;
		right.weight_right = critRecord.weight_right;
		right.sum_right = std::move(critRecord.sum_right);
		right.sq_sum_right = std::move(critRecord.sq_sum_right);
	}
};

struct ClassificationCriterion{
	struct CriterionRecord {
		std::size_t current_pos;
		double impurity;
		double impurity_left;
		double impurity_right;
		double improvement;

		double weight_left;
		double weight_right;

		std::vector<int> class_counts_left;
		std::vector<int> class_counts_right;
	};
	
	static unsigned int leafLabel(CriterionRecord const& record){
		auto pos = std::max_element(record.class_counts_right.begin(),record.class_counts_right.end());
		return pos - record.class_counts_right.begin();
	}
	
	static double leafLoss(CriterionRecord const& record){
		auto pos = std::max_element(record.class_counts_right.begin(),record.class_counts_right.end());
		return 1.0 - (*pos)/(record.weight_right);
	}
	
	static CriterionRecord initCriterion(
		UIntVector const& labels, std::vector<unsigned int> const& weights, std::size_t nClasses
	) {
		// initialize records
		CriterionRecord critRecord;
		critRecord.improvement = 0;
		critRecord.current_pos = 0;
		critRecord.class_counts_right.resize(nClasses, false);

		// compute all class ratios (right side)
		double sum_weights = 0;
		for (std::size_t k = 0; k < labels.size(); k++) {
			unsigned int weight = weights[k];
			unsigned int label = labels[k];
			critRecord.class_counts_right[label] += weight;
			sum_weights += weight;
		}

		// compute gini impurity
		critRecord.impurity = 0.0;
		for (unsigned int i = 0; i < nClasses; i++) {
			double pmk = critRecord.class_counts_right[i] / sum_weights;
			critRecord.impurity += pmk * (1.0 - pmk);
		}

		// impurity of node
		critRecord.impurity_left = 0.0;
		critRecord.impurity_right = critRecord.impurity;
		critRecord.weight_left = 0;
		critRecord.weight_right = sum_weights;

		return critRecord;
	}

	static void updateCriterion(
		CriterionRecord& critRecord,
		std::size_t new_pos,
		std::vector<KeyValuePair<double,unsigned int> >& XF,
		UIntVector const& labels,
		std::vector<unsigned int> const& weights
	) {
		if(critRecord.class_counts_left.empty())
			critRecord.class_counts_left.resize(critRecord.class_counts_right.size(),false);
		
		for (std::size_t k = critRecord.current_pos; k < new_pos; k++) {
			unsigned int weight = weights[XF[k].value];
			unsigned int label = labels[XF[k].value];
			critRecord.class_counts_left[label] += weight;
			critRecord.class_counts_right[label] -= weight;
			critRecord.weight_left += weight;
			critRecord.weight_right -= weight;
		}
		critRecord.current_pos = new_pos;
		critRecord.impurity_left = 0.0;
		critRecord.impurity_right = 0.0;

		for (std::size_t i = 0; i < critRecord.class_counts_left.size(); i++) {
			// left impurity
			double pmk_left = critRecord.class_counts_left[i] / critRecord.weight_left;
			critRecord.impurity_left += pmk_left * (1.0 - pmk_left);

			// right impurity
			double pmk_right = critRecord.class_counts_right[i] / critRecord.weight_right;
			critRecord.impurity_right += pmk_right * (1.0 - pmk_right);
		}

		double weight_all = critRecord.weight_left + critRecord.weight_right;
		double fraction_left = critRecord.weight_left / weight_all;
		double fraction_right = critRecord.weight_right / weight_all;

		// improvement
		critRecord.improvement = critRecord.impurity - fraction_left * critRecord.impurity_left - fraction_right * critRecord.impurity_right;
	}
	
	static void split(CriterionRecord critRecord, CriterionRecord& left, CriterionRecord& right){
		left.current_pos = 0;
		left.impurity = critRecord.impurity_left;
		left.impurity_left = 0;
		left.impurity_right = critRecord.impurity_left;
		left.improvement = 0;
		left.weight_left = 0;
		left.weight_right = critRecord.weight_left;
		left.class_counts_right = std::move(critRecord.class_counts_left);
		
		right.current_pos = 0;
		right.impurity = critRecord.impurity_right;
		right.impurity_left = 0;
		right.impurity_right = critRecord.impurity_right;
		right.improvement = 0;
		right.weight_left = 0;
		right.weight_right = critRecord.weight_right;
		right.class_counts_right = std::move(critRecord.class_counts_right);
	}
};

 template<class LabelType, class Criterion>
class TreeBuilder
{
public:
	typedef typename Batch<LabelType>::type LabelBatch;
	typedef blas::matrix<double, blas::column_major> DataBatch;
	typedef CART::Bootstrap<DataBatch, LabelBatch> Bootstrap;
private:
	typedef typename Criterion::CriterionRecord CriterionRecord;
	
	struct TraversalRecord{
		std::size_t nodeId;
		std::size_t start;
		std::size_t end;
		unsigned depth;
		std::vector<bool> constFeatures;
		double priority;
		CriterionRecord criterion;

		bool operator<(TraversalRecord const& other)const{
			return priority < other.priority;
		}
	};
	
	struct SplitRecord{
		unsigned int feature;
		double threshold;
		unsigned pos;
		double improvement;
		double impurity_left;
		double impurity_right;
		CriterionRecord criterion;
		
		bool operator<(SplitRecord const& other)const{
			return improvement < other.improvement;
		}
	};
public:
	std::size_t m_max_features;///< number of attributes to randomly test at each inner node
	std::size_t m_min_samples_leaf; ///< minimum number of samples in a leaf node
	std::size_t m_min_split; ///< minimum number of samples to be considered a split
	std::size_t m_max_depth;///< maximum depth of the tree
	double m_epsilon;///< Minimum difference between two values to be considered different
	double m_min_impurity_split;///< stops splitting when the impority is below a threshold
	
	CARTree<LabelType> buildTree(
		random::rng_type& rng,
		Bootstrap& bootstrap
	){
		//create root of the tree
		CARTree<LabelType> tree(bootstrap.data.size2(), bootstrap.labelDim);
		tree.createRoot();
		
		//small helper function to create the leafs.
		auto makeLeaf=[&](TraversalRecord const& record){
			if(record.end - record.start == 1){
				tree.transformLeafNode(record.nodeId, getBatchElement(bootstrap.labels,record.start));
			}else{
				tree.transformLeafNode(record.nodeId, Criterion::leafLabel(record.criterion));
			}
		};
		
		//push root entry into the priority queue
		std::priority_queue<TraversalRecord> queue;
		TraversalRecord record = {0, 0,bootstrap.indices.size(), 0, std::vector<bool>(bootstrap.data.size2(),false),0};
		record.criterion = Criterion::initCriterion(bootstrap.labels, bootstrap.weights, bootstrap.labelDim);
		if(!enqueueRecord(queue,record))
			makeLeaf(record);

		//while the priority queue is not empty, perform splits and create leaves
		while (!queue.empty()) {
			record = queue.top();
			queue.pop();
			// find the best split
			// if there is no valid split, this is a leaf node, which we create
			SplitRecord split;
			if(!findSplit(rng, record, split, bootstrap)){
				makeLeaf(record);
				continue;
			}
			//create split node
			auto const& node = tree.transformInternalNode(record.nodeId, split.feature, split.threshold);
			
			//swap
			std::size_t start = record.start;
			std::size_t end = record.end;
			std::size_t pos = start + split.pos;
			bootstrap.partition(start, end, split.threshold, split.feature);
			
			//enqueue new nodes taking priority into account
			unsigned leafDepth = record.depth + 1;
			double priority = double(leafDepth);
			TraversalRecord left = {node.leftId, start, pos, leafDepth, record.constFeatures, priority};
			TraversalRecord right = {node.rightIdOrIndex, pos, end, leafDepth, record.constFeatures, priority};
			Criterion::split(std::move(split.criterion),left.criterion,right.criterion);
			//enqueue childs if they do not already statisfy condition for a leaf(e.g. too small)
			if(!enqueueRecord(queue,left))
				makeLeaf(left);
			if(!enqueueRecord(queue,right))
				makeLeaf(right);
		}
		return tree;
	}
private:

	bool enqueueRecord(std::priority_queue<TraversalRecord>& queue, TraversalRecord const& record){
		bool isLeaf = false;
		std::size_t numSamples = record.end - record.start;
		isLeaf |= record.depth == m_max_depth;
		isLeaf |= numSamples < 2 * m_min_samples_leaf;
		isLeaf |= numSamples < m_min_split;
		isLeaf |= record.criterion.impurity <= m_min_impurity_split;
		if(!isLeaf)
			queue.push(record);
		return !isLeaf;
	}
	
	// Compute the best split based on the impurity measure
	// the split is stored in the traversal record and has all information
	// to perform the actual splitting
	bool findSplit(
		random::rng_type& rng,
		TraversalRecord& record,
		SplitRecord& split,
		Bootstrap const& bootstrap
	){
		std::vector<unsigned> randomFeatures(bootstrap.data.size2());
		std::iota(randomFeatures.begin(),randomFeatures.end(),0);
		std::shuffle(randomFeatures.begin(), randomFeatures.end(),rng);
		
		std::size_t start = record.start;
		std::size_t end = record.end;
		
		//vector for storing split feature values. This gives faster memory access later
		std::vector<KeyValuePair<double,unsigned int> > XF(end - start);
		split.improvement = 0.0;
		CriterionRecord bestCriterion;
		for (std::size_t j = 0; j < randomFeatures.size(); j++) {
			// Break as soon as at least max_features and a non-trivial split can be found
			if (j >= m_max_features && split.improvement > 0.0) {
				break;
			}
			unsigned feature = randomFeatures[j];
			//only check the feature if it is not already known to be constant
			if(record.constFeatures[feature]){
				continue;
			}
			// Copy data in XF for faster lookup and
			// compute minimum and maximum to check if it is constant
			double minf = std::numeric_limits<double>::max();
			double maxf = -std::numeric_limits<double>::max();
			for (std::size_t i = start; i < end; i++) {
				double f = bootstrap.data(bootstrap.indices[i],feature);
				XF[i - start].key = f;
				XF[i - start].value =  i;
				minf = std::min(minf, f);
				maxf = std::max(maxf, f);
			}
			//compute split
			SplitRecord newSplit;
			if (maxf <= minf + m_epsilon){//no reason to check with a constant split
				newSplit.improvement = 0.0;
				record.constFeatures[feature]  = true;
			}else{
				newSplit = computeOptimalThreshold(XF, bootstrap, record.criterion);
			}
			newSplit.feature = feature;
			split=std::max(split,newSplit);
		}
		
		//if we could not find any improvement, this is a leaf
		return (split.improvement > 0.0);
	}
	
	SplitRecord computeOptimalThreshold(
		std::vector<KeyValuePair<double,unsigned int> >& XF,
		Bootstrap const& bootstrap,
		CriterionRecord criterion//copied because it is changed
	){
		
		// Important: We are checking a non-constant feature here.
		std::sort(XF.begin(),XF.end());

		// init split
		SplitRecord bestSplit;
		bestSplit.improvement = 0;
		bestSplit.threshold = -std::numeric_limits<double>::max();
		bestSplit.pos = 0;
		bestSplit.criterion = criterion;
		
		//choosing start position and end this way ensures that splits 
		//have correct sizes (also saves some iteration cost)
		int end = XF.size() - m_min_samples_leaf + 1;
		int p = m_min_samples_leaf - 1;
		while (p < end) {

			// Increase counter until feature difference is significant
			while ((p + 1 < end) && (XF[p+1].key <= XF[p].key + m_epsilon)){
				p++;
			}

			// p==end possible; here, we already have p > start!
			p += 1;
			if (p == end) break;

			// update criterion w.r.t. new position p
			Criterion::updateCriterion(criterion, p, XF, bootstrap.labels, bootstrap.weights);

			// store results if improvement is better than before
			if (criterion.improvement > bestSplit.improvement) {
				// compute threshold as the average
				double threshold = (XF[p - 1].key + XF[p].key) / 2.0;
				//check for numerical stability of the threshold
				if (threshold == XF[p].key) {
					threshold = XF[p - 1].key;
				}
				//update best split
				bestSplit.improvement = criterion.improvement;
				bestSplit.threshold = threshold;
				bestSplit.pos = p;
				bestSplit.impurity_left = criterion.impurity_left;
				bestSplit.impurity_right = criterion.impurity_right;
			}
		}
		Criterion::updateCriterion(bestSplit.criterion, bestSplit.pos, XF, bootstrap.labels, bootstrap.weights);
		return bestSplit;
	}

};
}}
#endif
