/*! ======================================================================
 *
 *  \file Hypervolume.h
 *
 *  \brief Implementation of several algorithms for calculating 
 *	the hypervolume of a set of points.
 * 
 *  \author Thomas Vo&szlig; <thomas.voss@rub.de>
 *
 *  \par 
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 * 
 *  \par Project:
 *      MOO-EALib
 *  <BR>
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
 *  Foundation, Inc., 675 Mass Ave, Camâbridge, MA 02139, USA.
 */

#include <SharkDefs.h>
#include <MOO-EALib/Hypervolume.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <errno.h>

/* Auxiliary functions and variables for the Overmars-Yap algorithm. */
static unsigned NO_OBJECTIVES;
static double	SQRT_NO_DATA_POINTS;

int compare( const void * a, const void * b ) {
	double * x = (double*) a;
	double * y = (double*) b;
	
	if( x[NO_OBJECTIVES-1] == y[NO_OBJECTIVES-1] )
		return( 0 );
		
	if( x[NO_OBJECTIVES-1] < y[NO_OBJECTIVES-1] )
		return( -1 );
		
	// if( x[NO_OBJECTIVES-1] > y[NO_OBJECTIVES-1] )
	
	return( 1 );
}

int 	covers				( double * cuboid, 		double * regionLow						);
int 	partCovers			( double * cuboid, 		double * regionUp						);
int 	containsBoundary	( double * cub, 		double * regLow, 	int split			);
double 	getMeasure			( double * regionLow, 	double * regionUp						);
int 	isPile				( double * cuboid, 		double * regionLow, double * regionUp	);
int 	binaryToInt			( int * bs );
void 	intToBinary			( int i, 				int * result							);
double 	computeTrellis		( double * regLow, 		double * regUp, 	double * trellis	);
double 	getMedian			( double * bounds, 		int length								);

double 	stream				( double * regionLow, 	double * regionUp, 	double * points, 
							  unsigned noPoints, 	int split, 			double cover		);

double overmars_yap( double * points, double * referencePoint, unsigned noObjectives, unsigned noPoints ) {
	NO_OBJECTIVES 		= noObjectives;
	SQRT_NO_DATA_POINTS = sqrt( (double)noPoints );
	
	double * regLow = new double[NO_OBJECTIVES];
	std::fill( regLow, regLow + NO_OBJECTIVES, MAXDOUBLE );
	
	// Calculate Bounding Box
	double * p = points;
	for( unsigned i = 0; i < noPoints; i++ ) {
		for( unsigned j = 0; j < NO_OBJECTIVES; j++ ) {
			regLow[j] = Shark::min( regLow[j], *p );
			
			++p;
		}
	}
	
	double d = stream( regLow, referencePoint, points, noPoints, 0, referencePoint[NO_OBJECTIVES-1] );
	
	delete [] regLow;
	return( d );
}

int covers(double * cuboid, double * regionLow)
{
	unsigned i;
	for (i = 0; i < NO_OBJECTIVES - 1; i++)
	{
		if (cuboid[i] > regionLow[i])
			return (0);
	}
	return (1);
}

int partCovers( double * cuboid, double * regionUp ) {
	unsigned i;
	for (i = 0; i < NO_OBJECTIVES - 1; i++) {
		if (cuboid[i] >= regionUp[i])
			return (0);
	}
	return (1);
}

int containsBoundary(double * cub, double * regLow, int split) {
	if (regLow[split] >= cub[split]) {
		return -1;
	} else {
		int j;
		for (j = 0; j < split; j++) {
			if (regLow[j] < cub[j]) {
				return 1;
			}
		}
	}
	return 0;
}

double getMeasure(double * regionLow, double * regionUp)
{
	double volume = 1.0;
	unsigned i;
	for (i = 0; i < NO_OBJECTIVES - 1; i++)
	{
		volume *= (regionUp[i] - regionLow[i]);
	}

	return (volume);
}

int isPile(double * cuboid, double * regionLow, double * regionUp)
{
	unsigned pile = NO_OBJECTIVES;
	unsigned i;
	for (i = 0; i < NO_OBJECTIVES - 1; i++)
	{
		if (cuboid[i] > regionLow[i])
		{
			if (pile != NO_OBJECTIVES)
			{
				return (-1);
			}

			pile = i;
		}
	}

	return (pile);
}

int binaryToInt(int * bs)
{
	int result = 0;
	unsigned i;
	for (i = 0; i < NO_OBJECTIVES - 1; i++)
	{
		result += bs[i] * (int) pow(2.0, (double)i);
	}

	return (result);
}

void intToBinary(int i, int * result)
{
	unsigned j;
	for (j = 0; j < NO_OBJECTIVES - 1; j++) 
		result[j] = 0;

	int rest = i;
	int idx = 0;

	while (rest != 0)
	{
		result[idx] = (rest % 2);

		rest = rest / 2;
		idx++;
	}
}

double computeTrellis(double * regLow, double * regUp, double * trellis)
{
	unsigned i, j;
	int * bs = (int*)malloc((NO_OBJECTIVES - 1) * sizeof(int));
	for (i = 0; i < NO_OBJECTIVES - 1; i++) bs[i] = 1;

	double result = 0;

	int noSummands = binaryToInt(bs);
	int oneCounter; double summand;

	for (i = 1; i <= (unsigned)noSummands; i++)
	{
		summand = 1;
		intToBinary(i, bs);
		oneCounter = 0;

		for (j = 0; j < NO_OBJECTIVES - 1; j++)
		{
			if (bs[j] == 1)
			{
				summand *= regUp[j] - trellis[j];
				oneCounter++;
			}
			else
				summand *= regUp[j] - regLow[j];
		}

		if (oneCounter % 2 == 0)
			result -= summand ;
		else
			result += summand;
	}

	free(bs);

	return(result);
}

int double_compare(const void *p1, const void *p2)
{
	double i = *((double *)p1);
	double j = *((double *)p2);

	if (i > j)
		return (1);
	if (i < j)
		return (-1);
	return (0);
}

double getMedian(double * bounds, int length)
{
	if (length == 1)
	{
		return bounds[0];
	}
	else if (length == 2)
	{
		return bounds[1];
	}

	qsort(bounds, length, sizeof(double), double_compare);

	return(length % 2 == 1 ? bounds[length / 2] : (bounds[length / 2] + bounds[length / 2 + 1]) / 2);
}

double stream(double * regionLow,
			  double * regionUp,
			  double * points,
			  unsigned noPoints,
			  int split,
			  double cover ) {
	double coverOld;
	coverOld = cover;
	int coverIndex = 0;
	int c;

	double result = 0;

	double dMeasure = getMeasure(regionLow, regionUp);
	while (cover == coverOld && coverIndex < (double)noPoints)
	{
		if (covers(points + (coverIndex * NO_OBJECTIVES), regionLow))
		{
			cover = points[coverIndex * NO_OBJECTIVES + NO_OBJECTIVES - 1];
			result += dMeasure * (coverOld - cover);
		}
		else
			coverIndex++;
	}

	for (c = coverIndex; c > 0; c--) {
		if (points[(c - 1) * NO_OBJECTIVES + NO_OBJECTIVES - 1] == cover) {
			coverIndex--;
		}
	}

	if (coverIndex == 0)
	{
		return (result);
	}

	int allPiles = 1; int i;

	int  * piles = (int*)malloc(coverIndex * sizeof(int));
	for (i = 0; i < coverIndex; i++)
	{
		piles[i] = isPile(points + i * NO_OBJECTIVES, regionLow, regionUp);
		if (piles[i] == -1)
		{
			allPiles = 0;
			break;
		}
	}

	if (allPiles)
	{
		double * trellis = (double*)malloc((NO_OBJECTIVES - 1) * sizeof(double));
		for (c = 0; c < (int)NO_OBJECTIVES - 1; c++)
		{
			trellis[c] = regionUp[c];
		}

		double current = 0.0;
		double next = 0.0;
		i = 0;
		do
		{
			current = points[i * NO_OBJECTIVES + NO_OBJECTIVES - 1];
			do
			{
				if ( points[i * NO_OBJECTIVES + piles[i]] < trellis[piles[i]])
				{
					trellis[piles[i]] = points[i * NO_OBJECTIVES + piles[i]];
				}
				i++;
				if (i < coverIndex)
				{
					next = points[i * NO_OBJECTIVES + NO_OBJECTIVES - 1];
				}
				else
				{
					next = cover;
					break;
				}

			}
			while (next == current);
			result += computeTrellis(regionLow, regionUp, trellis)
					  * (next - current);
		}
		while (next != cover);
		free(trellis);
	}
	else
	{
		double bound = -1.0;
		double * boundaries = (double*) malloc(coverIndex * sizeof(double));
		unsigned boundIdx = 0;
		double * noBoundaries = (double*)malloc(coverIndex * sizeof(double));
		unsigned noBoundIdx = 0;

		do
		{
			for (i = 0; i < coverIndex; i++)
			{
				int contained = containsBoundary(points + i * NO_OBJECTIVES, regionLow, split);
				if (contained == 1)
				{
					boundaries[boundIdx] = points[i * NO_OBJECTIVES + split];
					boundIdx++;
				}
				else if (contained == 0)
				{
					noBoundaries[noBoundIdx] = points[i * NO_OBJECTIVES + split];
					noBoundIdx++;
				}
			}

			if (boundIdx > 0)
			{
				bound = getMedian(boundaries, boundIdx);
			}
			else if (noBoundIdx > SQRT_NO_DATA_POINTS)
			{
				bound = getMedian(noBoundaries, noBoundIdx);
			}
			else
			{
				split++;
			}
		}
		while (bound == -1.0);

		free(boundaries); free(noBoundaries);

		double * pointsChild = new double[coverIndex * NO_OBJECTIVES];//(doublep*)malloc(coverIndex * sizeof(doublep*));
		int pointsChildIdx = 0;

		double * regionUpC = (double*)malloc(NO_OBJECTIVES * sizeof(double));
		memcpy(regionUpC, regionUp, NO_OBJECTIVES * sizeof(double));
		regionUpC[split] = bound;

		for (i = 0; i < coverIndex; i++)
		{
			if (partCovers(points + i * NO_OBJECTIVES, regionUpC))
			{
				std::copy( points + i*NO_OBJECTIVES, points + i*NO_OBJECTIVES + NO_OBJECTIVES, pointsChild + pointsChildIdx*NO_OBJECTIVES );
				//pointsChild[pointsChildIdx] = points[i * NO_OBJECTIVES];
				pointsChildIdx++;
			}
		}

		if (pointsChildIdx > 0)
		{
			result += stream(regionLow, regionUpC, pointsChild, pointsChildIdx, split, cover);
		}

		pointsChildIdx = 0;

		double * regionLowC = (double*)malloc(NO_OBJECTIVES * sizeof(double));
		memcpy(regionLowC, regionLow, NO_OBJECTIVES * sizeof(double));
		regionLowC[split] = bound;
		for (i = 0; i < coverIndex; i++)
		{
			if (partCovers(points + i * NO_OBJECTIVES, regionUp))
			{
				// pointsChild[pointsChildIdx] = points[i];
				std::copy( points + i*NO_OBJECTIVES, points + i*NO_OBJECTIVES + NO_OBJECTIVES, pointsChild + pointsChildIdx*NO_OBJECTIVES );
				pointsChildIdx++;
			}
		}
		if (pointsChildIdx > 0)
		{
			result += stream(regionLowC, regionUp, pointsChild, pointsChildIdx, split, cover);
		}

		free(regionUpC);
		free(regionLowC);
		delete [] pointsChild;
	}

	free(piles);

	return (result);
}


int cmp(const void* p1, const void* p2)
{
	const double* d1 = (double*)p1;
	const double* d2 = (double*)p2;
	if (*d1 < *d2) return -1;
	else if (*d1 > *d2) return 1;
	else return 0;
}

void sortByLastObjective(double* points, unsigned int noObjectives, unsigned int noPoints)
{
	unsigned int i;
	double* p;

	p = points;
	for (i=0; i<noPoints; i++)
	{
		std::swap(p[0], p[noObjectives - 1]);
		p += noObjectives;
	}

	qsort(points, noPoints, noObjectives * sizeof(double), cmp);

	p = points;
	for (i=0; i<noPoints; i++)
	{
		std::swap(p[0], p[noObjectives - 1]);
		p += noObjectives;
	}
}
		
//! \brief Comparison of the last component of two fitness vectors.		
struct LastObjectiveComparator {
	static unsigned int NO_OBJECTIVES;
	static int compare( const void * p1, const void * p2 ) {
		const double * d1 = reinterpret_cast<const double*>( p1 );
		const double * d2 = reinterpret_cast<const double*>( p2 );				
				
		if (d1[NO_OBJECTIVES-1] < d2[NO_OBJECTIVES-1]) return -1;
			else if (d1[NO_OBJECTIVES-1] > d2[NO_OBJECTIVES-1]) return 1;
			else return 0;
		}
	};
unsigned int LastObjectiveComparator::NO_OBJECTIVES = 0;

double hypervolume(double* points, double* referencePoint, unsigned int noObjectives, unsigned int noPoints) {
	unsigned int i;
	if (noObjectives == 0) {
		throw SHARKEXCEPTION("[hypervolume] dimension must be positive");
	}
	else if (noObjectives == 1) {
		// trivial
		double m = 1e100;
		for (i=0; i<noPoints; i++) if (points[i] < m) m = points[i];
		double h = *referencePoint - m;
		if (h < 0.0) h = 0.0;
		return h;
	}
	else if (noObjectives == 2) {
		// sort by last objective
		LastObjectiveComparator::NO_OBJECTIVES = 2;
		qsort(points, noPoints, noObjectives * sizeof(double), LastObjectiveComparator::compare );
		// sortByLastObjective(points, noObjectives, noPoints);
		double h = (referencePoint[0] - points[0]) * (referencePoint[1] - points[1]);
		double diffDim1; unsigned int lastValidIndex = 0;
		for (i=1; i<noPoints; i++) {
			diffDim1 = points[2*lastValidIndex] - points[2*i];  // Might be negative, if the i-th solution is dominated.
			if( diffDim1 > 0 ) {
				h += diffDim1 * (referencePoint[1] - points[2*i+1]);
				lastValidIndex = i;
			}
		}
		return h;
	} else {
		LastObjectiveComparator::NO_OBJECTIVES = noObjectives;
		qsort(points, noPoints, noObjectives * sizeof(double), LastObjectiveComparator::compare );
		// sortByLastObjective(points, noObjectives, noPoints);
		return overmars_yap(points, referencePoint, noObjectives, noPoints);
	}
}
