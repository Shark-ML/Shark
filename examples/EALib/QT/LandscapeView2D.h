//===========================================================================
/*!
 *  \file LandscapeView2D.h
 *
 *  \author  T. Glasmachers
 *  \date    2008
 *
 *  \par Copyright (c) 2008:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-27974<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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


#ifndef _LandscapeView2D_H_
#define _LandscapeView2D_H_


#include "View.h"
#include "Optimization.h"


// View onto the fitness landscape of a two-dimensional search space.
// The visible area is a subset of the rectangle [-2, 2] x [-2, 2].
class LandscapeView2D : public View
{
	Q_OBJECT

public:
	LandscapeView2D(QString title, Experiment* experiment, QWidget* parent = NULL);
	~LandscapeView2D();

	void OnReset();
	void OnChanged(int evals);

	inline void monitorPopulation() { m_monitorPopulation = true; }
	inline void monitorSearchDistribution() { m_monitorSearchDistribution = true; }

	QSize sizeHint() const;

protected:
	void recomputeImage(int sx, int sy);

	void paintEvent(QPaintEvent* event);
	void resizeEvent(QResizeEvent* event);

	bool m_monitorPopulation;
	bool m_monitorSearchDistribution;

	Array<double> m_value;
	QImage* m_image;

	// data to image: (x, y) |--> m_scale * (x, y) + (m_offsetX, m_offsetY)
	// image to data: (x, y) |--> 1/m_scale * (y - m_offsetX , y - m_offsetY)
	double m_offsetX;				// offset in pixels
	double m_offsetY;				// offset in pixels
	double m_scale;					// pixels per unit

	unsigned int m_palette[1024];
};


#endif
