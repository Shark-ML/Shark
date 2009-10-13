//===========================================================================
/*!
 *  \file ControlWidget.h
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


#ifndef _ControlWidget_H_
#define _ControlWidget_H_

#include <Qt>
#include <QWidget>
#include <QLabel>
#include <QPushButton>


class ManualControlWidget : public QWidget
{
	Q_OBJECT

public:
	ManualControlWidget(QString experiment, QWidget* parent = NULL);
	~ManualControlWidget();

	bool isNext();
	bool isStop();

public slots:
	void onNext();
	void onStop();
	void onDestroy();

protected:
	void closeEvent(QCloseEvent* event);

	QLabel label;
	QPushButton next;
	QPushButton stop;
	bool wasNext;
	bool wasStop;
};


#endif
