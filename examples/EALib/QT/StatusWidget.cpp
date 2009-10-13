//===========================================================================
/*!
 *  \file StatusWidget.cpp
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


#include "StatusWidget.h"
#include <QApplication>
#include <QDesktopWidget>


StatusWidget::StatusWidget(const char* title, QWidget* parent)
: QWidget(parent)
, label("", this)
{
	setWindowTitle(title);

	QDesktopWidget* desktop = QApplication::desktop();
	QRect screen = desktop->screenGeometry();
	int cx = (screen.left() + screen.right()) / 2;
	int cy = (screen.top() + screen.bottom()) / 2;
	setGeometry(cx - 200, cy - 50, 400, 100);
	label.setGeometry(10, 10, 380, 80);
	label.setAlignment(Qt::AlignCenter);
}

StatusWidget::~StatusWidget()
{
}


void StatusWidget::setText(const char* text)
{
	label.setText(text);
	//QCoreApplication::instance()->processEvents();
	//usleep(200);
	QCoreApplication::instance()->processEvents();
}
