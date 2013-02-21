/*

  This file is part of Ext JS 4

  Copyright (c) 2011 Sencha Inc

  Contact:  http://www.sencha.com/contact

  GNU General Public License Usage
  This file may be used under the terms of the GNU General Public License version 3.0 as published by the Free Software Foundation and appearing in the file LICENSE included in the packaging of this file.  Please review the following information to ensure the GNU General Public License version 3.0 requirements will be met: http://www.gnu.org/copyleft/gpl.html.

  If you are unsure which license is appropriate for your use, please contact the sales department at http://www.sencha.com/contact.

*/
Ext.require([
    'Ext.window.Window',
    'Ext.chart.*'
]);

Ext.define('ProbeModel', {
    extend: 'Ext.data.Model',
    fields: ['name', 'value'],

    proxy: {
        type: 'ajax',
        url: 'http://localhost:9090/ProbeManager',
        reader: {
            type: 'json'
        }
    }
});

var ProbeStore = Ext.create('Ext.data.Store', {
    xtype: 'store.store',
    autoLoad: true,
    autoSync: true,
    storeId: 'MyJsonStore',
    proxy: {
        type: 'ajax',
        url: 'http://localhost:9090/ProbeManager',
        reader: {
            type: 'json'
        }
    },
    fields: [
        {
            name: 'name',
            type: 'string'
        },
        {
            name: 'value',
            type: 'float'
        }
    ]
});

Ext.onReady(function () {

    Ext.create('Ext.Window', {
	xtype: 'window',
	height: 360,
	width: 492,
	layout: {
            type: 'fit'
	},
	title: 'Shark Machine Learning Library',
	items: [
            {
		xtype: 'gridpanel',
		title: 'Registered Probes',
		store: 'ProbeStore,
		columns: [
                    {
			xtype: 'gridcolumn',
			dataIndex: 'name',
			text: 'Name'
                    },
                    {
			xtype: 'numbercolumn',
			autoScroll: true,
			dataIndex: 'value',
			text: 'Value'
                    }
		],
		viewConfig: {

		}
            }
	]
    }).show();
    
});
