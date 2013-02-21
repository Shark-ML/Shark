/*

This file is part of Ext JS 4

Copyright (c) 2011 Sencha Inc

Contact:  http://www.sencha.com/contact

GNU General Public License Usage
This file may be used under the terms of the GNU General Public License version 3.0 as published by the Free Software Foundation and appearing in the file LICENSE included in the packaging of this file.  Please review the following information to ensure the GNU General Public License version 3.0 requirements will be met: http://www.gnu.org/copyleft/gpl.html.

If you are unsure which license is appropriate for your use, please contact the sales department at http://www.sencha.com/contact.

*/
Ext.require(['Ext.data.*', 'Ext.grid.*']);
Ext.onReady(function() {
    // wrapped in closure to prevent global vars.
    Ext.define('Probe', {
        extend: 'Ext.data.Model',
        fields: ['name', 'value', 'timestamp']
    });

    var ProbesStore = new Ext.data.Store({
        model: 'Probe',
        proxy: {
            type: 'ajax',
            url: '/ProbeManager',
            reader: {
                type: 'json',
                root: 'probes'
            }
        },
        autoLoad: true
    });

    var intr = setInterval(function() {
        ProbesStore.load(function(records, operation, success) {
            console.log('loaded records');
        })
    }, 5000);

    var grid = Ext.create('Ext.Window', {
        renderTo: Ext.getBody(),
        xtype: 'window',
        title: 'Shark Machine Learning Library',
        width: 638,
        height: 609,
        layout: {
            type: 'vbox',
            align: 'stretch'
        },
        maximized: true,
        maximizable: true,
        hidden: false,
        items: [
                {
                    xtype: 'grid',
                    title: 'Registered Probes',
                    flex: 3,
                    columnLines: true,
                    stripeRows: true,
                    store: ProbesStore,
                    columns: [
                        {
                            dataIndex: 'name',
                            header: 'Name',
                            sortable: true
                        },
                        {
                            xtype: 'numbercolumn',
                            dataIndex: 'value',
                            header: 'Value',
                            sortable: true
                        },
                        {
                            dataIndex: 'timestamp',
                            header: 'Timestamp [us]',
                            sortable: true
                        }
                    ]
                }, 
                {
                    xtype: 'chart',
                    store: ProbesStore,
                    hidden: false,      
                    flex: 1,
                    shadow: true,
                    animated: true,            
                    axes: [
                        {
                            title: 'Value',
                            type: 'Numeric',
                            position: 'left',
                            fields: ['value'],
                            grid: true
                        },
                        {
                            title: 'Probe',
                            type: 'Category',
                            position: 'bottom',
                            fields: ['name']
                        }
                    ],
                    series: [
                        {
                            type: 'bar',
                            xField: 'name',
                            yField: 'value',
                            column: true                             
                        }
                    ]
                },
                {
                    xtype: 'chart',
                    store: ProbesStore,
                    flex: 1,            
                    axes: [
                        {
                            title: 'Timestamp [us]',
                            type: 'Numeric',
                            position: 'left',
                            scale: 'logarithmic',
                            fields: ['timestamp'],
                            grid: true
                        },
                        {
                            title: 'Probe',
                            type: 'Category',
                            position: 'bottom',
                            fields: ['name'],
                            grid: true
                        }
                    ],
                    series: [
                        {
                            title: 'Last update',
                            type: 'bar',
                            axis: 'left',
                            xField: 'name',
                            yField: 'timestamp',
                            column: true         
                        }
                    ]
                }
            ]
    });

});

