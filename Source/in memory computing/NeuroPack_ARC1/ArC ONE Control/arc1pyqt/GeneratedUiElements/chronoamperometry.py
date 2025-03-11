# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/foucault/Projects/arc1_pyqt/uis/chronoamperometry.ui'
#
# Created by: PyQt5 UI code generator 5.15.5
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ChronoAmpParent(object):
    def setupUi(self, ChronoAmpParent):
        ChronoAmpParent.setObjectName("ChronoAmpParent")
        ChronoAmpParent.resize(497, 261)
        self.verticalLayout = QtWidgets.QVBoxLayout(ChronoAmpParent)
        self.verticalLayout.setObjectName("verticalLayout")
        self.titleLabel = QtWidgets.QLabel(ChronoAmpParent)
        self.titleLabel.setObjectName("titleLabel")
        self.verticalLayout.addWidget(self.titleLabel)
        self.descriptionLabel = QtWidgets.QLabel(ChronoAmpParent)
        self.descriptionLabel.setObjectName("descriptionLabel")
        self.verticalLayout.addWidget(self.descriptionLabel)
        self.scrollArea = QtWidgets.QScrollArea(ChronoAmpParent)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 477, 168))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame = QtWidgets.QFrame(self.scrollAreaWidgetContents)
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 3, 0, 1, 1)
        self.biasEdit = QtWidgets.QLineEdit(self.frame)
        self.biasEdit.setObjectName("biasEdit")
        self.gridLayout.addWidget(self.biasEdit, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.numReadsBox = QtWidgets.QSpinBox(self.frame)
        self.numReadsBox.setMinimum(2)
        self.numReadsBox.setMaximum(100000)
        self.numReadsBox.setObjectName("numReadsBox")
        self.gridLayout.addWidget(self.numReadsBox, 2, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 0, 2, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pwValEdit = QtWidgets.QLineEdit(self.frame)
        self.pwValEdit.setMinimumSize(QtCore.QSize(60, 0))
        self.pwValEdit.setObjectName("pwValEdit")
        self.horizontalLayout_3.addWidget(self.pwValEdit)
        self.pwMulComboBox = QtWidgets.QComboBox(self.frame)
        self.pwMulComboBox.setObjectName("pwMulComboBox")
        self.horizontalLayout_3.addWidget(self.pwMulComboBox)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)
        self.gridLayout.addLayout(self.horizontalLayout_3, 1, 1, 1, 1)
        self.gridLayout.setColumnStretch(1, 2)
        self.gridLayout.setColumnStretch(2, 5)
        self.horizontalLayout_2.addWidget(self.frame)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout.addWidget(self.scrollArea)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.applyOneButton = QtWidgets.QPushButton(ChronoAmpParent)
        self.applyOneButton.setObjectName("applyOneButton")
        self.horizontalLayout.addWidget(self.applyOneButton)
        self.applyRangeButton = QtWidgets.QPushButton(ChronoAmpParent)
        self.applyRangeButton.setObjectName("applyRangeButton")
        self.horizontalLayout.addWidget(self.applyRangeButton)
        self.applyAllButton = QtWidgets.QPushButton(ChronoAmpParent)
        self.applyAllButton.setObjectName("applyAllButton")
        self.horizontalLayout.addWidget(self.applyAllButton)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(ChronoAmpParent)
        QtCore.QMetaObject.connectSlotsByName(ChronoAmpParent)

    def retranslateUi(self, ChronoAmpParent):
        _translate = QtCore.QCoreApplication.translate
        ChronoAmpParent.setWindowTitle(_translate("ChronoAmpParent", "Form"))
        self.titleLabel.setText(_translate("ChronoAmpParent", "Chronoamperometry"))
        self.descriptionLabel.setText(_translate("ChronoAmpParent", "Read a device continuously under bias"))
        self.biasEdit.setText(_translate("ChronoAmpParent", "1.0"))
        self.label_2.setText(_translate("ChronoAmpParent", "Bias duration"))
        self.label_3.setText(_translate("ChronoAmpParent", "Number of reads"))
        self.label.setText(_translate("ChronoAmpParent", "Bias"))
        self.pwValEdit.setText(_translate("ChronoAmpParent", "10"))
        self.applyOneButton.setText(_translate("ChronoAmpParent", "Apply to One"))
        self.applyRangeButton.setText(_translate("ChronoAmpParent", "Apply to Range"))
        self.applyAllButton.setText(_translate("ChronoAmpParent", "Apply to All"))
