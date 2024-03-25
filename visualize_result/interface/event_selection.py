from PyQt5.QtWidgets import QMainWindow, QWidget, QGridLayout, QListWidget, QHBoxLayout
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt
from utils.event_class import Event, ms_to_time

class EventSelectionWindow(QMainWindow):
	def __init__(self, main_window):
		super().__init__()

		self.main_window = main_window

		# Defining some variables of the window
		self.title_window = "Event Selection"

		# Setting the window appropriately
		self.setWindowTitle(self.title_window)
		self.set_position()

		self.palette_main_window = self.palette()
		self.palette_main_window.setColor(QPalette.Window, Qt.black)

		# Initiate the sub-widgets
		self.init_window()

	def init_window(self):

		# Read the available labels
		self.labels = list()
		with open('../config/classes.txt') as file:
			for cnt, line in enumerate(file):
				self.labels.append(line.rstrip())

		self.list_widget = QListWidget()
		self.list_widget.clicked.connect(self.clicked)

		for item_nbr, element in enumerate(self.labels):
			self.list_widget.insertItem(item_nbr,element)

		# Layout the different widgets
		central_display = QWidget(self)
		self.setCentralWidget(central_display)
		final_layout = QHBoxLayout()
		final_layout.addWidget(self.list_widget)
		central_display.setLayout(final_layout)

		self.to_second = False
		self.to_third = False
		self.first_label = None
		self.second_label = None

	def clicked(self, qmodelindex):
		print("clicked")

	def set_position(self):
		self.xpos_window = self.main_window.pos().x()+self.main_window.frameGeometry().width()//4
		self.ypos_window = self.main_window.pos().y()+self.main_window.frameGeometry().height()//4
		self.width_window = self.main_window.frameGeometry().width()//2
		self.height_window = self.main_window.frameGeometry().height()//2
		self.setGeometry(self.xpos_window, self.ypos_window, self.width_window, self.height_window)

	def keyPressEvent(self, event):

		if event.key() == Qt.Key_Return:
			if not self.to_second and not self.to_third:
				self.first_label = self.list_widget.currentItem().text()
				self.list_widget_second.setFocus()
				self.to_second=True

		if event.key() == Qt.Key_Escape:
			self.to_second=False
			self.to_third=False
			self.first_label = None	
			self.second_label = None
			self.hide()
			self.main_window.setFocus()