from PyQt5.QtWidgets import QTreeWidgetItem, QTreeWidget, QWidget, QPushButton, QStyle, QSlider, QHBoxLayout, QVBoxLayout, QFileDialog, QGridLayout, QListWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl
import re


class ListDisplay(QWidget):

	def __init__(self, main_window):
		super().__init__()

		self.max_width = 300
		self.setMaximumWidth(self.max_width)

		self.main_window = main_window

		self.layout = QGridLayout()
		self.setLayout(self.layout)

		self.list_widget = QTreeWidget()
		self.list_widget.setHeaderHidden(True)
		# self.list_widget.setColumnCol(1)
		self.layout.addWidget(self.list_widget)

		self.list_widget.itemDoubleClicked.connect(self.doubleClicked)

	def doubleClicked(self, item):
		data = item.text(0)
		pattern = '(\d{0,2}:\d{0,2}).*'
		if re.match(pattern, data):
			time = re.search(pattern, data).group(1).split(':')
			# print(time)
			position = (int(time[0]) * 60 + int(time[1])) * 1000
			# row = self.list_widget.currentRow()
			# position = self.main_window.list_manager.event_list[row].position
			if self.main_window.media_player.play_button.isEnabled():
				self.main_window.media_player.set_position(position)
			self.main_window.setFocus()

	def display_list(self, event_list):
		self.list_widget.clear()
		current_label = None
		header_item = None
		for item_nbr, element in enumerate(event_list):
			if (current_label != element.get_label()):
				current_label = element.get_label()
				header_item = QTreeWidgetItem([current_label])
				self.list_widget.addTopLevelItem(header_item)
			child_item = QTreeWidgetItem([element.to_text()])
			header_item.addChild(child_item)