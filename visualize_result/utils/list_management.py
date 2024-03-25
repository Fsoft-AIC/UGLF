from utils.event_class import Event
import json
import os

class ListManager:

	def __init__(self):
		self.event_list = list()

	def create_list_from_json(self, path, half, filter_score=0):
		self.event_list.clear()
		self.event_list = self.read_json(path, half, filter_score)
		self.sort_list()

	def get_event_list(self):
		return self.event_list

	def delete_event(self, index):
		self.event_list.pop(index)
		self.sort_list()

	def add_event(self, event):
		self.event_list.append(event)
		self.sort_list()

	def sort_list(self):
		position = list()
		for event in self.event_list:
			position.append(event.position)
		self.event_list = [x for _,x in sorted(zip(position,self.event_list), key = lambda x: (x[1].get_label(), x[0]))]
		
	def read_json(self, path, half, filter_score):
		event_list = list()
		with open(path) as file:
			data = json.load(file)["predictions"]

			for event in data:
				tmp_half = int(event["gameTime"][0])
				score = round(float(event['confidence']), 2)
				if (score < filter_score):
					continue
				if tmp_half == half:
					tmp_time = event["gameTime"][4:]
					tmp_position = 0
					if "position" in event:
						tmp_position = int(event["position"])
					else:
						tmp_position = int((int(tmp_time[0:2])*60 + int(tmp_time[3:]))*1000)
					tmp_label = event["label"]
					event_list.append(Event(tmp_label, tmp_half, tmp_time, tmp_position, score))
		return event_list
