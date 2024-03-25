class Event:

	def __init__(self, label=None, half=None, time=None, position=None, score=0):

		self.label = label
		self.half = half
		self.time = time
		self.position = position
		self.score = score

	def to_text(self):
		return self.time + " (score: " + str(self.score) + ")"

	def get_label(self):
		return self.label

	def __lt__(self, other):
		self.position < other.position

def ms_to_time(position):
	minutes = int(position//1000)//60
	seconds = int(position//1000)%60
	return str(minutes).zfill(2) + ":" + str(seconds).zfill(2)