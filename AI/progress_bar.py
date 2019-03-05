import sys 
class ProgressBar:
	def __init__(self,start_message = "progress: ",total = 1,end_message = " ",total_chars = 40):
		self.start_message = start_message 
		self.end_message = end_message
		self.total = total
		self.total_chars = total_chars
		self.current_iter_count = 0
		self.current_char_count = 0	
		sys.stdout.write(start_message)
		sys.stdout.write("[%s]" % (" " * total_chars))
		sys.stdout.write(end_message)
		sys.stdout.flush()
		sys.stdout.write("\b" * (total_chars + len(end_message)+1))

	def update(self):
		self.current_iter_count += 1
		if self.current_iter_count//(self.total/self.total_chars) > self.current_char_count:
			self.current_char_count += 1
			sys.stdout.write("-")
			sys.stdout.flush()

		if self.total == self.current_iter_count: 
			sys.stdout.write("\n")

if __name__ == "__main__":
	import time
	a = ProgressBar("start ",total = 100,total_chars = 40,end_message = " end")
	for i in range(100):
		time.sleep(0.1)
		a.update()
