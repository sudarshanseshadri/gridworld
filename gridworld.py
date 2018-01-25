import random

class GridWorld():
	def __init__(self, size, start_state, terminal_state_d):
		self.size = size
		assert self.in_bounds(start_state)
		self.start = start_state
		self.terminal_dict = {}
		assert len(terminal_state_d.keys()) > 0
		for state in terminal_state_d:
			assert self.in_bounds(state)
		self.terminal_d = terminal_state_d
		self.curr_state = self.start
		self.done = (self.curr_state in self.terminal_d)
		assert not self.done 
		self.penalty = -0.1

	def get_state(self):
		return self.curr_state

	def reset(self):
		self.curr_state = self.start
		self.done = False

	def in_bounds(self, state):
		return (0 <= state[0] < self.size) and (0 <= state[1] < self.size)

	def do_action(self, action):
		assert not self.done
		state = self.curr_state
		if state in self.terminal_d:
			self.done = True
			return self.terminal_d[state], True
		assert self.in_bounds(state)
		assert 0 <= action < 4
		if action == 0: #right
			next_state = (state[0], state[1]+1)
		elif action == 1: #up
			next_state = (state[0]-1, state[1])
		elif action == 2: #left
			next_state = (state[0], state[1]-1)
		else: #down
			next_state = (state[0]+1, state[1])
		if not self.in_bounds(next_state):
			return self.penalty, False
		self.curr_state = next_state
		return self.penalty, False

	def actions(self):
		if self.done:
			return []
		else:
			return [0, 1, 2, 3]

	def state_size(self):
		return 2

	def action_size(self):
		return 4

if __name__ == '__main__':
	small_world = GridWorld(4, (0,0), {(1, 1):1, (3, 3):10})
	total = 0
	while not small_world.done:
		print "Current state: {}".format(small_world.get_state())
		action = random.choice(small_world.actions())
		print "Taking action: {}".format(action)
		reward, _ = small_world.do_action(action)
		total += reward
		print "Got reward: {}".format(reward)
	print "Finished at state {} with reward {}".format(small_world.get_state(), total)


