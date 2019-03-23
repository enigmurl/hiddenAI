import pygame
from hiddenAI.sequential import * 
from hiddenAI.neat import NEAT
from hiddenAI.optimizers import * 
from hiddenAI.layers.main_layers import *
from hiddenAI.layers.activations import *
import random

WIDTH = 800
HEIGHT= 700
SCREEN = pygame.display.set_mode((WIDTH,HEIGHT))
class Pillar:
	def __init__(self,xvalue,opening_start):
		OPENINGHEIGHT = 200
		self.xvalue = xvalue
		self.opening_start = opening_start
		self.opening_end = opening_start+OPENINGHEIGHT
		self.delete = False

	def draw(self,bird_position_x):
		if (-100<=self.xvalue-bird_position_x<=WIDTH):#-100 is the width of a pillar
			pass
		elif self.xvalue-bird_position_x<-100:
			self.delete = True
			return
		else:
			return
		pygame.draw.rect(SCREEN,(255,255,255),pygame.Rect(self.xvalue-bird_position_x,0,100,self.opening_start))
		pygame.draw.rect(SCREEN,(255,255,255),pygame.Rect(self.xvalue-bird_position_x,self.opening_end,100,HEIGHT-self.opening_end))

	def detect_collision(self,bird_position,bird_size = 10):
		screen_x = self.xvalue - bird_position[0]
		bird_screen_position = (200,bird_position[1])
		if screen_x-bird_size <= bird_screen_position[0]<= screen_x+bird_size+100:#100 is the width of a pillar
			return bird_position[1] + bird_size> self.opening_end or bird_position[1] - bird_size < self.opening_start
class Bird:
	def __init__(self):
		self.yvalue = HEIGHT/2
		self.velocity = 0
		self.score = 0
		self.alive = True
		self.can_flap = 0	

	def tick(self,dt,flap = False):
		if self.alive:
			self.score += dt
		self.yvalue += self.velocity * dt
		if self.yvalue < 0 or self.yvalue > HEIGHT:
			self.alive = False
		self.draw()
		if flap and self.can_flap <= 0:
			self.flap()
			self.can_flap = 1/10
			return
		self.can_flap -= dt
		self.velocity += 750 * dt #gravity is 10 pixels per seconds^2

	def draw(self,bird_size = 10,x_center = 200):
		pygame.draw.rect(SCREEN,(125,0,125),pygame.Rect(x_center-bird_size,self.yvalue-bird_size,bird_size*2,bird_size*2))

	def flap(self):
		self.velocity = -300
	

num_pillars = 10000
num_birds = 50
model = Sequential(7,FullyConnected(3),Bias(),Sigmoid(),FullyConnected(2),Bias(),Sigmoid(),optimizer = None) 
neat = NEAT(model,num_per_generation = num_birds)
all_bird_nets = [Sequential(7,FullyConnected(3),Bias(),Sigmoid(),FullyConnected(2),Bias(),Sigmoid(),optimizer = None) for _ in range(num_birds)] 
all_real_birds = [Bird() for _ in range(num_birds)]
running = True
clock = pygame.time.Clock()
gen_num=0
world_x = 0
while running:
	pillars = [Pillar(750*x + 1000,random.randint(0,HEIGHT-200)) for x in range(num_pillars)]#pillars are spaced 500 apart,and the first is 1000 away from the bird
	print("GENERATION",neat.generation,"WORLD SCORE:",world_x,"MAX SCORE",max([bird.score for bird in all_real_birds]))
	world_x = 0
	gen_running = True
	all_bird_nets = neat.new_generation(all_bird_nets,[bird.score for bird in all_real_birds],mutation_max_size = 5)
	all_real_birds = [Bird() for _ in range(num_birds)]
	world_score = 0
	while gen_running:
		SCREEN.fill((0,0,0))
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
				gen_running = False
				break
		dead_pillars = []
		for pillar_num,pillar in enumerate(pillars):
			pillar.draw(world_x)
			if pillar.delete:
				dead_pillars.append(pillar_num)
		for pillar_num in dead_pillars:
			del pillars[pillar_num]
		first_pillar = pillars[0] 
		dt = clock.tick(24)/1000
		world_x += dt* 250
		alive = 0
		for bird_num,bird_net in enumerate(all_bird_nets):
			real_bird = all_real_birds[bird_num]
			if not real_bird.alive:
				continue
			real_bird.score = world_x
			real_bird.draw()
			if first_pillar.detect_collision((world_x,real_bird.yvalue)):
				real_bird.alive = False
				continue
			alive += 1
			
			result = bird_net.run(np.array([real_bird.yvalue/HEIGHT,1-real_bird.yvalue/HEIGHT,real_bird.velocity/100,(first_pillar.opening_start)/HEIGHT,1-(first_pillar.opening_start)/HEIGHT,(first_pillar.xvalue-world_x)/1000,real_bird.can_flap]))
			real_bird.tick(dt,result[0]>result[1])
		if alive ==0:
			gen_running = False
		pygame.display.flip()			
neat.save_model_to_file("stored_weights/flappy_bird")	
