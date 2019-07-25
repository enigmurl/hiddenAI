import numpy  as np
import pygame as pg
from hiddenAI.sequential import Sequential
from hiddenAI.layers.convolution import Convolution2D
from hiddenAI.layers.main_layers import *
from hiddenAI.layers.activations import *
from hiddenAI.neat import NEAT
class GameBoard:
	def __init__(self):
		self.scalar = 3
		self.inner = self.scalar * np.array([[0,0],[40,0],[40,10],[100,10],[132,58],[108,93],[0,40]])
		self.outer = self.scalar * np.array([[0,-18],[40,-18],[100,-8],[150,58],[130,112],[-18,58],[-18,-18]])
		self.outsidepoint = self.computeoutsidepoint()

	def computeoutsidepoint(self):
		return (200,200)
	
	def draw(self,screen,dCord):
		pg.draw.lines(screen,(255,255,255),True,self.outer - dCord)
		pg.draw.lines(screen,(255,255,255),True,self.inner - dCord)

	def onSegment(self,p,q,r):
		return q[0] <= max(p[0],r[0]) and q[0] >= min(p[0],r[0]) and  q[1] <= max(p[1],r[1]) and q[1] >= min(p[1],r[1]) 
	
	def orientation(self,p,q,r):
		val = (q[1] - p[1]) * (r[0]-q[0]) - (q[0] - p[0]) * (r[1]-q[1])
		if val == 0:
			return val
		return 1 if val>0 else 2

	def intersect(self,p1,q1,p2,q2):
		o1 = self.orientation(p1,q1,p2)
		o2 = self.orientation(p1,q1,q2)
		o3 = self.orientation(p2,q2,p1)
		o4 = self.orientation(p2,q2,q1)
	
		if (o1 != o2 and o3 != o4):
			return True

		if (o1 == 0 and self.onSegment(p1,p2,q1)):
			return True
	
		if (o2 == 0 and self.onSegment(p1,q2,q1)):
			return True
		
		if (o3 == 0 and self.onSegment(p2,p1,q2)):
			return True
		
		if (o4 == 0 and self.onSegment(p2,q1,q2)):
			return True
		return False

	def isInside(self,cord):
		count = 0
		for i in range(-1,len(self.outer)-1):
			a1 = self.outer[i]
			a2 = self.outer[i+1]
			if self.intersect(a1,a2,cord,self.outsidepoint):
				count += 1
		for i in range(-1,len(self.inner)-1):
			a1 = self.inner[i]
			a2 = self.inner[i+1]
			if self.intersect(a1,a2,cord,self.outsidepoint):
				count += 1
		return count%2 ==1

class Car:
	def __init__(self,x,y):
		self.x = x
		self.y = y
		self.start = (x,y)
		self.angle = 0
		self.engine = 0
		self.speed = 10
		self.alive = True
		self.length = 10
		self.distscore = 0
		self.travscore = 0	
		self.score = 1	

	def turn(self,angle):
		self.angle += angle
	
	def accl(self,change):
		self.engine = np.clip(self.engine+change,0,1)

	def findpoints(self,dcord):
		angles = [self.angle,self.angle + 0.349066,self.angle - 0.349066,self.angle + 0.698132,self.angle - 0.698132]
		lengths = [self.length*10,self.length*8,self.length*8,self.length*6,self.length*6]
		pnts = []
		for ind,angle in enumerate(angles):
			x = (self.x + np.cos(angle) * lengths[ind]) - dcord[0]
			y = (self.y + np.sin(angle) * lengths[ind]) - dcord[1]
			pnts.append((x,y))
		return pnts
	def line_intersection(self,line1, line2):
		xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
		ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

		def det(a, b):
			return a[0] * b[1] - a[1] * b[0]

		div = det(xdiff, ydiff)
		if div == 0:
			return line2[1]
			raise Exception('lines do not intersect')

		d = (det(*line1), det(*line2))
		x = det(d, xdiff) / div
		y = det(d, ydiff) / div
		return x, y

	def dist(self,line):
		return ((line[0][0]-line[1][0]) ** 2 + (line[0][1]-line[1][1])**2 )**0.5

	def intersects(self,line,gameboard):
		minLength = self.dist(line)
		cord = line[1]
		for i in range(-1,len(gameboard.outer)-1):
			a1 = gameboard.outer[i]
			a2 = gameboard.outer[i+1]
			if gameboard.intersect(a1,a2,line[0],line[1]):
				x,y = self.line_intersection((a1,a2),line)
				totaldist = self.dist((line[0],(x,y)))
				if totaldist < minLength:
					minLength = totaldist
					cord = (x,y)	
		for i in range(-1,len(gameboard.inner)-1):
			a1 = gameboard.inner[i]
			a2 = gameboard.inner[i+1]
			if gameboard.intersect(a1,a2,line[0],line[1]):
				x,y = self.line_intersection((a1,a2),line)
				totaldist = self.dist((line[0],(x,y)))
				if totaldist < minLength:
					minLength = totaldist
					cord = (x,y)
		return minLength/self.dist(line),cord
	
	def tick(self,screen,dt,gameboard,dcord,net):
		if not gameboard.isInside((self.x,self.y)):
			self.alive = False#CHANGE TO FALSE
		self.x += np.cos(self.angle) * self.engine * self.speed
		self.y += np.sin(self.angle) * self.engine * self.speed
		pnts = self.findpoints(dcord)
		c1 = (self.x - dcord[0],self.y-dcord[1])
		lengths = [self.engine]
		for pnt in pnts:
			length,cor =  self.intersects(((self.x,self.y),(pnt[0]+dcord[0],pnt[1]+dcord[1])),gameboard)
			lengths.append(length)	
			#pg.draw.line(screen,(255,120,0),c1,(cor[0]-dcord[0],cor[1]-dcord[1]))
		cord1 = self.x - dcord[0] + self.length*np.cos(self.angle),self.y-dcord[1]+self.length*np.sin(self.angle)
		cord2 = self.x - dcord[0] + self.length*np.cos(self.angle + 2.79253),self.y-dcord[1]+self.length*np.sin(self.angle + 2.79253)
		cord3 = self.x - dcord[0] + self.length*np.cos(self.angle + 3.49066),self.y-dcord[1]+self.length*np.sin(self.angle + 3.49066)
		dEngine,dAngle = net.run(np.array(lengths))
		self.accl((dEngine - 0.5)/3)
		self.travscore += self.engine * dt
		distscore  =self.dist((self.start,(self.x,self.y)))
		if distscore>self.distscore:
			self.distscore = distscore
		self.score = self.travscore * self.distscore
		self.angle += (dAngle - 0.5)/10
		pg.draw.polygon(screen,(255,255,0),[cord1,cord2,cord3])
 
gb = GameBoard()

numcars = 25
model = Sequential(6,
					FullyConnected(4),
					Bias(),
					Sigmoid(),
					FullyConnected(2),
					Bias(),
					Sigmoid(),optimizer = None)
					
neat = NEAT(model,numcars)
startx,starty = -10,-11
running = True
WIDTH,HEIGHT = 800,800
screen = pg.display.set_mode((WIDTH,HEIGHT))
clock = pg.time.Clock()
dcord = np.array([-WIDTH/2,-HEIGHT/2])
cars = [Car(startx,starty) for _ in range(numcars)]
nets = [Sequential(6,
					FullyConnected(4),
					Bias(),
					Sigmoid(),
					FullyConnected(2),
					Bias(),
					Sigmoid(),optimizer = None) for _ in range(numcars)]
while running:
	genRunning = True
	nets = neat.new_generation(nets,[car.score for car in cars],mutation_max_size = 10)	
	print("NEW GENERATION",neat.generation,max([car.score for car in cars]))
	cars = [Car(startx,starty) for _ in range(numcars)]
	genTime = 0
	while genRunning:
		dt = clock.tick()/1000
		mCar = 0
		mval = -1
		
		for ind,car in enumerate(cars):
			if car.score>mval and car.alive:
				mCar = car
				mval = car.score
		try:
			dcord = np.array([mCar.x-WIDTH/2,mCar.y-HEIGHT/2])
		except:
			pass
		genTime += dt
		screen.fill((0,0,0))
		for event in pg.event.get():
			if event.type == pg.QUIT:
				genRunning = False
				running = False
				break
		numalive = 0
		for ind,net in enumerate(nets):

			car = cars[ind]
			if not car.alive:
				continue
			numalive += 1
			car.tick(screen,dt,gb,dcord,net)
		if numalive == 0 or genTime > 20:
			genRunning = False
			break	
		gb.draw(screen,dcord)
		pg.display.flip()

neat.save_model_to_file("stored_weights/racecar")
