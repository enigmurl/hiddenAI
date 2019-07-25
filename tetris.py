import numpy  as np
import pygame as pg
from hiddenAI.layers.convolution import Convolution2D
from hiddenAI.layers.main_layers import *
from hiddenAI.layers.activations import *
from hiddenAI.neat import NEAT


maxpieces = 10000
generationsize = 50
model = None
neat = NEAT(model,generationsize)

class Gameboard:
	PIECEMAP = [np.array([[1,1],[1,1]]),
				np.array([[1],[1],[1],[1]]),
				np.array([[1,1],[0,1],[0,1]]),
				np.array([[0,1],[0,1],[1,1]]),
				np.array([[0,1],[1,1],[0,1]]),
				np.array([[0,1],[1,1],[1,0]]),
				np.array([[1,0],[1,1],[0,1]])
				]
			
	def __init__(self,rows,cols,pieceslist):
		self.rows = rows
		self.cols = cols
		self.gameboard = np.zeros((cols,rows))
		self.pieceslist = pieceslist

	def tick(self,screen):
		
	
	def nextpiece(self):
		return self.pieceslist.pop(0)
	
	def draw(self,screen):
		for x in range(self.cols):
			for y in range(self.rows):
				if self.gameboard[x,y] == 0:
					color = (200,200,200)
				elif self.gameboard[x,y] == 1:
					color = (200,175,175)
				else:
					color = (100,100,200)
				pg.draw.rect(screen,color,pg.Rect(x*10,y*10,9,9))
	
running = True
screen = pg.display.set_mode((800,800))
gb = Gameboard(30,10,[2])
while running:
	screen.fill((0,0,0))
	for event in pg.event.get():
		if event.type == pg.QUIT:
			running = False
			break
	gb.draw(screen)
	pg.display.flip()
