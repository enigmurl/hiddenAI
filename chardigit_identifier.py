import pygame as pg
from hiddenAI.sequential import Sequential
from hiddenAI.layers.convolution import Convolution2D
from hiddenAI.layers.pooling import MaxPooling2D
from hiddenAI.layers.main_layers import *
from hiddenAI.layers.activations import *
import numpy as np
import sys

WIDTH = 1000
HEIGHT = 800
SCREEN = pg.display.set_mode((WIDTH,HEIGHT))

def text(prediction,x,y,size,font = "Helvetica",color = (255,255,255)):
	text = str(prediction)
	font = pg.font.SysFont(font, size)
	fontsize = font.size(text)
	text = font.render(text, True, color)
	SCREEN.blit(text, (x-fontsize[0]/2, y-fontsize[1]/2))

pg.init()
middle_x,middle_y = WIDTH/2,HEIGHT/2
boundaryW,boundaryH = 28*(WIDTH/3)//28,28*(WIDTH/3)//28
boundaryrect = pg.Rect(middle_x-boundaryW/2,middle_y-boundaryH/2,boundaryW,boundaryH)
resetrect =    pg.Rect(middle_x-boundaryW/2,middle_y+boundaryH/2 + 10,boundaryW/2 - 5,50)
guessrect =    pg.Rect(middle_x+5,middle_y+boundaryH/2 + 10,boundaryW/2 - 5,50)
running = True
mode = sys.argv[1]
character_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
digit_map = "0123456789"
pixelarray = np.zeros((1,28,28))
lastprediction = "NONE"
if mode == "character":
	model = Sequential((1,28,28),
					Convolution2D(num_filters = 32,filter_size = (3,3),stride = (3,3),pad = 1),
					MaxPooling2D(pooling_size = (2,2),stride = (2,2)),
					ReLU(), 
					Convolution2D(num_filters = 64,filter_size = (3,3),stride = (3,3)),
					MaxPooling2D(pooling_size = (2,2),stride = (2,2)),
					ReLU(), 
					FullyConnected(128),
					Bias(),
					ReLU(),
					FullyConnected(62),
					Bias(),
					Softmax(),optimizer = None)
	model.open_from_file("stored_weights/characterweights")
else:
	model = Sequential((1,28,28),
				Convolution2D(num_filters = 32,filter_size = (3,3),stride = (3,3)),
				Bias(),	
				ReLU(),	
				MaxPooling2D(pooling_size = (2,2),stride = (2,2)),
				Convolution2D(num_filters = 64, filter_size = (3,3),stride = (3,3)),
				Bias(),
				ReLU(),
				MaxPooling2D(pooling_size = (2,2),stride = (2,2)),
				FullyConnected(128),
				Bias(),
				ReLU(),
				FullyConnected(10),
				Bias(),
				Softmax(),optimizer = None)
	model.open_from_file("stored_weights/digitweights")
while running:
	SCREEN.fill((0,0,0))
	for event in pg.event.get():
		if event.type == pg.QUIT:
			running = False
			break
	if pg.mouse.get_pressed()[0] or pg.mouse.get_pressed()[1]:
		mosposx,mosposy = pg.mouse.get_pos()
		if middle_x-boundaryW/2 <= mosposx <= middle_x-5 and middle_y+boundaryH/2+10<= mosposy <= middle_y+boundaryH/2+60:
			pixelarray = np.zeros((1,28,28))
		elif middle_x+5<=mosposx <= middle_x+boundaryW/2 and middle_y+boundaryH/2+10<= mosposy <= middle_y+boundaryH/2+60:
			result = model.run(pixelarray)
			maxresult = -1
			maxind = -1
			for ind,val in enumerate(result):
				if val > maxresult:
					maxresult = val
					maxind = ind
			lastprediction = character_map[maxind]
		elif middle_x-boundaryW/2 <= mosposx <= middle_x+boundaryW/2:
			if middle_y-boundaryH/2 <= mosposy <= middle_y+boundaryH/2:
				loc = int(28*(mosposx-(middle_x-boundaryW/2))/boundaryW),int(28*(mosposy-(middle_y-boundaryH/2))/boundaryH)#integer division doesnt work for sum reason
				pixelarray[0,loc[1],loc[0]] += 0.8
				locs = ((loc[0]+1,loc[1]),(loc[0]-1,loc[1]),(loc[0],loc[1]+1),(loc[0],loc[1]-1))	
				for loc in locs:
					try:
						pixelarray[0,loc[1],loc[0]] += 0.3
					except:
						pass
				pixelarray = np.clip(pixelarray,0,1)
	for y,row in enumerate(pixelarray[0]):
		for x,val in enumerate(row):
			color = (val*255,val*255,val*255)
			gridrect = pg.Rect(middle_x - boundaryW/2 + x * boundaryW//28,middle_y - boundaryH/2 + y * boundaryH//28,boundaryW//28,boundaryH//28)
			pg.draw.rect(SCREEN,color,gridrect)
	text("RESET",middle_x - boundaryW/4,middle_y+boundaryH/2+35,30)
	text("GUESS",middle_x + boundaryW/4,middle_y+boundaryH/2+35,30)
	text("PREDICTION",middle_x,middle_y+boundaryH/2+85,30)
	text(lastprediction,middle_x,middle_y+boundaryH/2+115,30)
	pg.draw.rect(SCREEN,(255,255,255),boundaryrect,2)
	pg.draw.rect(SCREEN,(255,255,255),resetrect,2)
	pg.draw.rect(SCREEN,(255,255,255),guessrect,2)
	pg.display.flip()
