from mplsoccer.pitch import Pitch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
from scipy.spatial import distance
import matplotlib.animation as animation
import cv2


#================================================ FUNCTIONS START =====================================================

#updated belman equation
#n => new and p => present , heu => heuristic, euc_d => euclidean distance

def move(euc_d_max,
         heu_val_p, p,
         heu_val_n_1, p_1,
         heu_val_n_2, p_2,
         heu_val_n_3, p_3):
    values = []
    p_goal = (197, 134)
    panelty_y, panelty_x = range(70,190),range(140,180)
    euc_d_p = distance.euclidean(p,p_goal)
    euc_d_1 = distance.euclidean(p_1,p_goal)
    euc_d_2 = distance.euclidean(p_2,p_goal)
    euc_d_3 = distance.euclidean(p_3,p_goal)
    values.append((1-heu_val_n_1/255) * euc_d_1/euc_d_max + heu_val_p/255*euc_d_p/euc_d_max)
    values.append((1-heu_val_n_2/255) * euc_d_2/euc_d_max + heu_val_p/255*euc_d_p/euc_d_max)
    values.append((1-heu_val_n_3/255) * euc_d_3/euc_d_max + heu_val_p/255*euc_d_p/euc_d_max)
    for i in values:
    	if (i[0] not in panelty_x) and (i[1] not in panelty_y):
    		values.pop(i)
    	
    minimum = min(values)
    
    if minimum == values[0]:
        return p_1
    elif minimum == values[1]:
        return p_2
    elif minimum == values[2]:
        return p_3

def final_frame(image, 
                a_r_1_x,a_r_1_y,
                a_r_2_x,a_r_2_y,
                a_b_1_x,a_b_1_y,
                a_b_2_x,a_b_2_y,
                a_b_3_x,a_b_3_y, 
                ball_x,ball_y):
	img = np.copy(image)
	color = (0,0,255)
	cv2.rectangle(img, (a_r_1_x,a_r_1_y), (a_r_1_x+10,a_r_1_y+10), color, -1)
	cv2.rectangle(img, (a_r_1_x+10,a_r_1_y),(a_r_1_x+20,a_r_1_y+10), color, -1)
	cv2.rectangle(img,(a_r_1_x+10,a_r_1_y+10), (a_r_1_x+20,a_r_1_y+20),color, -1)
	
	cv2.rectangle(img, (a_r_2_x,a_r_2_y), (a_r_2_x+10,a_r_2_y+10), color, -1)
	cv2.rectangle(img, (a_r_2_x+10,a_r_2_y),(a_r_2_x+20,a_r_2_y+10), color, -1)
	cv2.rectangle(img,(a_r_2_x+10,a_r_2_y+10), (a_r_2_x+20,a_r_2_y+20),color, -1)
	
	color = (255,0,0)
	cv2.rectangle(img, (a_b_1_x,a_b_1_y), (a_b_1_x+10,a_b_1_y+10), color, -1)
	cv2.rectangle(img, (a_b_1_x+10,a_b_1_y),(a_b_1_x+20,a_b_1_y+10), color, -1)
	cv2.rectangle(img,(a_b_1_x+10,a_b_1_y+10), (a_b_1_x+20,a_b_1_y+20),color, -1)
	
	cv2.rectangle(img, (a_b_2_x,a_b_2_y), (a_b_2_x+10,a_b_2_y+10), color, -1)
	cv2.rectangle(img, (a_b_2_x+10,a_b_2_y),(a_b_2_x+20,a_b_2_y+10), color, -1)
	cv2.rectangle(img,(a_b_2_x+10,a_b_2_y+10), (a_b_2_x+20,a_b_2_y+20),color, -1)
	
	cv2.rectangle(img, (a_b_3_x,a_b_3_y), (a_b_3_x+10,a_b_3_y+10), color, -1)
	cv2.rectangle(img, (a_b_3_x+10,a_b_3_y),(a_b_3_x+20,a_b_3_y+10), color, -1)
	cv2.rectangle(img,(a_b_3_x+10,a_b_3_y+10), (a_b_3_x+20,a_b_3_y+20),color, -1)
		
	return(img)
	

def get_points(x1, y1, x2, y2):
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points

def agent_move(b_x, b_y,u):
	p = get_points(u[-11],u[-10],b_x,b_y)
	(a_r_1_x,a_r_1_y) = p[10]
	p = get_points(u[-9],u[-8],b_x,b_y)
	(a_r_2_x,a_r_2_y) = p[10]
	p = get_points(u[-7],u[-6],u[10],u[11])
	(a_b_1_x,a_b_1_y) = p[-10]
	p = get_points(u[-5],u[-4],b_x,b_y)
	(a_b_2_x,a_b_2_y) = p[10]
	p = get_points(u[-3],u[-2],b_x,b_y)
	(a_b_3_x,a_r_3_y) = p[10]
	
	return(a_r_1_x,a_r_1_y,
	a_r_2_x,a_r_2_y,
	a_b_1_x,a_b_1_y,
	a_b_2_x,a_b_2_y,
	a_b_3_x,a_b_3_y)
	

# Initialisation
def initialisation(soccer_ground):

	#number of rows and columns
	row, col = soccer_ground.shape[:2]
		
	#goal dimensions
	goal_y, goal_x = range(120,150),range(190,205) 
	
	#Panelty dimensions
	panelty_y, panelty_x = range(70,190),range(140,180) # panelty area dimensions 
	
	#agent world dimensions
	agent_world_x,agent_world_y = range(0,soccer_ground.shape[1]),range(0,soccer_ground.shape[0])#agent environment dimentions
	#agent playground dimensions
	agent_play_x,agent_play_y = range(10,185),range(10,256) #agent playground area dimensions

	p1 = (int((agent_world_x[0]+agent_world_x[-1])/2),
	      int((agent_world_y[0]+agent_world_y[-1])/2))
	
	p2 = (int((goal_x[0]+goal_x[-1])/2),
	      int((goal_y[0]+goal_y[-1])/2))
	
	#goal Center
	goal_center_x,goal_center_y = p2
	
	#maximum euclidean distance
	euc_d_max = distance.euclidean(p1, p2)

	v1 = []
	v2 = []
	for i in range(20,186):
	    v1.append(i)
	for j in range(20,247):
	    v2.append(j)
	
	#initial ball location
	ball_x, ball_y = 190,20 #ball initial point
	
	#Initial agent team red locations
	a_r_1_x, a_r_1_y = 180,0
	

	a_b_1_x, a_b_1_y = random.choice(panelty_x),random.choice(panelty_y)
	
	#Initial agent team blue locations
	a_r_2_x, a_r_2_y = random.choice(v1), random.choice(v2)
	
	a_b_2_x, a_b_2_y = random.choice(v1) , random.choice(v2) 

	a_b_3_x, a_b_3_y = random.choice(v1) , random.choice(v2) 
	
	#heuristic matrix
	center_x, center_y = (goal_x[0] + int(goal_x[-1]-goal_x[0]), 
		            goal_y[0] + int(goal_y[-1]-goal_y[0]))
	heu_matrix = np.zeros((row,col))
	heu_matrix[panelty_y[0]:panelty_y[-1],panelty_x[0]:panelty_x[-1]+10] = 200
	heu_matrix[agent_play_y[0]:agent_play_y[-1],10:20] = 10
	heu_matrix[agent_play_y[0]:agent_play_y[-1],20:40] = 20
	heu_matrix[agent_play_y[0]:agent_play_y[-1],40:60] = 30
	heu_matrix[agent_play_y[0]:agent_play_y[-1],60:80] = 40
	heu_matrix[agent_play_y[0]:agent_play_y[-1],80:100] = 60
	heu_matrix[agent_play_y[0]:agent_play_y[-1],100:120] = 80
	heu_matrix[agent_play_y[0]:agent_play_y[-1],120:140] = 100
	heu_matrix[50:70,140:] = 100
	heu_matrix[panelty_y[-1]:200,140:] = 100
	heu_matrix[10:50,140:] = 80
	heu_matrix[200:-10,140:] = 80
	heu_matrix[:,goal_x[0]:agent_world_x[-1]] = 0
	heu_matrix[goal_y[0]:goal_y[-1],goal_x[0]:goal_x[-1]] = 255
	#plt.gray()
	#plt.xticks([])
	#plt.yticks([])
	#plt.imshow(heu_matrix)
	#plt.show()

	
	return (row, col, 
	goal_x, goal_y, 
	panelty_y, panelty_x,
	agent_world_x,agent_world_y,
	agent_play_x,agent_play_y, 
	goal_center_x,goal_center_y,
	euc_d_max,
	ball_x, ball_y,
	a_r_1_x, a_r_1_y,a_r_2_x, a_r_2_y, a_b_1_x, a_b_1_y,a_b_2_x, a_b_2_y, a_b_3_x, a_b_3_y,
	heu_matrix)

def game(game_start,points,soccer_ground, u, ball_hold):
#game(game_start,v,soccer_ground,updated_values,ball_hold)
	if (game_start == 'True'):
		(row, col, 
		goal_x, goal_y, 
		panelty_y, panelty_x,
		agent_world_x,agent_world_y,
		agent_play_x,agent_play_y, 
		goal_center_x,goal_center_y,
		euc_d_max,
		ball_x, ball_y,
		a_r_1_x, a_r_1_y,a_r_2_x, a_r_2_y, a_b_1_x, a_b_1_y,a_b_2_x, a_b_2_y, a_b_3_x, a_b_3_y,
		heu_matrix) = initialisation(soccer_ground)
		c = (row, col, 
			goal_x, goal_y, 
			panelty_y, panelty_x,
			agent_world_x,agent_world_y,
			agent_play_x,agent_play_y, 
			goal_center_x,goal_center_y,
			euc_d_max,
			ball_x, ball_y,
			a_r_1_x, a_r_1_y,a_r_2_x, a_r_2_y, a_b_1_x, a_b_1_y,a_b_2_x, a_b_2_y, a_b_3_x, a_b_3_y,
			heu_matrix)
		points = get_points(a_r_1_x,a_r_1_y,goal_center_x,goal_center_y)
		
#		passing
#		if  (c[-9], c[-8]) or (c[-5], c[-4]) or (c[-3], c[-2])or (c[-7],c[-6]) not in points :
		if  (c[-5] in panelty_x and c[-4] in panelty_y) and (c[-3] in panelty_x and c[-2] in panelty_y):
			v = get_points(c[-11],c[-10],c[-9],c[-8])
			f = final_frame(soccer_ground,
					   c[-11], c[-10],
					   c[-9], c[-8],
					   c[-7], c[-6],
					   c[-5],c[-4],
					   c[-3],c[-2],
					   c[13],c[14])
			game_start = 'False'
			ball_hold = 'a_r_2'
			return(c,v,f,game_start,ball_hold)
		#direct goal
		elif (c[-5] not in panelty_x or c[-4] not in panelty_y) and (c[-3] not in panelty_x or c[-2] not in panelty_y):
			f = final_frame(soccer_ground,
		             c[-11], c[-10],
		             c[-9], c[-8],
		             c[-7], c[-6],
		             c[-5],c[-4],
		             c[-3],c[-2],
		             c[13],c[14])
			v = get_points(c[-11],c[-10],goal_center_x,goal_center_y)
			ball_hold = 'None' # to make condition for ball_hold = 0
			game_start = 'True'
			return (c,v,f,game_start,ball_hold)
		
		#moving
		else:
			if (c[-5], c[-4]) or (c[-3], c[-2]) or (c[-7],c[-6]) in points:
				v = get_points(c[-11],c[-10],c[-9],c[-8])
				f = final_frame(soccer_ground,
					   c[-11], c[-10],
					   c[-9], c[-8],
					   c[-7], c[-6],
					   c[-5],c[-4],
					   c[-3],c[-2],
					   c[13],c[14])
				game_start = 'False'
				ball_hold = 'a_r_2'
				return(c,v,f,game_start,ball_hold)
			else:
				(i,j) = (ball_x,ball_y)
				ball_x, ball_y = move(euc_d_max,
				         c[-1](i,j),
				         c[-1](i-1,j), (i-1,j),
				         c[-1](i,j+1),(i,j+1) ,
				         c[-1](i+1,j+1), (i+1,j+1))
				(a_r_1_x, a_r_1_y,
				a_r_2_x, a_r_2_y,
				a_b_1_x, a_b_1_y,
				a_b_2_x,a_b_2_y,
				a_b_3_x,a_b_3_y) = agent_move(ball_x,ball_y,c)
				f = final_frame(soccer_ground,
		           		a_r_1_x, a_r_1_y,
					a_r_2_x, a_r_2_y,
					a_b_1_x, a_b_1_y,
					a_b_2_x,a_b_2_y,
					a_b_3_x,a_b_3_y)
				v = (ball_x,ball_y)
				game_start = 'False'
				ball_hold = 'a_r_1'
				return(c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],
				ball_x, ball_y,a_r_1_x, a_r_1_y,a_r_2_x, a_r_2_y, a_b_1_x, a_b_1_y,a_b_2_x, a_b_2_y, 					a_b_3_x,a_b_3_y,c[-11],v,f,game_start,ball_hold)

	elif(game_start =='False' and ball_hold == 'a_r_1'):
#		condition for direct goal
		if  (u[-9], u[-8]) or (u[-5], u[-4]) or (u[-3], u[-2]) or (u[-7], u[-6])  not in points:
			f = final_frame(soccer_ground,
		             u[-11], u[-10],
		             u[-9], u[-8],
		             u[-7], u[-6],
		             u[-5],u[-4],
		             u[-3],u[-2],
		             u[13],u[14])
			v = get_points(u[-11],u[-10],u[10],u[11])
			c = u
			game_start = 'True'
			return(c,v,f,game_start,ball_hold)
			
		#when direct goal is blocked by opposition
		if (u[-9], u[-8]) or (u[-5], u[-4]) or (u[-3], u[-2]) or (u[-7],u[-6]) in points:
			if (u[-5], u[-4]) or (u[-3], u[-2]) or (u[-7],u[-6]) in points:
				v = get_points(u[-11],u[-10],u[-9],u[-8])
				c = u
				f = final_frame(soccer_ground,
					   c[-11], c[-10],
					   c[-9], c[-8],
					   c[-7], c[-6],
					   c[-5],c[-4],
					   c[-3],c[-2],
					   c[13],c[14])
				game_start = 'False'
				ball_hold = 'a_r_2'
				return(c,v,f,game_start,ball_hold)
			else:
				(i,j) = (ball_x,ball_y)
				c = u
				ball_x, ball_y = move(euc_d_max,
					         c[-1](i,j),
					         c[-1](i-1,j), (i-1,j),
					         c[-1](i,j+1),(i,j+1) ,
					         c[-1](i+1,j+1), (i+1,j+1))
				(a_r_1_x, a_r_1_y,
				a_r_2_x, a_r_2_y,
				a_b_1_x, a_b_1_y,
				a_b_2_x,a_b_2_y,
				a_b_3_x,a_b_3_y) = agent_move(ball_x,ball_y,c)
				f = final_frame(soccer_ground,
					a_r_1_x, a_r_1_y,
					a_r_2_x, a_r_2_y,
					a_b_1_x, a_b_1_y,
					a_b_2_x,a_b_2_y,
					a_b_3_x,a_b_3_y)
				v = (ball_x,ball_y)
				game_start = 'False'
				ball_hold = 'a_r_1'
				return(c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],
					ball_x, ball_y,a_r_1_x, a_r_1_y,a_r_2_x, a_r_2_y, a_b_1_x, a_b_1_y,a_b_2_x, a_b_2_y, 					a_b_3_x,a_b_3_y,c[-11],v,f,game_start,ball_hold)
					
	elif(game_start =='False' and ball_hold == 'a_r_2'):
		#condition for direct goal
		if  (u[-9], u[-8]) or (u[-5], u[-4]) or (u[-3], u[-2]) or (u[-7], u[-6])  not in points:
			f = final_frame(soccer_ground,
		             u[-11], u[-10],
		             u[-9], u[-8],
		             u[-7], u[-6],
		             u[-5],u[-4],
		             u[-3],u[-2],
		             u[13],u[14])
			v = get_points(u[-9],u[-8],u[10],u[11])
			c = u
			game_start = 'True'
			return(c,v,f,game_start,ball_hold)
		if (u[-9], u[-8]) or (u[-5], u[-4]) or (u[-3], u[-2]) or (u[-7],u[-6]) in points:
			if (u[-5], u[-4]) or (u[-3], u[-2]) or (u[-7],u[-6]) in points:
				v = get_points(u[-11],u[-10],u[-9],u[-8])
				c = u
				f = final_frame(soccer_ground,
					   c[-11], c[-10],
					   c[-9], c[-8],
					   c[-7], c[-6],
					   c[-5],c[-4],
					   c[-3],c[-2],
					   c[13],c[14])
				game_start = 'False'
				ball_hold = 'a_r_1'
				return(c,v,f,game_start,ball_hold)
			else:
				(i,j) = (ball_x,ball_y)
				c = u
				ball_x, ball_y = move(euc_d_max,
					         c[-1](i,j),
					         c[-1](i-1,j), (i-1,j),
					         c[-1](i,j+1),(i,j+1) ,
					         c[-1](i+1,j+1), (i+1,j+1))
				(a_r_1_x, a_r_1_y,
				a_r_2_x, a_r_2_y,
				a_b_1_x, a_b_1_y,
				a_b_2_x,a_b_2_y,
				a_b_3_x,a_b_3_y) = agent_move(ball_x,ball_y,c)
				f = final_frame(soccer_ground,
					a_r_1_x, a_r_1_y,
					a_r_2_x, a_r_2_y,
					a_b_1_x, a_b_1_y,
					a_b_2_x,a_b_2_y,
					a_b_3_x,a_b_3_y)
				v = (ball_x,ball_y)
				game_start = 'False'
				ball_hold = 'a_r_2'
				return(c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],
					ball_x, ball_y,a_r_1_x, a_r_1_y,a_r_2_x, a_r_2_y, a_b_1_x, a_b_1_y,a_b_2_x, 						a_b_2_y, 	a_b_3_x,a_b_3_y,c[-11],v,f,game_start,ball_hold)


	elif(game_start =='False' and ball_hold == 'None'):
		ball_hold = 'a_r_1'
		v=0
		u=0
		(row, col, 
		goal_x, goal_y, 
		panelty_y, panelty_x,
		agent_world_x,agent_world_y,
		agent_play_x,agent_play_y, 
		goal_center_x,goal_center_y,
		euc_d_max,
		ball_x, ball_y,
		a_r_1_x, a_r_1_y,a_r_2_x, a_r_2_y, a_b_1_x, a_b_1_y,a_b_2_x, a_b_2_y, a_b_3_x, a_b_3_y,
		heu_matrix) = initialisation(soccer_ground)
		c = (row, col, 
			goal_x, goal_y, 
			panelty_y, panelty_x,
			agent_world_x,agent_world_y,
			agent_play_x,agent_play_y, 
			goal_center_x,goal_center_y,
			euc_d_max,
			ball_x, ball_y,
			a_r_1_x, a_r_1_y,a_r_2_x, a_r_2_y, a_b_1_x, a_b_1_y,a_b_2_x, a_b_2_y, a_b_3_x, a_b_3_y,
			heu_matrix)
		points = get_points(a_r_1_x,a_r_1_y,goal_center_x,goal_center_y)
		
		#direct goal
		if  (c[-9], c[-8]) or (c[-5], c[-4]) or (c[-3], c[-2])or (c[-7],c[-6]) not in points:
			f = final_frame(soccer_ground,
		             c[-11], c[-10],
		             c[-9], c[-8],
		             c[-7], c[-6],
		             c[-5],c[-4],
		             c[-3],c[-2],
		             c[13],c[14])
			v = get_points(c[-11],c[-10],ball_x,ball_y)
			ball_hold = 'None' # to make condition for ball_hold = 0
			game_start = 'True'
			return (c,v,f,game_start,ball_hold)
		
		#passing or moving
		if (c[-9], c[-8]) or (c[-5], c[-4]) or (c[-3], c[-2]) or (c[-7],c[-6]) in points:
			if (c[-5], c[-4]) or (c[-3], c[-2]) or (c[-7],c[-6]) in points:
				v = get_points(c[-11],c[-10],c[-9],c[-8])
				f = final_frame(soccer_ground,
					   c[-11], c[-10],
					   c[-9], c[-8],
					   c[-7], c[-6],
					   c[-5],c[-4],
					   c[-3],c[-2],
					   c[13],c[14])
				game_start = 'False'
				ball_hold = 'a_r_2'
				return(c,v,f,game_start,ball_hold)
			else:
				(i,j) = (ball_x,ball_y)
				ball_x, ball_y = move(euc_d_max,
				         c[-1](i,j),
				         c[-1](i-1,j), (i-1,j),
				         c[-1](i,j+1),(i,j+1) ,
				         c[-1](i+1,j+1), (i+1,j+1))
				(a_r_1_x, a_r_1_y,
				a_r_2_x, a_r_2_y,
				a_b_1_x, a_b_1_y,
				a_b_2_x,a_b_2_y,
				a_b_3_x,a_b_3_y) = agent_move(ball_x,ball_y,c)
				f = final_frame(soccer_ground,
		           		a_r_1_x, a_r_1_y,
					a_r_2_x, a_r_2_y,
					a_b_1_x, a_b_1_y,
					a_b_2_x,a_b_2_y,
					a_b_3_x,a_b_3_y)
				v = (ball_x,ball_y)
				game_start = 'False'
				ball_hold = 'a_r_1'
				return(c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],c[10],c[11],c[12],
				ball_x, ball_y,a_r_1_x, a_r_1_y,a_r_2_x, a_r_2_y, a_b_1_x, a_b_1_y,a_b_2_x, a_b_2_y, 					a_b_3_x,a_b_3_y,c[-11],v,f,game_start,ball_hold)
		
		
#============================================= FUNCTIONS END =====================================================

#================================================ main =====================================================
soccer_ground = plt.imread('soccer_ground.png')
soccer_ground = cv2.resize(soccer_ground,(205,266))
s1,s2 = soccer_ground.shape[:2]
v = 0
updated_values = 0
game_start = 'True'
ball_hold = 'a_r_1'
#=====================================================================================================>>>
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
##writer = None
#(h, w) = (None, None)
#zeros = None
#writer = cv2.VideoWriter('Jaw Line.mkv', fourcc, 10,(s2, s1 ), True)
#writer = cv2.VideoWriter('output_1.avi',fourcc, 10, (s2,s2))
#out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, (s2,s1))
#=============================================== THE RUN ==============================================
final_set = []
while(True):
	#===================================LogiC cOdE =================================
	if game_start == 'True':
		(row, col,
		goal_x, goal_y, 
		panelty_y, panelty_x,
		agent_world_x,agent_world_y, 
		agent_play_x,agent_play_y, 
		goal_center_x,goal_center_y,
		euc_d_max,
		ball_x, ball_y,
		a_r_1_x, a_r_1_y,a_r_2_x, a_r_2_y, a_b_1_x, a_b_1_y,a_b_2_x, a_b_2_y, a_b_3_x, a_b_3_y,
		heu_matrix),v,f,game_start,ball_hold  = game(game_start,v,soccer_ground,updated_values,ball_hold)
		print('if section')
		print(game_start, ball_hold)
	else:	
		(row, col,
		goal_x, goal_y, 
		panelty_y, panelty_x,
		agent_world_x,agent_world_y, 
		agent_play_x,agent_play_y, 
		goal_center_x,goal_center_y,
		euc_d_max,
		ball_x, ball_y,
		a_r_1_x, a_r_1_y,a_r_2_x, a_r_2_y, a_b_1_x, a_b_1_y,a_b_2_x, a_b_2_y, a_b_3_x, a_b_3_y,
		heu_matrix),v,f,game_start,ball_hold = game(game_start,v,soccer_ground, updated_values, ball_hold)
		print('else section')
		print(game_start,  ball_hold)
	updated_values = (row, col,
			goal_x, goal_y, 
			panelty_y, panelty_x,
			agent_world_x,agent_world_y, 
			agent_play_x,agent_play_y, 
			goal_center_x,goal_center_y,
			euc_d_max,
			ball_x, ball_y,
			a_r_1_x, a_r_1_y,a_r_2_x, a_r_2_y, a_b_1_x, a_b_1_y,a_b_2_x, a_b_2_y, a_b_3_x, a_b_3_y,
			heu_matrix) 
	print(len(v))
		
	for i in range(10,len(v)):
		frame = np.copy(f)
		cv2.circle(frame,(v[i][0],v[i][1]) ,5, (0, 0, 0), -1)		
		final_set.append(frame)
		cv2.imshow('Frame',frame)
		cv2.waitKey(15)
	cv2.waitKey(1000)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
#writer.write(frame)


#for i in final_set:
#	writer.write(i)
		
cv2.destroyAllWindows()
#out.release()
#writer.release()
#writer1.release()


