# In this script we are building a Neural Network (Multi-layer Perceptron model) which can have variable number of Hidden layers
# We have implemented backpropogation, forward-pass and referred to "https://github.com/akshayush/FeedForward-and-BackPropagation-DNN-for-multiclass-classification/blob/master/FeedNeural_Network_on_Moon_Data.ipynb"
# the following code as a reference to build our network.
# The script also gives a graphical interface to the user of how the network is functioning but with a single hidden layer 
# We have used pygame for the GUI for this example

import pygame
import random
import matplotlib
import matplotlib.pyplot as plt
import pylab
import numpy as np
import time


#Initializing all the global variables
xdata=[]
ydata=[]
pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Times New Roman', 40)
BLACK = ( 0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = ( 255, 0, 0)
BLUE = ( 0, 0, 255)
GOLD=( 230, 215, 0)
PINK=(231,161,176)
COLOR_INACTIVE = pygame.Color('lightskyblue3')
COLOR_ACTIVE = pygame.Color('dodgerblue2')
FONT = pygame.font.Font(None, 32)
screen = pygame.display.set_mode((1500,1000)) 
pygame.display.set_caption('Perceptron GUI') 
clock = pygame.time.Clock() 
#weight=0
background_image = pygame.image.load("./bg1.jpg").convert()
Xa1=0
Xa2=0
white = (255, 255, 255)


# Class to build a Neural Network with variable number of hidden layers
class NeuralNetwork():
  global Xa1,Xa2,y_true
  def __init__(self,n_inputs,n_outputs,hidden_sizes=[2],w1=0.1,w3=0.4,w2=0.5,w4=0.3,w5=0.2,w6=0.6):
    self.nx=n_inputs
    self.ny=n_outputs
    self.nh=len(hidden_sizes)
    self.sizes=[self.nx]+hidden_sizes+[self.ny]
    self.W={}
    self.learning_parameter=0.7

    for i in range(self.nh+1):
      self.W[i+1]=np.random.randn(self.sizes[i],self.sizes[i+1])

    self.W[self.nh+1]=np.array([np.array([w5]),np.array([w6])])
    self.W[self.nh]=np.array([np.array([w1,w2]),np.array([w3,w4])])

# sigmoid function
  def sigmoid(self,x):
    return 1.0/(1.0+np.exp(-x))

# softmax function 
  def softmax(self,x):
    exps=np.exp(x)
    return exps/np.sum(exps)

# this function performs forward propogation
  def forward_pass(self,x):
    self.A={}
    self.H={}
    self.H[0]=x.reshape(1,-1)
    for i in range(self.nh):
      self.A[i+1]=np.matmul(self.H[i],self.W[i+1])
      self.H[i+1]=self.sigmoid(self.A[i+1])
    self.A[self.nh+1]=np.matmul(self.H[self.nh],self.W[self.nh+1])
    self.H[self.nh+1]=self.sigmoid(self.A[self.nh+1])
    return self.H[self.nh+1]

# Makes prediction on the basis of forward pass
  def predict(self,X):
    Y_pred=[]
    for x in X:
      y_pred=self.forward_pass(x)
      Y_pred.append(y_pred)
    return np.array(Y_pred).squeeze()

# Makes prediction for hidden layer based on forward propogation
  def predict_H(self,X,layer,neuron):
    Y_pred=[]
    for x in X:
      y_pred=self.forward_pass(x)
      Y_pred.append([item[neuron] for item in self.H[layer]])
    return np.array(Y_pred)

  def grad_sigmoid(self,x):
    return x*(1-x)

  def grad_softmax(self,x):
    return x*(1-x)

# Loss function
  def loss_function(self,x,y_true):
    y=self.predict(x)
    return 0.5*(y-y_true)**2

# Gradient descent
  def grad(self,x,y):
    self.forward_pass(x)
    self.dW={}
    self.dH={}
    self.dA={}
    L=self.nh+1
    self.dA[L]=np.multiply((self.H[L]-y),self.grad_sigmoid(self.H[L]))
    for k in range(L,0,-1):
      self.dW[k]=np.matmul(self.H[k-1].T,self.dA[k])
      self.dH[k-1]=np.matmul(self.dA[k],self.W[k].T)
      self.dA[k-1]=np.multiply(self.dH[k-1],self.grad_sigmoid(self.H[k-1]))

# Updating weights
  def grad_update(self):
    for k in range(self.nh+1,0,-1):
      self.W[k]=self.W[k]-self.learning_parameter*self.dW[k]

# Taking user inputs 
class InputBox:
    i=0
    def __init__(self, x, y, w, h, text=''):  
        InputBox.i+=1      
        self.j=InputBox.i-1
        self.rect = pygame.Rect(x, y, w, h)
        self.color = COLOR_INACTIVE
        self.col = WHITE
        self.text = text
        self.txt_surface = FONT.render(text, True, self.col)
        self.active = False

# GUI
    def handle_event(self, event):
        global Xa1,Xa2,y_true
        if event.type == pygame.MOUSEBUTTONDOWN:
            # If the user clicked on the input_box rect.
            if self.rect.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
            # Change the current color of the input box.
            self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    print(self.j)
                    if self.j<6:
                        W[self.j]=float(self.text)                    
                    elif self.j==6:
                        Xa1=float(self.text)
                        print(Xa1)
                    elif self.j==7:
                        Xa2=float(self.text)
                        print(Xa2)
                    elif self.j==8:
                        y_true=int(self.text)
                        print(y_true)
                    print(self.text)
                    
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                # Re-render the text.
                self.txt_surface = FONT.render(self.text, True, self.col)

    def update(self):
        # Resize the box if the text is too long.
        width = max(200, self.txt_surface.get_width()+10)
        self.rect.w = width

    def draw(self, screen):
        # Blit the text.
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        # Blit the rect.
        pygame.draw.rect(screen, self.color, self.rect, 2)

def initial():
    global quit,d,W
    mouse=pygame.mouse.get_pos()   
    click=pygame.mouse.get_pressed()
    screen.fill(BLACK)
    screen.blit(background_image, [0, 0])
    textsurface = myfont.render('Enter Initial Weights: ', False, WHITE)
    screen.blit(textsurface,(620,10))

    clock = pygame.time.Clock()
    textsurface = myfont.render('W1: ', False, WHITE)
    screen.blit(textsurface,(100,120))
    input_box1 = InputBox(200, 140, 140, 32)
    textsurface = myfont.render('W2: ', False, WHITE)
    screen.blit(textsurface,(100,220))
    input_box2 = InputBox(200, 240, 140, 32)
    textsurface = myfont.render('W3: ', False, WHITE)
    screen.blit(textsurface,(100,320))
    input_box3 = InputBox(200, 340, 140, 32)
    textsurface = myfont.render('W4: ', False, WHITE)
    screen.blit(textsurface,(100,420))
    input_box4 = InputBox(200, 440, 140, 32)
    textsurface = myfont.render('W5: ', False, WHITE)
    screen.blit(textsurface,(100,520))
    input_box5 = InputBox(200, 540, 140, 32)
    textsurface = myfont.render('W6: ', False, WHITE)
    screen.blit(textsurface,(100,620))
    input_box6 = InputBox(200, 640, 140, 32)

    textsurface = myfont.render('Xa1: ', False, WHITE)
    screen.blit(textsurface,(600,120))
    input_box7 = InputBox(800, 140, 140, 32)
    textsurface = myfont.render('Xa2: ', False, WHITE)
    screen.blit(textsurface,(600,220))
    input_box8 = InputBox(800, 240, 140, 32)
    textsurface = myfont.render('True Y: ', False, WHITE)
    screen.blit(textsurface,(600,320))
    input_box9 = InputBox(800, 340, 140, 32)


    input_boxes = [input_box1, input_box2, input_box3, input_box4, input_box5, input_box6, input_box7, input_box8, input_box9] 
    done = False
    while not done or quit:
        for event in pygame.event.get():
            mouse=pygame.mouse.get_pos()   
            click=pygame.mouse.get_pressed()
            if event.type == pygame.QUIT:                
                done = True
            for box in input_boxes:
                box.handle_event(event)

        for box in input_boxes:
            box.update()

        for box in input_boxes:
            box.draw(screen)
        
        if mouse[0]<760 and mouse[0]>610 and mouse[1]<970 and mouse[1]>900:         
            pygame.draw.rect(screen, GREEN,(610,900,150,70))
            if click[0]==1: 
                done = True
                d=2                      
                fill()
        else:
            pygame.draw.rect(screen,BLUE,(610,900,150,70))
        textsurface = myfont.render('GO', False, WHITE)
        screen.blit(textsurface,(620,910))

        pygame.display.flip()
        clock.tick(30)

def pause():
    global quit,d,W,Xa1,Xa2,y_true,xdata,ydata
    paused = True
    x=np.array([Xa1,Xa2],dtype=float)
    white = (255, 255, 255)
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_c:
                    paused=False
                elif event.key==pygame.K_q:
                    pygame.quit()

        screen.fill(white)
        screen.blit(background_image, [0, 0])
        pygame.draw.rect(screen, WHITE,(30,150,225,400),1)
        textsurface=myfont.render('Input Layer', False, GOLD)    
        screen.blit(textsurface, (60, 60))     
        textsurface=myfont.render('Xa1', False, GOLD)
        screen.blit(textsurface, (60, 225))
        textsurface=myfont.render(str(Xa1), False, GOLD)
        screen.blit(textsurface, (150, 225))
        pygame.draw.polygon(screen, GREEN, ((240,240), (240,260), (510,260), (510,280), (540,250), (510, 220), (510,240))) 
        pygame.draw.line(screen, GREEN, (240,260), (540,450), 15)   
        pygame.draw.ellipse(screen, WHITE,(40,200,200,100),2)            
        textsurface=myfont.render('Xa2', False, GOLD)  
        screen.blit(textsurface, (60, 425))
        textsurface=myfont.render(str(Xa2), False, GOLD)  
        screen.blit(textsurface, (150, 425))
        pygame.draw.polygon(screen, GREEN, ((240,440), (240,460), (510,460), (510,480), (540,450), (510, 420), (510,440)))
        pygame.draw.line(screen, GREEN, (240,440), (540,250), 15)  
        pygame.draw.ellipse(screen, WHITE,(40,400,200,100),2)     
        
        add=500    
        out=y_true

        textsurface = myfont.render('W1: '+str(W[0])[0:5], False, WHITE)
        screen.blit(textsurface,(350,180))
        textsurface = myfont.render('W2: '+str(W[1])[0:5], False, WHITE)
        screen.blit(textsurface,(450,390))
        textsurface = myfont.render('W3: '+str(W[2])[0:5], False, WHITE)
        screen.blit(textsurface,(450,280))
        textsurface = myfont.render('W4: '+str(W[3])[0:5], False, WHITE)
        screen.blit(textsurface,(350,480))
        textsurface = myfont.render('W5: '+str(W[4])[0:5], False, WHITE)
        screen.blit(textsurface,(850,210))
        textsurface = myfont.render('W6: '+str(W[5])[0:5], False, WHITE)
        screen.blit(textsurface,(850,450))    
        textsurface = myfont.render('True Output: '+str(out), False, WHITE)
        screen.blit(textsurface,(1240,290))

        pygame.draw.rect(screen, WHITE,(30+add,150,225,400),1)
        textsurface=myfont.render('Hidden Layer', False, GOLD)    
        screen.blit(textsurface, (540, 60))     
        textsurface=myfont.render('Yc', False, GOLD)
        screen.blit(textsurface, (60+add, 225))   
        textsurface=myfont.render(str(NN.predict_H([x],1,0)[0][0])[0:5], False, GOLD)  
        screen.blit(textsurface, (130+add, 225))
        pygame.draw.line(screen, GREEN, (740,250), (1050,330), 15)     
        pygame.draw.ellipse(screen, WHITE,(40+add,200,200,100),2)            
        textsurface=myfont.render('Yd', False, GOLD)  
        screen.blit(textsurface, (60+add, 425))
        textsurface=myfont.render(str(NN.predict_H([x],1,1)[0][0])[0:5], False, GOLD)  
        screen.blit(textsurface, (130+add, 425))
        pygame.draw.ellipse(screen, WHITE,(40+add,400,200,100),2) 
        pygame.draw.line(screen, GREEN, (740,450), (1050,370), 15)   
        pygame.draw.rect(screen, WHITE,(30+(add*2),250,400,225),1)
        textsurface=myfont.render('Output Layer', False, GOLD)    
        screen.blit(textsurface, (1040, 60)) 
        textsurface=myfont.render('Yf', False, GOLD)    
        screen.blit(textsurface, (60+(add*2), 325))
        textsurface=myfont.render(str(NN.predict([x]))[0:5], False, GOLD)  
        screen.blit(textsurface, (120+(add*2), 325))
        pygame.draw.ellipse(screen, WHITE,(40+(add*2),300,200,100),2)    
        pygame.draw.polygon(screen, GREEN, ((1240,340), (1240,360), (1410,360), (1410,380), (1440,350), (1410, 320), (1410,340))) 


        textsurface = myfont.render('Press C to continue or Q to quit.',False,WHITE)
        screen.blit(textsurface,(620,910))
        pygame.display.update()
        clock.tick(5)

def fill():    
    global quit,d,W,Xa1,Xa2,y_true,xdata,ydata
    NN.W[1][0][0]=W[0]
    NN.W[1][0][1]=W[1]
    NN.W[1][1][0]=W[2]
    NN.W[1][1][1]=W[3]
    NN.W[2][0][0]=W[4]
    NN.W[2][1][0]=W[5]
    x=np.array([Xa1,Xa2],dtype=float)
    xdata=[]
    ydata=[]
    for i in range(10):
        xdata.append(i)
        ydata.append(NN.loss_function([x],y_true))
        print(NN.loss_function([x],y_true))
        NN.grad(x,y_true)
        NN.grad_update()
        W[0]=NN.W[1][0][0]
        W[1]=NN.W[1][0][1]
        W[2]=NN.W[1][1][0]
        W[3]=NN.W[1][1][1]
        W[4]=NN.W[2][0][0]
        W[5]=NN.W[2][1][0]

        time.sleep(0.1)
  

    mouse=pygame.mouse.get_pos()   
    click=pygame.mouse.get_pressed()
    screen.fill(BLACK)
    screen.blit(background_image, [0, 0])
    pygame.draw.rect(screen, WHITE,(30,150,225,400),1)
    textsurface=myfont.render('Input Layer', False, GOLD)    
    screen.blit(textsurface, (60, 60))     
    textsurface=myfont.render('Xa1', False, GOLD)
    screen.blit(textsurface, (60, 225))
    textsurface=myfont.render(str(Xa1), False, GOLD)
    screen.blit(textsurface, (150, 225))
    pygame.draw.polygon(screen, GREEN, ((240,240), (240,260), (510,260), (510,280), (540,250), (510, 220), (510,240))) 
    pygame.draw.line(screen, GREEN, (240,260), (540,450), 15)   
    pygame.draw.ellipse(screen, WHITE,(40,200,200,100),2)            
    textsurface=myfont.render('Xa2', False, GOLD)  
    screen.blit(textsurface, (60, 425))
    textsurface=myfont.render(str(Xa2), False, GOLD)  
    screen.blit(textsurface, (150, 425))
    pygame.draw.polygon(screen, GREEN, ((240,440), (240,460), (510,460), (510,480), (540,450), (510, 420), (510,440)))
    pygame.draw.line(screen, GREEN, (240,440), (540,250), 15)  
    pygame.draw.ellipse(screen, WHITE,(40,400,200,100),2)     
    
    add=500    
    out=y_true

    textsurface = myfont.render('W1: '+str(W[0])[0:5], False, WHITE)
    screen.blit(textsurface,(350,180))
    textsurface = myfont.render('W2: '+str(W[1])[0:5], False, WHITE)
    screen.blit(textsurface,(450,390))
    textsurface = myfont.render('W3: '+str(W[2])[0:5], False, WHITE)
    screen.blit(textsurface,(450,280))
    textsurface = myfont.render('W4: '+str(W[3])[0:5], False, WHITE)
    screen.blit(textsurface,(350,480))
    textsurface = myfont.render('W5: '+str(W[4])[0:5], False, WHITE)
    screen.blit(textsurface,(850,210))
    textsurface = myfont.render('W6: '+str(W[5])[0:5], False, WHITE)
    screen.blit(textsurface,(850,450))    
    textsurface = myfont.render('True Output: '+str(out), False, WHITE)
    screen.blit(textsurface,(1240,290))

    pygame.draw.rect(screen, WHITE,(30+add,150,225,400),1)
    textsurface=myfont.render('Hidden Layer', False, GOLD)    
    screen.blit(textsurface, (540, 60))     
    textsurface=myfont.render('Yc', False, GOLD)
    screen.blit(textsurface, (60+add, 225))   
    textsurface=myfont.render(str(NN.predict_H([x],1,0)[0][0])[0:5], False, GOLD)  
    screen.blit(textsurface, (130+add, 225))
    pygame.draw.line(screen, GREEN, (740,250), (1050,330), 15)     
    pygame.draw.ellipse(screen, WHITE,(40+add,200,200,100),2)            
    textsurface=myfont.render('Yd', False, GOLD)  
    screen.blit(textsurface, (60+add, 425))
    textsurface=myfont.render(str(NN.predict_H([x],1,1)[0][0])[0:5], False, GOLD)  
    screen.blit(textsurface, (130+add, 425))
    pygame.draw.ellipse(screen, WHITE,(40+add,400,200,100),2) 
    pygame.draw.line(screen, GREEN, (740,450), (1050,370), 15)   
    pygame.draw.rect(screen, WHITE,(30+(add*2),250,400,225),1)
    textsurface=myfont.render('Output Layer', False, GOLD)    
    screen.blit(textsurface, (1040, 60)) 
    textsurface=myfont.render('Yf', False, GOLD)    
    screen.blit(textsurface, (60+(add*2), 325))
    textsurface=myfont.render(str(NN.predict([x]))[0:5], False, GOLD)  
    screen.blit(textsurface, (120+(add*2), 325))
    pygame.draw.ellipse(screen, WHITE,(40+(add*2),300,200,100),2)    
    pygame.draw.polygon(screen, GREEN, ((1240,340), (1240,360), (1410,360), (1410,380), (1440,350), (1410, 320), (1410,340))) 




def welcome_message():
    global quit, d, W
    mouse=pygame.mouse.get_pos()   
    click=pygame.mouse.get_pressed()
    screen.fill(BLACK)
    screen.blit(background_image, [0, 0])
    textsurface=myfont.render('Perceptron GUI', False, GOLD)
    screen.blit(textsurface, (590, 300)) 
    if mouse[0]<780 and mouse[0]>600 and mouse[1]<570 and mouse[1]>500:         
        pygame.draw.rect(screen, GREEN,(600,500,180,70))
        if click[0]==1:
            d=1
            initial()  
    else:
        pygame.draw.rect(screen,BLUE,(600,500,180,70))
    
    textsurface = myfont.render('START', False, WHITE)
    screen.blit(textsurface,(630,510)) 


welcome_message()

quit = False
d=0
W=[0.0,0.0,0.0,0.0,0.0,0.0]

y_true=1
NN=NeuralNetwork(n_inputs=2,n_outputs=1,hidden_sizes=[2]) 


# MAIN FUNCTION
while not quit:     
    mouse=pygame.mouse.get_pos()   
    click=pygame.mouse.get_pressed()        
    screen.fill((50, 50, 50))     
    screen.blit(background_image, [0, 0])
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit = True  
        elif event.type == pygame.KEYDOWN:
            if event.key==pygame.K_p:
                pause()
               
    if d==0:
        welcome_message()   
    if d==1:
        initial()  
    if d==2:
        fill() 
      

    
    pygame.display.update()
    pygame.display.flip()
pygame.quit()



#plot data

# xdata=[]
# ydata=[]

# Dynamic Plotting of the graphs for LOSS and PREDICTIONS using matplotlib
x=np.array([Xa1,Xa2])
NN=NeuralNetwork(n_inputs=2,n_outputs=1,hidden_sizes=[2])
print(NN.loss_function([x],1))
xdata=[]
ydata1=[]
ydata2=[]
ydata3=[]
ydata4=[]
fig, axes = plt.subplots(2, 1)

line1, = axes[0].plot(xdata, ydata1, 'r-')
line2, = axes[1].plot(xdata, ydata2, 'r-')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)
axes[0].title.set_text("Prediction")
axes[0].set_ylim(0,1)
axes[0].set_xlim(0,100)
axes[1].title.set_text("Loss Graphs")
axes[1].set_ylim(0,0.1)
axes[1].set_xlim(0,100)

for i in range(100):
  NN.grad(x,y=y_true)
  NN.grad_update()
  xdata.append(i)
  ydata1.append(NN.predict([x]))
  ydata2.append(NN.loss_function([x],y_true))
  ydata3.append(NN.W[2][0])
  ydata3.append(NN.W[2][1])
  line1.set_xdata(xdata)
  line1.set_ydata(ydata1)
  line2.set_xdata(xdata)
  line2.set_ydata(ydata2)

  plt.draw()
  plt.pause(1e-17)
  time.sleep(0.1)
plt.plot()
plt.show()
