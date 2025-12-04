# DOFUS 3 CLICKBOT WITH PYTHON
#### Video demo: <URL HERE>
#### Description:

## Context

### The game

Dofus (3) is a French MMORPG that sets the player on a quest to collect all 6 primordial dragon eggs, the "Dofus". It turned into a nation wide mania among kids and teenagers as soon as it came out in 2006. It is based on a turn by turn combat system, and as most MMORPGs, puts the emphasys on farming, i.e killing mobs and collecting resources to gain experience and, above all, money. It has a long history of battle between developpeurs and bots, programs that execute various game actions automatically to gain resources.

I build a click-bot using python 3 that automatically collects resources on any path defined by the user. A click bot is a basic type of bot that orients itself with visual cues and takes action by taking control of the mouse and keyboard. Since I'm working on a mac with the Retina technology that interacts in complicated ways with the tools used for image screen capture, the bot probably only works for macs with similar screen resolution (2560x1600 Retina screen).

The world of Dofus in made up of maps, a discretization of the game's world. A player can go from one map to its neighbouring maps by going south, north, east or west. 
Each of those maps may contain resources such as wood or ore, that have a general cooldown after a player collects them.
After collecting a resource spot, the player has a samll chance of being attacked by the resource's protector, a single monster that is fairly easy to deal with. If the player wins that fight, he gets rewarded whith a bag of said resource.

### The Bot

So the bot needs to:
-Know which map he is on in order to orient itself (maps are identified by a tuple of coordinates [x, y]).
-Find a path from one map to another (there may be obstacles that prevent the player from going certain directions).
-Recognize resources and know wether they are available.
-Monitor wether the inventory of the character is full and if it is, go to the bank to empty it.
-Know when it has been attacked by a monster and handle the fight. 

### Quick word on screen capture

There are two tools used in this project to capture all or part of the screen.
Using the CoreGraphics module of the Quartz (Apple's drawing engine) library yields screenshots with the true resolution, but is slower. We use them for precise applications on small parts of the screen (text recognition for example). 
Using the mss library is faster but the library is tricked by Retina into using a 900x1400 resolution instead of 1600x2560, so the quality of the image suffers. We use this whenever we can get away with it to gain time.

### Quick word on template matching

Template matching came in very handy in this project. It works by simply running convolution on an image, using a template (the thing we want to recognize in the image) as the kernel of the convolution. It works only when the thing to match is *very* similar to the template, but when it works it's a very easy and efficient solution.

## Path finding

### Map recognition

This part is easy enough. The coordinates of the map appear on the top left hand corner in white text over the game's background. I capture the corresponding area using CoreGraphics, then do some image processing to keep only the text. This is achieved by converting the image to HSV. The text corresponds to pixels with a Hue of 0 (grey pixels). We convert grey pixels to black and non-grey pixels to white, so that the text appears black on white. 
We then run the image through Google's open source Optical Character Recognition (Tesseract) using the pytesseract library to retrieve the coordinates after regular expressing matching.

### Walls 

First to find a valid path through the game's world, we need to know where we can't go. Dofus' map system is like a grid where the player can go from one grid to any of its neighbours. Any ? Well not exactly, some transition are impossible, because of city walls, lakes, dense forests, etc... 
We store all those "walls" as tuples of map coordinates in a text file, separated by semicolons. So if ((x1, y1), (x2, y2)) appears in the file, then the player cannot join (x1, y1) form (x2, y2) and vice-versa. 
To create the file containing all the walls for a given zone, I made a utilitary app using tkinter that draws a grid of all possible coordinates. I can click on any ridge to toggle it then save those ridges as the walls of the zone.
This is only intended for the development and not made available in the final app.

### Path finding algorithm

I used the classic A* algorithm described in the ai50 course, using the Manhattan distance and the wall system described above to determine accessible neighbour nodes (maps in our case).
I then execute parts of the path one by one and check before moving on so that the bot is more reliable.

## Resource identification

### Quick word on image processing

Resources in the game include types of trees, ores, herbs, etc... A given type of resource (iron for example) only has one template. So I first thought of identifying the resources with template matching, maybe in combination with specific image processing. But for many resources, trees especially, this proved impossible. On some maps, only small parts of the resource are visible, and shading provides variations to the base template, so that template matching becomes unreliable. 
Training Convolutional Neurol Netework would have been chellenging for the same reasons coupled to the lack of samples to train it on. 
As I would have needed to label the training data myself anyway, I thought I might as well record the resources poisitions by hand.

### Database solution

I built another helper app with tkinter. This one allows me to make buttons corresponding to any resources. After activating those buttons, I can add the positions of the resources to a databse by simply clicking where they are in-game. The app automatically retrieves the map position using the map recognition method described previously.

### Availibility checking

Once I know where resources are supposed to be, it becomes easier to check for availability. If I hover the mouse on the resource, then a pop up appears indicating the availability in a short sentence. This pop up will always be the same, so template matching works well in that case. I hover over the resource, take a screenshot around the mouse using mms to go faster, and check for the template saying "Couper" or "Ceuiller" or "Recolter" depending on the resource (the game is in French). Those templates can be found at /resources/templates/resources.

### Banking

The main bot loop includes regular checking of the inventory. An icon indicates wether the inventory is full, almost full or not full. Whenever the inventory is detected to be full using template matching, the bot goes to the bank and empties its inventory with a series of clicks. This part of the bot doesn't have feedback, but I found that using appropriate wait times between clicks, the process was reliable.

### Name tag monitoring and screenshot

When a resource is clicked, the character moves towards it, then starts gathering. If we click on the next resource before it reaches the first one, it will cancel the ongoing action. 
To know wether the bot is done travelling to the resource and if it's safe for the programme to keep going, we monitor the character's movements using its nametag. It would be very cumbersome to use template matching directly on the sprite of the character, but fortunately by pressing 'p' the player can display a nametag directly above the character's head, in a bright yellow on a black background.
We take a screenshot of that name and can then track the position of the character using template matching. 
The app includes a "Capture Nametag" button to help the user create the template for his character. The user simplly draws a rectangular region surrounding the nametage, provides the name of his character under which the nametag will be registered, and the app does the rest, cutting out the nametag by detecting the bright yellow color of the text

## Combat

This part got technical on the image-processing front. I tried a lot of different things, mostly to practice the methods I learned in ai50. I will just detail the thing that worked and quickly mention the many things that didn't.

### Combat status checking

This part was easy enough, I mainly repurposed the map recognition code. 
When a fight begins, a big green button appears at the bottom right of the screen. In the preparation phase, this button says "PRET" (ready). During the character's turn, the button says "FIN DE TOUR" (end your turn). During the monster's turn, the button still says "FIN DE TOUR", but it's grayed out. 
Anyway by running optical character recognition on that button, I can know:
-Wether a fight is going on (This is checked regularly during the bot's operation)
-What is the phase of combat is, therefore what action the character should take.

### Entity recognition

Dofus fight system is turn by turn, and takes place on a checkboard. The entities move on a grid with a range determined by their Movement Points (MP). So to fight the monster, I need to know where it is, and where I am. 
Under any allied entity appears a blue square border. Under ennemies it is red. By tweeking the parts of this border I keep as a template, I managed to reliably detect allied and ennemy entities using template matching. For our use can, we only need to detect one ally and one ennemy, as the fight against resource protectors is one-on-one.

### Movement tile recognition

The checkboard the fight takes place on has a different shape depending on the map. It can also include obstacles. To automate the fight would have required to run a CNN (Convolutional Neural Network) to recognize the tiles of the checkboard which would have required a lot of computation. My computer can barely manage running the game already. 
Thankfully the game highlights in green all the tiles my character can move to. This would make my job way easier (so I thought).
-I tried selecting pixels based on their HSV values to only keep the green of the tiles. But because of shade variations and a bit of the original tile shining through, this was not specific enough and I would always get some other background elements (leaves and whatnot).
-I tried automating the process with a logistic regression but it failed for the same reasons.
-I tried running a Multi Layer Perceptron on the RGB values but it failed for the same reasons.
-I tried K neighbours classification but it failed for the same reasons. 
-I decided to go full compute and train a CNN on the screenshot to identify the movement tiles. In that case, matching an entire image to an image made of restricted categories is called semantic segmentation (which is to say, "those pixels belong to that category"). I used the first method + some elbow grease to prepare the training data, a dozen images was enough. This worked very well, but even after some optimization, running the network on an image of the lowest quality possible took upwards of two seconds.
I wasn't satisfied, so I thought of a similar but quicker way. Knowing the MP of the character, I can compute where it can move *in theory*. By taking small 20x20 screenshots of those theoretical postions, I can determine wheter I can go there *in practice* by running them through a feather weight CNN classification algorithm. Remember, the tiles I can indeed go to are highlighted in green.

So I prepared the training data:
I took small screenshots of the tiles surrouding my character for 16 maps, for a total of 640 20x20 screenshots. I classified those screenshots (tile or non-tile), and trained a CCN network on the classification task. I got away with using only one convolutional layer made of 16 different kernel, which was drastically faster than the previous iteration. The bottle neck became the template matching, and after some small optimizations I was happy. 

### Putting it all together

So after detecting the combat the bot can:
-Identify itself and the monster. 
-Get as close to the monster as possible (this part is pretty cool: the checkboard is tilted and comptressed along the y dimension for a 3D illusion, so the distance used can be proved to be the Manhattan distance for the non-orthogonal system of the checkboard.)
-Attack if in range of the monster (once at a time, checking for the end of fight pop up in between)
-Pass the turn

## Putting it all together 2

The user is presented with an app where he can chose:
-The resources to collect
-The path to follow
-The zone concerned (the game has a zone system that makes some map coordinates degenerate)
-The name of the bot character. The resources/templates/names folder should include a screenshot of the name tag of the character. This is not absolutely necessary but it makes the bot more efficient. 

The user can open a sub app to draw out a path and another sub app to capture the name tag of the character he wants to use as a bot.

The bot then goes through the path in a loop, collecting resources, handling fights, and going to the bank when it needs to. 

BEWARE: The bot relies on UI elements that are movable. Those elemets (inventory, spell bar with a single line of spells, fight button) all need to be in their default position for the bot to function properly. 





