# RillAi
[RillAi](https://www.twitch.tv/theredhotbr/clip/InnocentFineAardvarkTheThing) is an Artificial Inteligence project that teaches a computer to play the first level of Crash Bash by using Reinforcement Learning.

# Getting Started

Make sure that you have a legitimate copy of the NTSC-U version of Crash Bash before proceeding.

RillAi uses the PSX emulator BizHawk to gather data from its RAM memory in order to read the current state of the game, make a prediction and play. Download the latest version of BizHawk [here](https://github.com/TASVideos/BizHawk/releases) and setup lua socket using [this](https://stackoverflow.com/questions/33428382/add-luasocket-to-program-bizhawk-shipped-with-own-lua-environment) tutorial by Zeromus.

You'll need [Python](https://www.python.org) to run the Neural Network script, with the dependencies `tensorflow`, `numpy` and `matplotlib`. You can install those by using `pip install tensorflow`, `pip install numpy` and `pip install matplotlib`. BizHawk already comes with a lua interpreter, you only need to setup lua socket to run with it.

Clone this repository and move the files inside the `src/` folder into the root folder of your BizHawk.

# Running the A.I

## Training

* Open `Crash Bash (NTSC-U)` in your `BizHawk` emulator and make sure that you have `multitap` enabled with 4 gamepads in port 1 at `PSX -> Controller/Memcard Configuration`.

* On your game, go to `Battle Mode -> 4 Players -> Crash Ball` and pause your emulator. Creating a save state here is highly recommended, since you can use it to come back and restart the process at any time.

* From your command prompt run `python neural_network.py`. This will initialize the local server that will receive data from the emulator, train the network, make a prediction and return the inputs that each A.I should play.

* Inside BizHawk select `Tools -> Lua Console -> Script -> Open Script...` and open `game_data.lua`. If no error appeared in the lua console, congratulations, unpause the emulator and your A.I will start the learning process.

* The A.I is programmed to stop running once it reaches 10M steps played, each step consisting of 3 frames. However, the model is saved after each round is done, so you can stop the algorithm at anytime you want if you're already happy with the progress of the A.I.

# Algorithm

RillAi uses the Deep Q-Learning algorithm, working with a fully connected neural network. The input to the NN is the position of the character for the past 3 frames, the position of each ball for the past 3 frames, and whether the character has the ability to use the kick. Crashball has a hardcoded maximum of 5 balls per stage, but it takes a while for the game to reach that point. Consequently, to avoid unbalanced training of the neurons, each memory shuffles the balls position in the input array, so that it helps the A.I to avoid a bias towards the most active neurons. The output layer has size of 4, with a linear activation, and it represents which input the A.I should press (square, R1 + left, R1 + right or nothing). The network rewards `1` for every frame that the A.I defends a ball, penalizes `-1` for every frame that the A.I lets a ball in their goal, and `0` if otherwise.

# Gathering Data

I'm working on a [RAM map](https://docs.google.com/spreadsheets/d/1EB_qeiwz316mTHtjCwMXC-LzmyVOInbdGkYAV1kL3LA/edit?usp=sharing) of Crash Bash in order to understand how the minigame works and collect data from it.

The challenge here was that the game doesn't store in RAM any information when the ball colides with a player, so I had to implement a hack that gets that information when the game is calculating the colision and saves in a blank space of the RAM. The implementation of the hack is in the `game_data.lua`, which injects the assembly when you first load the script. I only modified temporary registers that aren't used in the respective functions, and I only added instructions in NOP instructions, to make sure that I wouldn't change a single aspect of the game.

# Results

RillAi managed to beat the Platinum Relic bots 100 times in a row, and get a few perfect rounds. RillAi won every round when facing one of the best Crash Bash players in the world for one hour.

This repository contains a pre-trained model with over 100h of training.