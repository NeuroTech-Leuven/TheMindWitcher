# Game modification
This document explains the decision process behind our choices regarding modifying the game 'The Witcher 3'. This part of the project assumes a classification decision about imaginary movement and emotions to be readily available as inputs.

## Choice of game
To link a neuro-interface with playing a game, there are several things to keep in mind:
1. Time sensitivity: A game with time sensitive inputs, like Mario Kart, does not work for this project. We need data from at least a few continuous seconds to be able to make an informed decision.
2. Error impact: When playing the game, a wrong decision should not impact your enjoyability of the game too much.
3. Modding availability: We need to be able to control the game without learning a new coding language or modifying source files.

A first explored option is a simple game that can be fully controlled with only the two different movement inputs: left and right. Then you would be playing exclusively with your mind and anyone, paralyzed people too, could play the game. However, simple games which could be played with only two controls almost always transgress against one of the above restrictions. For instance, Pong only requires two controls but is time sensitive and making a wrong move in checkers may ruin your chances of winning the game. 

The next option is to have a regular game wherein two actions are controlled by the imaginary movement inputs and otherwise you're playing regularly with controller or keyboard and mouse. The actions can be chosen not to be too time sensitive, and even if they were you can always use your regular controller to execute that action. Regarding error impact we shifted to an open world game, where actions can easily be reversed or ignored. Again, the right choice of actions helps here as well. In the end, 'The Witcher 3' was chosen for the following reasons:
- It has an in-built debug console, which means the game environment can be easily modified. 
- There are several magical actions performed by the main character, linking those to a brain command amplifies the magical feeling.
- It is a well-known, highly awarded game with lots of online support, available mods, etc.

## Keyboard presses
To execute in-game actions, virtual keyboard presses are used. This is done with Python using the [keyboard package](https://pypi.org/project/keyboard/). There are two points of attention:
1. The game tab must be in focus as to receive the keyboard presses.
2. A key must be held down long enough relative to the frame rate. With a frame rate of 50fps, the key needs to be pressed at least 1/50 = 0.02 seconds. We assumed a minimum frame rate of 30fps, widely considered to be the minimum for an enjoyable gaming experience. If your frame rate is lower, the actions will not be executed. 

## From classification to execution

In the following table are the possible data processing inputs, the related action in game and how to execute them on a QWERTY keyboard.
| Data processing decision | In-game action  | Control |
|--------------------------|-----------------|---------|
| IM: left                 | Cast a spell    | Q       |
| IM: right                | Call your horse | X       |
| EM: neutral              | Some clouds     | weatherCommand(Mid_Clouds)        |
| EM: happy                | Shining sun     | weatherCommand(Clear)         |
| EM: sad                  | Rain            | weatherCommand(Storm)         |
| EM: fear                 | Snow            | weatherCommand(Snow)        |

The weatherCommand function is a placeholder for several successive key presses. The flow of execution is shown below for when a neutral emotion decision is made.

![](WeatherCommand.svg)


Note: the keys to use may differ from one keyboard layout to another. As Neurotech Leuven is located in Belgium, we support AZERTY keyboards next to the internationally accepted QWERTY.