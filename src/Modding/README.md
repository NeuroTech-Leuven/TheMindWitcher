# Game modification
In our project the MindWitcher, we chose to work with the award winning game The Witcher 3 developed by CD Project Red, where you play as the famous Witcher Geralt of Rivia.
This document explains the decision process behind our choices regarding modifying the game 'The Witcher 3'. The game receives both inputs from the imaginary movement and emotions pipeline to then perform an action in the game, e.g., changing the game state, performing a movement, etc. 

## Three points of consideration
To link a neuro-interface with playing a game, there are several things to keep in mind:
1. __Time sensitivity__: a Brain Computer Interface requires several seconds of data to be able to make an informed decision. This fact excludes time sensitive games, such as Mario Kart.
2. __Error impact__: neurotech applications are never 100% accurate, therefore games where every move has a high impact are excluded.
3. __Modding availability__: we need to be able to control the game without learning a new coding language or modifying source files.

Out of all possible games there was one extra soft requirement we looked after. We sought to control a magical action, as linking those to a brain command amplifies the magical feeling. 

In the end, we chose 'The Witcher 3', which satisfies the three points of consideration.
1. __Time sensitivity__: There are a lot of controllable actions, so we can choose an action that is not (too) time sensitive. Even so, the player can use their regular controller to perform the action if needed.
2. __Error impact__: It is an open world game, where wrongly executed actions can easily be reversed or ignored.
3. __Modding availability__: Unlike many other games, it has an in-built debug console, which means the game environment can be easily modified.

The playable character Geralt can execute 'signs' as magical action and so the game also suits the extra requirement. Moreover, 'The Witcher 3' is a well-known, highly awarded game with lots of online support if anyone would need it.

## Virtual keyboard to control the game
To execute in-game actions, virtual keyboard presses are used. This is done with Python using the [keyboard package](https://pypi.org/project/keyboard/). It features hotkey support (e.g. Ctrl+V) and "maps keys as they actually are in your layout, with full internationalization support". This is important to support different keyboard layouts. As Neurotech Leuven is located in Belgium, we support AZERTY keyboards next to the internationally accepted QWERTY. An alternative to this Python package is [pynput](https://pypi.org/project/pynput/), but they have no clear features mentioned so the keyboard package is favored.

There are two points of attention:
1. __Focus__:The game tab must be in focus as to receive the keyboard presses.
2. __Timing__: A key must be held down long enough relative to the frame rate. 
    - For example, with a frame rate of 50fps, the key needs to be pressed at least 1/50 = 0.02 seconds. 
    - In our case we chose a minimum frame rate of 30fps, as it is widely considered to be the minimum for an enjoyable gaming experience.

## From classification to execution
Each classification in both pipelines is associated with an action in the game. For example, if the imagined movement pipeline classifies a left movement then Geralt will cast a spell. The below table gives a complete overview of the classifications and their in-game action, assuming a QWERTY keyboard.
| Data processing pipeline | Decision | In-game action  | Control |
|--------------------------|----------|-----------------|---------|
| Imaginary Movement       | left     | Cast a spell    | Q       |
| Imaginary Movement       | right    | Call your horse | X       |
| Emotions                 | neutral  | Change weather to some clouds     | weatherCommand(Mid_Clouds)        |
| Emotions                 | happy    | Change weather to shining sun     | weatherCommand(Clear)         |
| Emotions                 | sad      | Change weather to rain            | weatherCommand(Storm)         |
| Emotions                 | fear     | Change weather to snow            | weatherCommand(Snow)        |

Actions, such as casting a spell and calling your horse, are associated with key presses in the game, Q & X respectively. To change the weather a bit more work is required. Here, a command must be entered in the debug console. Rather than typing out everything one-by-one, the command is copied in the clipboard and later pasted in the debug console. The flow of execution shown below gives an idea how this happens for when a neutral emotion decision is made. 

![](../../docs/WeatherCommand.svg)

## Conclusion

The Witcher 3 is a game well suited to this project. Using virtual keyboard presses we can execute any command and with the in-built debug console of the game, even the game environment may easily be adapted.
