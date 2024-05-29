import keyboard as kb
import time
import win32api
import pyperclip as pc

# All code required to manipulate the game The Witcher 3. 
# NOTE: Make sure you don't run your game as administrator, otherwise the script won't be able to send keypresses to the game.

##
# Set command keys for manipulating the game The Witcher 3. It checks whether your keyboard layout 
# is AZERTY or QWERTY. Currently only those two supported.
#
# @return   COMMAND_KEY     Key to call console module in The Witcher 3
# @return   SPELL_KEY       Key to execute the cast sign action
# @return   HORSE_KEY       Key to call your horse in The Witcher 3
#
# To add further keyboard leyout configurations, use the following source:
# https://docs.google.com/spreadsheets/d/1aKasLNLaXtWuPFENOP72f_WaWC2SreXvLhsWmb9OuJM/edit#gid=1112433978
def setCommandKeys():
    current_language_code = win32api.GetKeyboardLayoutName()
    azerty_languages = ["0001080c", "00000813", "0000080c", "0000040c"]
    if current_language_code in azerty_languages:
        COMMAND_KEY = "ù"
        SPELL_KEY = "a"
    else:
        COMMAND_KEY = "~" 
        SPELL_KEY = "q"

    # The key to call your horse is always X
    HORSE_KEY = "X"

    return COMMAND_KEY, SPELL_KEY, HORSE_KEY

# Set command keys.
COMMAND_KEY, SPELL_KEY, HORSE_KEY = setCommandKeys()

##
# Code to press the given key. 
# 
# @input    key     The key to press
# @input    delay   Amount of time to hold the key down. This must be longer than the game frame period, which is 1/fps.
#                   We assume a minimum framerate of 30fps. 
def pressKey(key, delay=0.035):
    kb.press(key)
    time.sleep(delay)
    kb.release(key)

##
# Activate the debug console of The Witcher 3 and execute the given command.
#
# @input    command     The debug console command to execute.
# @input    delay       Amount of time to hold down the necessary keys. This must be longer than the 
#                       game frame period, which is 1/fps. We assume a minimum framerate of 30fps. 
# @input    WTcommand   Boolean that indicates whether the given command is a "WTcommand", which means
#                       the full command to execute looks like "changeweather(WT_{command})""
#
# More commands can be found at: https://commands.gg/witcher3/weather
def typeCommand(command, delay=0.035, WTcommand=True):

    # Call debug console
    pressKey(COMMAND_KEY, delay)

    if isinstance(command, list):
        for i,c in enumerate(command):
            fullCommand = c if not WTcommand else f"changeweather(WT_{c})"
            # Copy the command to clipboard
            pc.copy(fullCommand)

            # Use CTRL+V to copy the command into the debug console.
            kb.press("ctrl")
            time.sleep(delay)
            pressKey("v", delay)
            time.sleep(delay)
            kb.release("ctrl")

            if i < len(command)-1:
                kb.press("enter")
                time.sleep(delay)
                kb.press("enter")
    else:
        fullCommand = command if not WTcommand else f"changeweather(WT_{command})"
        # Copy the command to clipboard
        pc.copy(fullCommand)
        # Use CTRL+V to copy the command into the debug console.
        kb.press("ctrl")
        time.sleep(delay)
        pressKey("v", delay)
        time.sleep(delay)
        kb.release("ctrl")


    # Execute command and close debug console
    pressKey("enter", delay)
    pressKey(COMMAND_KEY, delay)