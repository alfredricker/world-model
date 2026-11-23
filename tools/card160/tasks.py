"""Card 160: the tasks, and each task's goal as MiniGrid states it (for the rules agent, which makes the example success steps)."""
import rules
from rules import DOOR, KEY, BALL, BOX, GOAL, EMPTY, PICK, TOGGLE

DEV = {"DoorKey-5x5": "MiniGrid-DoorKey-5x5-v0", "DoorKey-6x6": "MiniGrid-DoorKey-6x6-v0", "Unlock": "MiniGrid-Unlock-v0"}
TEST = {"DoorKey-8x8": "MiniGrid-DoorKey-8x8-v0", "DoorKey-16x16": "MiniGrid-DoorKey-16x16-v0", "UnlockPickup": "MiniGrid-UnlockPickup-v0",
        "BlockedUnlockPickup": "MiniGrid-BlockedUnlockPickup-v0", "KeyCorridorS3R3": "MiniGrid-KeyCorridorS3R3-v0"}
ALL = dict(DEV, **TEST)


def told_goal(task, codes):
    e = codes.of((EMPTY, 0, 0))
    if task.startswith("DoorKey"):
        return [("arrive", codes.of((GOAL, 1, 0)))]
    if task == "Unlock":
        return [("effect", codes.of((DOOR, c, 2)), codes.of((DOOR, c, 0)), TOGGLE) for c in range(6)]
    kind = BALL if task.startswith("KeyCorridor") else BOX
    return [("effect", codes.of((kind, c, 0)), e, PICK) for c in range(6)]
