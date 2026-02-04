# ABOUTME: Core types and enums for the garbling economics game
# ABOUTME: Defines asset qualities, signals, and actions

from enum import Enum


class AssetQuality(Enum):
    """True underlying state of the asset"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class Signal(Enum):
    """Observable signal (potentially garbled)"""
    BAD = 0
    NEUTRAL = 1
    GOOD = 2


class Action(Enum):
    """Receiver's action choice"""
    PASS = 0
    BUY = 1
