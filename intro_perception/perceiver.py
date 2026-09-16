"""Compatibility imports for the two intro perceivers."""

from intro_perception.classical import ClassicalGatePerceiver
from intro_perception.yolo import YoloGatePerceiver

IntroGatePerceiver = ClassicalGatePerceiver

__all__ = ["ClassicalGatePerceiver", "YoloGatePerceiver", "IntroGatePerceiver"]
