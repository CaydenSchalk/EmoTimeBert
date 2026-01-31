from dataclasses import dataclass

@dataclass
class EmpatheticDialogueParams:
    num_labels: int
    max_time: int = 16
    max_speakers: int = 2

@dataclass
class MELDParams:
    num_labels: int
    max_time: int = 24
    max_speakers: int = 9