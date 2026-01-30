from dataclasses import dataclass

@dataclass
class EmpatheticDialogueParams:
    num_labels: int
    max_time: int
    max_speakers: int

@dataclass
class MELDParams:
    num_labels: int
    max_time: int
    max_speakers: int