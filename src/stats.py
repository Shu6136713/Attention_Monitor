"""
Statistics Manager Module
Tracks attention metrics and calculates Gaze Aversion Rate (GA Rate).
"""

from enum import Enum
from src.gaze_logic import GazeState


class AttentionLevel(Enum):
    """Classification of attention level based on GA Rate."""
    FOCUSED = "Focused"      # GA Rate < 10%
    NORMAL = "Normal"        # GA Rate 10-20%
    DISTRACTED = "Distracted"   # GA Rate > 20%


class StatsManager:
    """
    Tracks attention statistics and calculates Gaze Aversion Rate.
    
    GA Rate (Gaze Aversion Rate) measures the percentage of time
    the user's gaze is off-screen relative to total task time.
    
    Formula:
        GA Rate = (Total OFF_SCREEN frames / Total frames) × 100%
    
    Classification thresholds:
        - < 10%: Focused - User is highly focused on task
        - 10-20%: Normal - User occasionally looks away for thinking
        - > 20%: Distracted - High cognitive load or discomfort
    """
    
    def __init__(self):
        """Initialize statistics tracking."""
        self.total_frames = 0          # Total number of frames processed
        self.off_screen_frames = 0     # Number of frames with gaze OFF_SCREEN
        self.on_screen_frames = 0      # Number of frames with gaze ON_SCREEN
        self.unknown_frames = 0        # Number of frames with UNKNOWN state
    
    def update(self, gaze_state: GazeState) -> None:
        """
        Update statistics with current frame's gaze state.
        
        :param gaze_state: Current gaze state (ON_SCREEN, OFF_SCREEN, or UNKNOWN).
        """
        self.total_frames += 1
        
        if gaze_state == GazeState.OFF_SCREEN:
            self.off_screen_frames += 1
        elif gaze_state == GazeState.ON_SCREEN:
            self.on_screen_frames += 1
        elif gaze_state == GazeState.UNKNOWN:
            self.unknown_frames += 1
    
    def get_ga_rate(self) -> float:
        """
        Calculate Gaze Aversion Rate (GA Rate).
        
        :return: GA Rate as percentage (0-100).
        """
        if self.total_frames == 0:
            return 0.0
        
        return (self.off_screen_frames / self.total_frames) * 100.0
    
    def classify_attention(self) -> AttentionLevel:
        """
        Classify attention level based on GA Rate thresholds.
        
        :return: AttentionLevel enum (FOCUSED, NORMAL, or DISTRACTED).
        """
        ga_rate = self.get_ga_rate()
        
        if ga_rate < 10.0:
            return AttentionLevel.FOCUSED
        elif ga_rate <= 20.0:
            return AttentionLevel.NORMAL
        else:
            return AttentionLevel.DISTRACTED
    
    def get_summary(self) -> dict:
        """
        Get a complete summary of attention statistics.
        
        :return: Dictionary containing:
            - total_frames: Total frames processed
            - off_screen_frames: Frames with gaze off-screen
            - on_screen_frames: Frames with gaze on-screen
            - unknown_frames: Frames with unknown state
            - ga_rate: Gaze Aversion Rate (percentage)
            - attention_level: Classification (AttentionLevel enum)
            - attention_label: Hebrew label string
        """
        ga_rate = self.get_ga_rate()
        attention_level = self.classify_attention()
        
        return {
            "total_frames": self.total_frames,
            "off_screen_frames": self.off_screen_frames,
            "on_screen_frames": self.on_screen_frames,
            "unknown_frames": self.unknown_frames,
            "ga_rate": ga_rate,
            "attention_level": attention_level,
            "attention_label": attention_level.value
        }
    
    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.total_frames = 0
        self.off_screen_frames = 0
        self.on_screen_frames = 0
        self.unknown_frames = 0




