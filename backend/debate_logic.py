from enum import Enum, auto

class DebatePhase(Enum):
    PRO_CONSTRUCTIVE = auto()
    CON_CONSTRUCTIVE = auto()
    CROSSFIRE_1_QA = auto()
    PRO_REBUTTAL = auto()
    CON_REBUTTAL = auto()
    CROSSFIRE_2_QA = auto()
    PRO_SUMMARY = auto()
    CON_SUMMARY = auto()
    FINAL_FOCUS_PRO = auto()
    FINAL_FOCUS_CON = auto()
    JUDGING_PHASE = auto()
    COMPLETED = auto()

class DebateState:
    def __init__(self, resolution: str, user_role: str, ai_difficulty: str, mode: str = "user-vs-ai"):
        self.resolution = resolution
        self.user_role = user_role.lower()
        self.ai_role = "con" if self.user_role == "pro" else "pro"
        self.ai_difficulty = ai_difficulty.lower()
        self.mode = mode  # "user-vs-ai" or "ai-vs-ai"
        self.turn_number = 1
        self.transcript = []

        # Role-based phase order
        self.phase_order = [
            DebatePhase.PRO_CONSTRUCTIVE,
            DebatePhase.CON_CONSTRUCTIVE,
            DebatePhase.CROSSFIRE_1_QA,
            DebatePhase.PRO_REBUTTAL,
            DebatePhase.CON_REBUTTAL,
            DebatePhase.CROSSFIRE_2_QA,
            DebatePhase.PRO_SUMMARY,
            DebatePhase.CON_SUMMARY,
            DebatePhase.FINAL_FOCUS_PRO,
            DebatePhase.FINAL_FOCUS_CON,
            DebatePhase.JUDGING_PHASE,
            DebatePhase.COMPLETED
        ] if self.user_role == "pro" else [
            DebatePhase.CON_CONSTRUCTIVE,
            DebatePhase.PRO_CONSTRUCTIVE,
            DebatePhase.CROSSFIRE_1_QA,
            DebatePhase.CON_REBUTTAL,
            DebatePhase.PRO_REBUTTAL,
            DebatePhase.CROSSFIRE_2_QA,
            DebatePhase.CON_SUMMARY,
            DebatePhase.PRO_SUMMARY,
            DebatePhase.FINAL_FOCUS_CON,
            DebatePhase.FINAL_FOCUS_PRO,
            DebatePhase.JUDGING_PHASE,
            DebatePhase.COMPLETED
        ]

        self.current_phase = self.phase_order[0]
        self.current_speaker = "user" if self.user_role == self.get_role_from_phase(self.current_phase) else "ai"

    def advance_phase(self):
        """Advance to the next major debate phase."""
        index = self.phase_order.index(self.current_phase)
        if index + 1 < len(self.phase_order):
            self.current_phase = self.phase_order[index + 1]
            self.turn_number = 1
            self._set_speaker_by_phase()
        else:
            self.current_phase = DebatePhase.COMPLETED

    def _set_speaker_by_phase(self):
        """Set the current speaker based on debate phase."""
        phase = self.current_phase

        if phase in [DebatePhase.PRO_CONSTRUCTIVE, DebatePhase.PRO_REBUTTAL, DebatePhase.PRO_SUMMARY, DebatePhase.FINAL_FOCUS_PRO]:
            self.current_speaker = "user" if self.user_role == "pro" else "ai"
        elif phase in [DebatePhase.CON_CONSTRUCTIVE, DebatePhase.CON_REBUTTAL, DebatePhase.CON_SUMMARY, DebatePhase.FINAL_FOCUS_CON]:
            self.current_speaker = "user" if self.user_role == "con" else "ai"
        elif self.is_crossfire_phase():
            # alternate after each user/AI message
            self.current_speaker = "user" if self.current_speaker == "ai" else "ai"
        elif phase == DebatePhase.JUDGING_PHASE:
            self.current_speaker = "ai"

    def get_role_from_phase(self, phase):
        if "PRO" in phase.name:
            return "pro"
        elif "CON" in phase.name:
            return "con"
        return None

    def is_crossfire_phase(self):
        return "CROSSFIRE" in self.current_phase.name

    def log_speech(self, speaker: str, role: str, speech_type: str, content: str):
        self.transcript.append({
            "speaker": speaker,
            "role": role.capitalize(),
            "speech_type": speech_type,
            "content": content
        })

    def get_expected_speech_type(self):
        phase = self.current_phase.name.lower()

        if "crossfire" in phase:
            return "Crossfire"
        elif "rebuttal" in phase:
            return "Rebuttal"
        elif "summary" in phase:
            return "Summary"
        elif "final_focus" in phase:
            return "Final Focus"
        elif "constructive" in phase:
            return "Constructive"
        else:
            return "Speech"

    def is_ai_turn(self) -> bool:
        return self.current_speaker == "ai"

    def get_state(self):
        return {
            "phase": self.current_phase.name,
            "turn": self.turn_number,
            "speaker": self.current_speaker,
            "user_role": self.user_role,
            "ai_role": self.ai_role,
            "difficulty": self.ai_difficulty,
            "transcript": self.transcript,
        }

    def get_transcript(self):
        return self.transcript
