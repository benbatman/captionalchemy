import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    duration: Optional[float] = None
    is_punctuation: bool = False
    is_sentence_ending: bool = False
    is_clause_ending: bool = False
    is_subword: bool = False

    def __post_init__(self):
        if self.duration is None:
            self.duration = self.end - self.start
        if self.start < 0 or self.end < 0:
            raise ValueError("Start and end times must be non-negative.")
        if self.end < self.start:
            raise ValueError("End time must be greater than start time.")
        if not self.word:
            raise ValueError("Word cannot be empty.")


@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    word_count: int = 0
    char_count: int = 0
    reading_speed_cps: float = 0.0
    break_reason: str = ""
    duration: Optional[float] = None

    def __post_init__(self):
        if self.duration is None:
            self.duration = self.end - self.start
        if not self.word_count:
            self.word_count = len(self.text.split())
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.reading_speed_cps and self.duration > 0:
            self.reading_speed_cps = self.char_count / self.duration


class BreakPriority(Enum):
    HARD_LIMIT = 1
    SENTENCE_END = 2
    CLAUSE_END = 3
    NATURAL_PAUSE = 4
    READING_SPEED = 5
    LINE_LENGTH = 6


class TimingAnalyzer:
    def __init__(
        self,
        max_chars_per_line: int = 42,
        max_lines: int = 2,
        max_duration: float = 6.0,
        min_duration: float = 1.0,
        optimal_reading_speed_cps: float = 17.0,  # characters per second
        max_reading_speed_cps: float = 20.0,  # characters per second
        min_gap: float = 0.1,  # Minimum gap between segments
        prefer_punctuation_breaks: bool = True,
        language_patterns: Optional[Dict] = None,
        custom_dictionary: Optional[List[str]] = None,
        lexicon_path: str = "words_dictionary.json",
    ):
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.optimal_reading_speed_cps = optimal_reading_speed_cps
        self.max_reading_speed_cps = max_reading_speed_cps
        self.min_gap = min_gap
        self.prefer_punctuation_breaks = prefer_punctuation_breaks
        self.custom_dictionary = set(custom_dictionary) if custom_dictionary else set()
        self.lexicon_path = lexicon_path

        self.sentence_endings = {".", "!", "?"}
        self.clause_endings = {",", ";", ":"}
        # Add any punctuation that should be attached to preceding word
        self.all_punctuation = {
            ".",
            ",",
            "!",
            "?",
            ";",
            ":",
            "-",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            '"',
        }

        # Default language patterns (unchanged)
        self.language_patterns = language_patterns or {
            "conjunctions": {
                "and",
                "but",
                "or",
                "so",
                "yet",
                "because",
                "although",
                "while",
            },
            "prepositions": {"in", "on", "at", "by", "for", "with", "from", "to", "of"},
            "articles": {"the", "a", "an"},
            "break_after": {
                "however",
                "therefore",
                "meanwhile",
                "furthermore",
                "moreover",
            },
        }
        self.contraction_suffix_pattern = re.compile(
            r"""^(?:[’'](?:s|m|re|ve|ll|d|t)$)""", re.IGNORECASE | re.VERBOSE
        )
        # self.lexicon = self._load_lexicon()

    def _load_lexicon(self) -> Dict[str, bool]:
        """
        Loads a lexicon from a JSON file.
        The lexicon is a dictionary where keys are words and values are True.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lexicon_path = os.path.join(current_dir, self.lexicon_path)
        try:
            with open(lexicon_path, "r", encoding="utf-8") as f:
                lexicon = json.load(f)
                lexicon_dict = {word.lower(): True for word in lexicon}
            return lexicon_dict
        except FileNotFoundError:
            logger.warning(
                f"Lexicon file {lexicon_path} not found. Using empty lexicon."
            )
            return {}

    def _handle_apostrophes(self, word_timings: List[WordTiming]) -> List[WordTiming]:
        """
        Whisper sometimes splits contractions like "don't" into "do" and "n't" to separate lines.
        This method merges them back into a single word.
        """
        processed: List[WordTiming] = []
        i = 0
        n = len(word_timings)

        while i < n:
            current = word_timings[i]

            # If there is a next token, check if next token is a contraction/possessive suffix
            if i + 1 < n:
                next_word = word_timings[i + 1]
                # Any token whose .word matches eject
                # (apostrophe + s/m/re/ve/ll/d/’t) as a suffix
                if self.contraction_suffix_pattern.match(next_word.word):
                    merged_word = current.word + next_word.word

                    # Create a new WordTiming spanning from current.start to nxt.end
                    merged_timing = WordTiming(
                        word=merged_word,
                        start=current.start,
                        end=next_word.end,
                    )
                    processed.append(merged_timing)

                    # Skip over next_word since it was merged
                    i += 2
                    continue

            # If no merge happened, just append current as‐is
            processed.append(current)
            i += 1

        return processed

    def _handle_split_words(self, word_timings: List[WordTiming]) -> List[WordTiming]:
        """
        Handles cases where Whisper splits long words into multiple segments.
        i.e. electromagnetism -> electrom - ag - net - ism
        This method merges them back into a single word.

         Uses:
          1) Heuristics to propose merges (short tokens, char‐type continuity).
          2) Lexicon membership for confirmation.
        """
        logger.info("Handling split words...")
        if not word_timings:
            return []

        merged: List[WordTiming] = []
        i = 0
        n = len(word_timings)

        while i < n:
            prefix = word_timings[i].word
            is_subword = word_timings[i].is_subword
            next_is_subword = word_timings[i + 1].is_subword if i + 1 < n else False
            start_time = word_timings[i].start
            end_time = word_timings[i].end

            # If the current token is not a subword (is_subword=False),
            # we can’t merge it with anything before i. Move on.
            if not is_subword and not next_is_subword:
                merged.append(word_timings[i])
                i += 1
                continue

            # Otherwise, try to merge as long as the next tokens are also is_subword=True.
            j = i + 1
            while j < n and word_timings[j].is_subword:
                token_j = word_timings[j].word

                if any(p in token_j for p in self.all_punctuation):
                    break

                # Lexicon check (only merge if prefix+token_j exists),
                # you can do something like:
                candidate = prefix + token_j
                # if (
                #     candidate.lower() not in self.lexicon
                #     and candidate.lower() not in self.custom_dictionary
                # ):
                #     break

                # If passed all tests, absorb this fragment into prefix:
                prefix = candidate
                end_time = word_timings[j].end
                j += 1

            # Build a single WordTiming from [i .. j-1]
            merged_word_timing = WordTiming(
                word=prefix,
                start=start_time,
                end=end_time,
                is_subword=False,  # After merging, it becomes a “full word.”
            )
            merged.append(merged_word_timing)
            i = j

        return merged

    def _preprocess_whisper_timing(
        self, word_timings: List[WordTiming]
    ) -> List[WordTiming]:
        """
        Adds flags to each WordTiming indicating
        punctuation or sentence/clause endings.
        """
        # Handle apostrophes and contractions first
        word_timings = self._handle_apostrophes(word_timings)

        # Merge split words
        word_timings = self._handle_split_words(word_timings)

        processed: List[WordTiming] = []
        for w in word_timings:
            wt = WordTiming(word=w.word, start=w.start, end=w.end, duration=w.duration)
            clean = w.word.strip()
            wt.is_punctuation = clean in self.all_punctuation
            wt.is_sentence_ending = clean in self.sentence_endings
            wt.is_clause_ending = clean in self.clause_endings
            processed.append(wt)
        return processed

    def calculate_speaking_rate(self, word_timings: List[WordTiming]) -> Dict:
        """
        Returns words_per_minute, cps, average durations, etc.
        """
        if not word_timings:
            return {}

        total_duration = word_timings[-1].end - word_timings[0].start
        total_words = len(word_timings)
        total_characters = sum(len(w.word) for w in word_timings)
        wpm = (total_words / total_duration) * 60 if total_duration > 0 else 0

        return {
            "words_per_minute": wpm,
            "characters_per_second": (
                (total_characters / total_duration) if total_duration > 0 else 0
            ),
            "average_word_duration": (
                (total_duration / total_words) if total_words > 0 else 0
            ),
            "total_duration": total_duration,
            "total_words": total_words,
            "speaking_speed_category": self._categorize_speaking_speed(wpm),
            "recommended_max_segment_duration": self._recommend_max_duration(wpm),
        }

    def _categorize_speaking_speed(self, wpm: float) -> str:
        if wpm < 100:
            return "slow"
        elif wpm < 150:
            return "normal"
        elif wpm < 200:
            return "fast"
        else:
            return "very fast"

    def _recommend_max_duration(self, wpm: float) -> float:
        if wpm > 180:
            return min(self.max_duration, 4.0)
        elif wpm > 140:
            return min(self.max_duration, 5.0)
        else:
            return self.max_duration

    def find_natural_breaks(
        self, word_timings: List[WordTiming], min_gap_seconds: float = 0.1
    ) -> List[Tuple[int, float, str]]:
        """
        Finds indices where there's enough gap or clause/sentence endings.
        """
        breaks = []
        for i in range(len(word_timings) - 1):
            cur = word_timings[i]
            nxt = word_timings[i + 1]
            gap = nxt.start - cur.end
            reason = self._analyze_break_type(cur.word, nxt.word, gap)
            if gap >= min_gap_seconds or reason in [
                "sentence_end_punctuation",
                "clause_end_punctuation",
            ]:
                breaks.append((i + 1, gap, reason))
        return breaks

    def _analyze_break_type(self, current_word: str, next_word: str, gap: float) -> str:
        """
        (Unchanged.) Returns a label like "sentence_end_punctuation", "neutral", etc.
        """
        cur = current_word.strip()
        nxt = next_word.strip()

        if cur in self.sentence_endings:
            return "sentence_end_punctuation"
        if cur in self.clause_endings:
            return "clause_end_punctuation"
        if nxt in self.all_punctuation:
            return "avoid_before_punctuation"

        cur_clean = cur.lower().strip(".,!?;:\"'")
        nxt_clean = nxt.lower().strip(".,!?;:\"'")

        if not cur_clean or not nxt_clean:
            return "punctuation_handling"
        if gap > 0.2:
            return "natural_pause"
        if nxt_clean in self.language_patterns["conjunctions"]:
            return "conjunction_before"
        if cur_clean in self.language_patterns["prepositions"]:
            return "avoid_preposition"
        if nxt_clean in self.language_patterns["articles"]:
            return "avoid_article"
        return "neutral"

    def suggest_subtitle_segments(
        self,
        word_timings: List[WordTiming],
        max_duration: Optional[float] = None,
        max_characters_per_line: Optional[int] = None,
    ) -> List[SubtitleSegment]:
        """
        Groups words into subtitle segments based on timing/length.
        """
        if not word_timings:
            return []

        # Preprocess word timings
        word_timings = self._preprocess_whisper_timing(word_timings)

        max_duration = max_duration or self.max_duration
        max_characters = max_characters_per_line or (
            self.max_chars_per_line * self.max_lines
        )

        analysis = self.calculate_speaking_rate(word_timings)
        adaptive_max = analysis.get("recommended_max_segment_duration", max_duration)

        natural_breaks = self.find_natural_breaks(word_timings)
        break_indices = {bp[0]: (bp[1], bp[2]) for bp in natural_breaks}

        segments: List[SubtitleSegment] = []
        current_start_idx = 0
        i = 0

        while i < len(word_timings):
            potential_segment = self._build_potential_segment(
                word_timings, current_start_idx, i
            )

            should_break = self._should_break_segment(
                potential_segment,
                adaptive_max,
                max_characters,
                i in break_indices,
                break_indices.get(i, (0, "neutral"))[1],
                i == len(word_timings) - 1,
            )

            if should_break["should_break"]:
                seg = SubtitleSegment(
                    start=word_timings[current_start_idx].start,
                    end=word_timings[i].end,
                    text=potential_segment["text"],
                    word_count=potential_segment["word_count"],
                    char_count=potential_segment["char_count"],
                    break_reason=should_break["reason"],
                )
                segments.append(seg)
                current_start_idx = i + 1
            i += 1

        if current_start_idx < len(word_timings):
            final_pot = self._build_potential_segment(
                word_timings, current_start_idx, len(word_timings) - 1
            )
            seg = SubtitleSegment(
                start=word_timings[current_start_idx].start,
                end=word_timings[-1].end,
                text=final_pot["text"],
                word_count=final_pot["word_count"],
                char_count=final_pot["char_count"],
                break_reason="final_segment",
            )
            segments.append(seg)

        return self._optimize_segments(segments)

    def _build_potential_segment(
        self, word_timings: List[WordTiming], start_idx: int, end_idx: int
    ) -> Dict:
        """
        Build a dictionary containing text, word_count, char_count, duration.
        """
        if start_idx > end_idx:
            return {
                "text": "",
                "word_count": 0,
                "char_count": 0,
                "duration": 0.0,
            }

        # Use the new, simpler _reconstruct_text
        segment_words = word_timings[start_idx : end_idx + 1]
        text = self._reconstruct_text(segment_words)

        # Count actual words
        actual_words = [
            w.word.strip()
            for w in segment_words
            if w.word.strip() and w.word.strip() not in self.all_punctuation
        ]
        return {
            "text": text,
            "word_count": len(actual_words),
            "char_count": len(text.replace(" ", "")),  # count non-space chars
            "duration": word_timings[end_idx].end - word_timings[start_idx].start,
        }

    def _reconstruct_text(self, word_timings: List[WordTiming]) -> str:
        """
        Does text reconstruction:

        - Strips each WordTiming.word.
        - If it's pure punctuation (one of self.all_punctuation),
            attach it directly to the previous token.
        - Otherwise, collect it as a normal word.
        - Finally, join with exactly one space and collapse any accidental extra whitespace.
        """
        if not word_timings:
            return ""

        tokens: List[str] = []
        for wt in word_timings:
            w = wt.word.strip()
            if not w:
                continue

            # If this stripped token is exactly punctuation (e.g. ".", ",", etc.)
            # and we have a previous token, attach it to the last token
            if w in self.all_punctuation:
                if tokens:
                    tokens[-1] = tokens[-1] + w
                else:
                    # If there's no previous token, just treat punctuation as its own token
                    tokens.append(w)
            else:
                # If previous token is a sentence-ending punctuation, capitalize the next word
                if tokens and tokens[-1] and tokens[-1][-1] in self.sentence_endings:
                    w = w.capitalize()
                # Check if previous token is a clause-ending punctuation, and lower case
                elif tokens and tokens[-1] and tokens[-1][-1] in self.clause_endings:
                    w = w.lower()
                # Normal word: append to list
                tokens.append(w)

        # Join everything with single spaces
        text = " ".join(tokens)

        # Just in case: collapse any multiple spaces, and strip ends
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _should_break_segment(
        self,
        potential_segment: Dict,
        max_duration: float,
        max_characters: int,
        is_natural_break: bool,
        break_type: str,
        is_last_word: bool,
    ) -> Dict:
        """ """
        duration = potential_segment["duration"]
        char_count = potential_segment["char_count"]
        if duration > max_duration:
            return {"should_break": True, "reason": f"duration_limit ({duration:.1f}s)"}

        if char_count > max_characters:
            return {
                "should_break": True,
                "reason": f"character_limit ({char_count} chars)",
            }

        if is_last_word:
            return {"should_break": True, "reason": "last_word"}

        # High priority: break on punctuation if segment isn’t tiny
        if break_type in ["sentence_end_punctuation", "clause_end_punctuation"]:
            if char_count > 15:
                return {"should_break": True, "reason": f"punct_{break_type}"}

        # Avoid breaking right before prepositions/articles/punctuation
        if break_type in [
            "avoid_preposition",
            "avoid_article",
            "avoid_before_punctuation",
        ]:
            return {"should_break": False, "reason": f"avoid_{break_type}"}

        # Reading speed optimization
        reading_speed = (char_count / duration) if duration > 0 else 0
        if reading_speed > self.max_reading_speed_cps and is_natural_break:
            return {
                "should_break": True,
                "reason": f"reading_speed ({reading_speed:.1f} cps)",
            }

        # Natural pause, if the segment is already somewhat long
        if break_type == "natural_pause" and char_count > 25:
            return {"should_break": True, "reason": "natural_pause"}

        # Conjunction breaks
        if break_type == "conjunction_before" and char_count > 30:
            return {"should_break": True, "reason": "conjunction_break"}

        return {"should_break": False, "reason": "continue_building"}

    def _optimize_segments(
        self, segments: List[SubtitleSegment]
    ) -> List[SubtitleSegment]:
        """
        Ensure minimum durations and minimal gaps between consecutive segments.
        """
        if not segments:
            return []

        optimized = []
        for seg in segments:
            # Enforce minimum display time
            if seg.duration and seg.duration < self.min_duration:
                seg.end = seg.start + self.min_duration

            if optimized:
                prev = optimized[-1]
                gap = seg.start - prev.end
                if gap < self.min_gap:
                    # Push them apart evenly
                    center = (prev.end + seg.start) / 2
                    prev.end = center - (self.min_gap / 2)
                    seg.start = center + (self.min_gap / 2)

            optimized.append(seg)

        return optimized
