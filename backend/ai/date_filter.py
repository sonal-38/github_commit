"""
Chronological and date constraint detection and parsing module.
Parses natural language date specifications into structured ISO 8601 UTC time ranges
and identifies target GitHub object types for exact relational filtering.
"""
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}


class DateConstraint:
    """Represents an extracted chronological filter condition."""
    def __init__(
        self,
        start_iso: Optional[str] = None,
        end_iso: Optional[str] = None,
        target_entity: Optional[str] = None,   # 'commits', 'pull_requests', 'issues', 'reviews', etc.
        date_field: Optional[str] = None,      # 'committed_at', 'created_at', 'merged_at', etc.
        is_transition: bool = False,           # transition / sequence query (before/after/transition)
        relative_keyword: Optional[str] = None,# 'recent', 'latest'
        raw_match: Optional[str] = None,
    ):
        self.start_iso = start_iso
        self.end_iso = end_iso
        self.target_entity = target_entity
        self.date_field = date_field
        self.is_transition = is_transition
        self.relative_keyword = relative_keyword
        self.raw_match = raw_match

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date_start": self.start_iso,
            "date_end": self.end_iso,
            "target_entity": self.target_entity,
            "date_field": self.date_field,
            "is_transition": self.is_transition,
            "relative_keyword": self.relative_keyword,
        }


class DateQueryDetector:
    """Detects and parses date/chronological constraints from natural language questions."""

    @classmethod
    def detect(cls, query: str) -> Optional[DateConstraint]:
        q = query.strip()
        q_lower = q.lower()

        # 1. Detect target entity and specific date column
        target_entity, date_field = cls._detect_entity_and_field(q_lower)

        # 2. Check for transition / sequential reasoning keywords
        is_transition = any(word in q_lower for word in [
            "transition", "evolve", "evolution", "chronological", "timeline",
            "from the old", "to the new", "before the", "after the", "what changed between"
        ])

        # 3. Check for relative latest/recent patterns
        has_recent = bool(re.search(r"\b(recent|recently|latest|newest)\b", q_lower))
        if has_recent:
            return DateConstraint(
                target_entity=target_entity,
                date_field=date_field,
                is_transition=True,
                relative_keyword="recent",
                raw_match="recent",
            )

        # 4. Pattern: "between <Month> <Day> and <Month> <Day> [Year]"
        # e.g., "between August 1 and August 15, 2026", "between August 1 and 15, 2026"
        between_match = re.search(
            r"between\s+([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{4}))?\s+and\s+([A-Za-z]+)?\s*(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?",
            q_lower
        )
        if between_match:
            m1_name, d1_str, y1_str, m2_name, d2_str, y2_str = between_match.groups()
            year = int(y2_str or y1_str or datetime.now(timezone.utc).year)
            m1 = MONTH_NAMES.get(m1_name)
            m2 = MONTH_NAMES.get(m2_name) if m2_name else m1
            d1 = int(d1_str)
            d2 = int(d2_str)

            if m1 and m2:
                start_dt = datetime(year, m1, d1, 0, 0, 0, tzinfo=timezone.utc)
                end_dt = datetime(year, m2, d2, 23, 59, 59, 999999, tzinfo=timezone.utc)
                return DateConstraint(
                    start_iso=start_dt.isoformat(),
                    end_iso=end_dt.isoformat(),
                    target_entity=target_entity,
                    date_field=date_field,
                    is_transition=is_transition,
                    raw_match=between_match.group(0),
                )

        # 5. Pattern: "after/since/from <Month> <Day>, <Year>" (with day specified)
        # Note: Must ensure day is not part of a 4-digit year like "from August 2026"
        after_day_match = re.search(
            r"\b(?:after|since|from)\s+([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b",
            q_lower
        )
        if after_day_match:
            m_name, d_str, y_str = after_day_match.groups()
            m = MONTH_NAMES.get(m_name)
            if m:
                day = int(d_str)
                # Ensure day is a valid day of month (1-31), not a 4-digit year fragment
                if 1 <= day <= 31:
                    year = int(y_str) if y_str else datetime.now(timezone.utc).year
                    start_dt = datetime(year, m, day, 0, 0, 0, tzinfo=timezone.utc)
                    return DateConstraint(
                        start_iso=start_dt.isoformat(),
                        end_iso=None,
                        target_entity=target_entity,
                        date_field=date_field,
                        is_transition=is_transition,
                        raw_match=after_day_match.group(0),
                    )

        # 6. Pattern: "before/until/prior to <Month> <Day>, <Year>" (with day specified)
        before_day_match = re.search(
            r"\b(?:before|until|prior to)\s+([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?\b",
            q_lower
        )
        if before_day_match:
            m_name, d_str, y_str = before_day_match.groups()
            m = MONTH_NAMES.get(m_name)
            if m:
                day = int(d_str)
                if 1 <= day <= 31:
                    year = int(y_str) if y_str else datetime.now(timezone.utc).year
                    end_dt = datetime(year, m, day, 0, 0, 0, tzinfo=timezone.utc)
                    return DateConstraint(
                        start_iso=None,
                        end_iso=end_dt.isoformat(),
                        target_entity=target_entity,
                        date_field=date_field,
                        is_transition=is_transition,
                        raw_match=before_day_match.group(0),
                    )

        # 7. Pattern: Month + 4-digit Year (e.g., "in August 2026", "during August 2026", "from August 2026", "August 2026")
        month_year_match = re.search(
            r"\b(?:in|during|for|from)?\s*([A-Za-z]+)\s+(\d{4})\b",
            q_lower
        )
        if month_year_match:
            m_name, y_str = month_year_match.groups()
            m = MONTH_NAMES.get(m_name)
            if m:
                year = int(y_str)
                start_dt = datetime(year, m, 1, 0, 0, 0, tzinfo=timezone.utc)
                if m == 12:
                    end_dt = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
                else:
                    end_dt = datetime(year, m + 1, 1, 0, 0, 0, tzinfo=timezone.utc)

                return DateConstraint(
                    start_iso=start_dt.isoformat(),
                    end_iso=end_dt.isoformat(),
                    target_entity=target_entity,
                    date_field=date_field,
                    is_transition=is_transition,
                    raw_match=month_year_match.group(0).strip(),
                )

        # 8. Pattern: Just a 4-digit Year (e.g. "in 2026", "during 2026")
        year_match = re.search(r"\b(?:in|during|throughout)\s+(\d{4})\b", q_lower)
        if year_match:
            year = int(year_match.group(1))
            start_dt = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            end_dt = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            return DateConstraint(
                start_iso=start_dt.isoformat(),
                end_iso=end_dt.isoformat(),
                target_entity=target_entity,
                date_field=date_field,
                is_transition=is_transition,
                raw_match=year_match.group(0),
            )

        # 9. If query is a transition question without explicit dates (e.g. "What happened before the caching changes?")
        if is_transition:
            return DateConstraint(
                target_entity=target_entity,
                date_field=date_field,
                is_transition=True,
                raw_match="transition",
            )

        return None

    @classmethod
    def _detect_entity_and_field(cls, q_lower: str) -> Tuple[Optional[str], Optional[str]]:
        """Maps question terminology to exact Supabase table and timestamp column."""
        # Commits
        if "commit" in q_lower:
            return "commits", "committed_at"

        # Pull Requests
        if "pr" in q_lower or "pull request" in q_lower:
            if "merge" in q_lower:
                return "pull_requests", "merged_at"
            if "close" in q_lower:
                return "pull_requests", "closed_at"
            if "update" in q_lower:
                return "pull_requests", "updated_at"
            return "pull_requests", "created_at"

        # Reviews / Review Comments
        if "review comment" in q_lower or "inline comment" in q_lower:
            return "review_comments", "created_at"
        if "review" in q_lower:
            return "reviews", "submitted_at"

        # Issues / Issue Comments
        if "issue comment" in q_lower or "discussion comment" in q_lower:
            return "issue_comments", "created_at"
        if "issue" in q_lower:
            if "close" in q_lower:
                return "issues", "closed_at"
            return "issues", "created_at"

        # Default fallback if general question
        return None, None
