"""
financial/healer.py
===================
Financial Self-Healing Layer.
Corrects contradicted financial figures and margins in-place,
appends verified regulatory citations, and provides an audit-grounded response.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .config import ClaimStatus, FinancialRiskLevel
from .detector import FinancialVerificationReport, VerifiedClaim


@dataclass
class FinancialHealingResult:
    """Outcome of healing an erroneous or hallucinated financial answer."""
    original_answer: str
    healed_answer: str
    healing_applied: bool
    patches_made: List[str] = field(default_factory=list)
    citations_appended: List[str] = field(default_factory=list)
    initial_risk: FinancialRiskLevel = FinancialRiskLevel.LOW
    final_risk: FinancialRiskLevel = FinancialRiskLevel.LOW

    def summary(self) -> str:
        lines = [
            f"  Healing Applied: {self.healing_applied}",
            f"  Risk Reduction : {self.initial_risk.value} → {self.final_risk.value}",
            f"  Patches Applied: {len(self.patches_made)}",
        ]
        for p in self.patches_made:
            lines.append(f"    — {p}")
        lines.append(f"\n  Healed Response:\n{self.healed_answer}")
        return "\n".join(lines)


class FinancialHealer:
    """
    In-place patcher for financial hallucinations.
    Replaces false numbers with audited facts and references verified filings.
    """

    def heal(
        self,
        report: FinancialVerificationReport,
        append_footnotes: bool = True
    ) -> FinancialHealingResult:
        """
        Take a verification report and return a corrected, grounded response.
        """
        if not report.is_hallucinating and report.contradicted_count == 0:
            return FinancialHealingResult(
                original_answer=report.answer,
                healed_answer=report.answer,
                healing_applied=False,
                patches_made=[],
                citations_appended=[],
                initial_risk=report.risk_level,
                final_risk=report.risk_level,
            )

        healed_text = report.answer
        patches: List[str] = []
        citations: List[str] = []
        footnote_idx = 1

        # Patch each contradicted claim
        for vc in report.verified_claims:
            if vc.status == ClaimStatus.CONTRADICTED and vc.evidence_value_str:
                claimed_str = vc.claimed_value_str
                correct_str = vc.evidence_value_str
                
                escaped_claimed = re.escape(claimed_str)
                # If replacing a percentage, consume any trailing % sign in the original text
                if "%" in correct_str:
                    pattern = rf"{escaped_claimed}\s*%?"
                else:
                    pattern = escaped_claimed

                replacement = f"{correct_str} [{footnote_idx}]"
                new_text, n = re.subn(pattern, replacement, healed_text, count=1, flags=re.IGNORECASE)

                if n > 0:
                    healed_text = new_text
                    patches.append(f"Corrected '{claimed_str}' → '{correct_str}' ({vc.claim.metric_label})")
                    if vc.evidence_source:
                        quote_snip = f' - "{vc.evidence_quote}"' if vc.evidence_quote else ""
                        citations.append(f"[{footnote_idx}] {vc.evidence_source}{quote_snip}")
                    footnote_idx += 1
                else:
                    # Fallback word-level number patch
                    num_only = str(int(vc.claim.numeric_value) if vc.claim.numeric_value.is_integer() else vc.claim.numeric_value)
                    if num_only in healed_text:
                        healed_text = healed_text.replace(num_only, f"{correct_str} [{footnote_idx}]", 1)
                        patches.append(f"Corrected figure '{num_only}' → '{correct_str}'")
                        if vc.evidence_source:
                            quote_snip = f' - "{vc.evidence_quote}"' if vc.evidence_quote else ""
                            citations.append(f"[{footnote_idx}] {vc.evidence_source}{quote_snip}")
                        footnote_idx += 1

        # Append citations as footnotes if requested
        if append_footnotes and citations:
            healed_text += "\n\n────────────────────────────────────────\nVerified Sources & Citations:\n"
            for cite in citations:
                healed_text += f"  {cite}\n"

        return FinancialHealingResult(
            original_answer=report.answer,
            healed_answer=healed_text,
            healing_applied=len(patches) > 0,
            patches_made=patches,
            citations_appended=citations,
            initial_risk=report.risk_level,
            final_risk=FinancialRiskLevel.LOW if len(patches) == report.contradicted_count else FinancialRiskLevel.MEDIUM,
        )
