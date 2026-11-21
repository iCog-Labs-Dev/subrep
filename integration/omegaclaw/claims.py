"""Check what an explanation claims against the SubRep evidence it was given.

The checks are deliberately narrow. Each pattern matches a specific wrong claim
(for example an operator exclusion described as a certification failure) in one
clause, and a claim that is negated or hypothetical ("not a certification
failure", "would have selected") is not counted. A paraphrase the patterns do not
cover passes; the live evaluation estimates how often that happens.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .contracts import (
    CERTIFICATION_STAGE,
    FACTUAL_FINDING,
    OBJECTIVE_COMPATIBILITY_STAGE,
    OPERATOR_STAGE,
    RISK_BUDGET_STAGE,
    RUNTIME_ADMISSIBILITY_STAGE,
    Finding,
    RecommendationRequest,
)

# Words within the four before a claim that make it negated or hypothetical.
_GUARD_WORDS = frozenset(
    {
        "not",
        "never",
        "no",
        "nor",
        "neither",
        "without",
        "rather",
        "instead",
        "unrelated",
        "independent",
        "independently",
        "regardless",
        "would",
        "could",
        "might",
        "if",
        "otherwise",
        "unless",
        "nothing",
        "none",
    }
)
_GUARD_WINDOW = 4
_SENTENCE_BREAK = re.compile(r"(?<=[.!?;])\s+")
_CLAUSE_BREAK = re.compile(
    r"\s*[,:\u2013\u2014]\s+|\s+(?:and|but|while|whereas|although|though)\s+"
)
_CARRY_SUBJECT = re.compile(r"^(?:it|its|this|that|the skill|this skill|that skill)\b")

_EXCLUDE_VERB = (
    r"(?:exclu\w*|remov\w*|reject\w*|filter\w*|eliminat\w*|drop\w*|block\w*|"
    r"disqualif\w*|ruled\s+out|screened\s+out)"
)
_CERTIFICATION = (
    r"(?:(?:sub ?rep\s+)?(?:cds\s*(?:/|or|and)\s*pds|cds|pds)\s+)?"
    r"(?:certification(?:\s+gates?)?|gates?|admission\s+gates?)"
)
# "Above the minimum" and "below the maximum" are within budget, so each limit has its own verbs.
_RISK_TARGET = r"(?:the\s+|its\s+|this\s+request(?:'s)?\s+)?(?:local\s+)?"
_RISK_MAXIMUM = (
    r"(?:risk[- ]budget|risk\s+limit|epsilon\s+(?:limit|cap|maximum)|"
    r"maximum(?:\s+certificate)?\s+epsilon|max_certificate_epsilon|maximum)"
)
_RISK_MINIMUM = r"(?:minimum(?:\s+admission)?\s+margin|minimum_admission_margin|minimum)"
_RUNTIME = r"runtime(?:[- ]admissibility)?(?:\s+(?:check|query|test))?"

_STAGE_CLAIMS: Mapping[str, tuple[re.Pattern[str], ...]] = {
    CERTIFICATION_STAGE: tuple(
        re.compile(pattern)
        for pattern in (
            rf"\bfail(?:ed|s|ing)?\s+(?:the\s+|its\s+|their\s+)?{_CERTIFICATION}",
            rf"\bfailure\s+(?:of|in|at)\s+(?:the\s+|its\s+)?{_CERTIFICATION}",
            rf"\b(?:did\s+not|didn't|does\s+not|doesn't|could\s+not|couldn't)\s+pass\s+"
            rf"(?:the\s+|its\s+)?{_CERTIFICATION}",
            rf"\b{_CERTIFICATION}\s+(?:failure|failed|rejected|rejection)\b",
            rf"\b{_EXCLUDE_VERB}\b(?:\s+\S+){{0,4}}?\s+(?:during|in|at|by|through|from|via)\s+"
            rf"(?:the\s+|its\s+|a\s+)?{_CERTIFICATION}\b",
            rf"\b{_CERTIFICATION}\s+(?:process\s+|stage\s+|step\s+)?{_EXCLUDE_VERB}",
            r"\b(?:not|never)\s+(?:been\s+)?certified\b",
            r"\buncertified\b",
            r"\b(?:holds|has|had|have|with|carries|carry)\s+no\s+(?:valid\s+)?certificate\b",
            r"\b(?:lacks?|lacking)\s+(?:a\s+)?(?:valid\s+)?certificate\b",
        )
    ),
    OPERATOR_STAGE: tuple(
        re.compile(pattern)
        for pattern in (
            rf"\b(?:{_EXCLUDE_VERB}|overrid\w*|vetoed)\b(?:\s+\S+){{0,3}}?\s+"
            r"(?:by|per|on|following|under)\s+(?:an?\s+|the\s+)?(?:explicit\s+)?"
            r"(?:operator|human|user|manual)",
            r"\b(?:operator|manual|user|human)[- ](?:exclusion|excluded|removal|removed|"
            r"override|instruction|decision|veto)\b",
            r"\b(?:operator|user|human)\s+(?:has\s+|had\s+)?(?:manually\s+|explicitly\s+)?"
            r"(?:excluded|removed|vetoed|blocked|ruled\s+out)\b",
            r"\bmanually\s+(?:excluded|removed)\b",
        )
    ),
    RUNTIME_ADMISSIBILITY_STAGE: tuple(
        re.compile(pattern)
        for pattern in (
            r"\b(?:not|never)\s+runtime[- ]admissible\b",
            r"\bruntime[- ]inadmissib\w*",
            r"\b(?:inadmissible|not\s+admissible)\s+(?:at\s+runtime|(?:under|for)\s+"
            r"(?:the\s+)?current)",
            rf"\b(?:fail\w*|did\s+not\s+pass|didn't\s+pass)\s+(?:the\s+)?{_RUNTIME}",
            rf"\b{_EXCLUDE_VERB}\b(?:\s+\S+){{0,3}}?\s+(?:by|at|during|in)\s+(?:the\s+)?"
            rf"{_RUNTIME}",
            r"\bruntime\s+(?:admissibility\s+)?(?:check|query|test)\s+(?:did\s+not|didn't|"
            r"does\s+not|doesn't)\s+admit",
            r"\b(?:weight\s+region|certified\s+region|w_x)\s+(?:does|did)(?:n't|\s+not)\s+"
            r"(?:cover|include|contain)",
            r"\b(?:outside|beyond)\s+(?:its|the)\s+(?:certified\s+)?(?:weight\s+)?region",
        )
    ),
    RISK_BUDGET_STAGE: tuple(
        re.compile(pattern)
        for pattern in (
            r"\b(?:exceed\w*|over|above|outside|violat\w*|breach\w*|beyond)\s+"
            + _RISK_TARGET
            + _RISK_MAXIMUM,
            r"\b(?:below|under|short\s+of)\s+" + _RISK_TARGET + _RISK_MINIMUM,
            r"\brisk[- ]budget\s+(?:exclusion|violation|failure|filter)\b",
            rf"\b{_EXCLUDE_VERB}\b(?:\s+\S+){{0,3}}?\s+(?:by|under|for|due\s+to|because\s+of)"
            r"\s+(?:the\s+|its\s+)?(?:local\s+)?(?:risk[- ]budget|risk\s+limit|epsilon|"
            r"admission\s+margin)",
            r"\bepsilon\s+(?:is\s+|was\s+)?(?:too\s+(?:high|large|big)|exceeds|above)",
            r"\bmargin\s+(?:is\s+|was\s+)?(?:too\s+(?:low|small)|insufficient|below)",
            r"\binsufficient\s+(?:admission\s+)?margin\b",
        )
    ),
    OBJECTIVE_COMPATIBILITY_STAGE: tuple(
        re.compile(pattern)
        for pattern in (
            r"\b(?:objective|dimension)\w*\s+(?:count\s+)?mismatch\b",
            r"\bdifferent\s+(?:number|set|count)\s+of\s+objectives\b",
            r"\bwrong\s+number\s+of\s+objectives\b",
            r"\b(?:objectives?|dimensions?)\s+(?:do|does)(?:n't|\s+not)\s+match\b",
        )
    ),
}
_STAGE_NAMES = {
    CERTIFICATION_STAGE: "a certification failure",
    OPERATOR_STAGE: "an operator exclusion",
    RUNTIME_ADMISSIBILITY_STAGE: "runtime inadmissibility",
    RISK_BUDGET_STAGE: "a risk-budget exclusion",
    OBJECTIVE_COMPATIBILITY_STAGE: "an objective-count mismatch",
}

_RUNNER_UP = r"(?:runner[- ]up|second[- ]best)"
_SELECTED_AFTER_ID = (
    r"\s+(?:was|is|has\s+been|got|gets|will\s+be|remains)\s+(?:the\s+)?"
    r"(?:selected|chosen|recommended|picked|selection|recommendation|winner|top\s+choice)\b"
)
_SELECTING_VERB = (
    r"\b(?:selected|selects|chose|chooses|chosen|recommends|recommended|picked|picks)\s+"
    r"(?:the\s+)?(?:skill\s+)?"
)
# A whole number counts only when no word follows it, so "(2 objectives)" is not a score.
_NUMBER = r"(?P<number>-?(?:\d+\.\d+|\.\d+|\d+(?![\d.]*\s*[a-z%])))"
_SCORE_AFTER_ID = (
    r"\s*(?:\(\s*(?:score\s*(?:of|=|:)?\s*)?|\s+at\s+|\s*[=:]\s*|,?\s+(?:with\s+)?"
    r"(?:a\s+|its\s+)?(?:final\s+)?(?:selection\s+)?score\s*(?:of|=|:)?\s*|\s+"
    r"(?:scored|scores|scoring)\s+(?:of\s+)?)" + _NUMBER
)
_SCORE_WORD = re.compile(
    r"\b(?:final\s+)?(?:selection\s+)?(?:score|scored)\s*(?:of|=|:|was|is|at)?\s*"
    r"(?:about\s+|approximately\s+|roughly\s+|~\s*)?" + _NUMBER
)
_DECIMAL = re.compile(r"(?<![\w.])-?(?:\d+\.\d+|\.\d+)(?![\w])")
_PERCENT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%")

_EXECUTION_GENERIC = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:was|were|has\s+been|have\s+been|had\s+been|is\s+being)\s+(?:successfully\s+|"
        r"already\s+|safely\s+)?(?:executed|deployed)\b",
        r"\b(?:after|upon|following|during)\s+(?:its\s+|the\s+)?(?:execution|deployment)\b",
        r"\b(?:execution|deployment)\s+(?:of\s+\S+\s+)?(?:showed|shows|confirmed|confirms|"
        r"demonstrated|demonstrates|succeeded|produced|resulted)\b",
    )
)
_EXECUTION_BY_SUBJECT = (
    r"\s+(?:(?:was|were|has\s+been|have\s+been|had\s+been|got|is\s+being)\s+"
    r"(?:successfully\s+|already\s+|safely\s+)?(?:run|launched|flown|executed|deployed)\b|"
    r"(?:successfully\s+|safely\s+)?(?:ran|executed|succeeded|crashed|landed)\b|"
    r"(?:successfully\s+)?completed\s+(?:the\s+|its\s+)?(?:task|mission|episode|landing)\b)"
)
_REAL_EVIDENCE = tuple(
    re.compile(pattern)
    for pattern in (
        r"\breal[- ]world\b",
        r"\breal\s+(?:results?|data|evidence|outcomes?|performance|measurements?|runs?|"
        r"tests?|flights?|episodes?)\b",
        r"\b(?:observed|measured|empirical|actual|field)\s+(?:results?|data|evidence|"
        r"outcomes?|performance|measurements?|gains?|improvements?)\b",
        r"\bempirically\b",
        r"\bin\s+(?:practice|deployment|production|the\s+real\s+world|the\s+field)\b",
        r"\b(?:was|were)\s+(?:observed|measured)\b",
    )
)
_ABSTENTION_TERMS = ("abstain", "no admissible", "no recommendation")


def explanation_findings(request: RecommendationRequest, explanation: str) -> list[Finding]:
    """Return a factual finding for every wrong claim the checks recognise."""

    text = _normalise(explanation)
    findings: list[Finding] = []
    findings.extend(_decision_named(request, text))
    findings.extend(_role_claims(request, text))
    findings.extend(_exclusion_stage_claims(request, text))
    findings.extend(_score_claims(request, text))
    findings.extend(_unsupported_numbers(request, text))
    findings.extend(_execution_claims(request, text))
    findings.extend(_evidence_label_claims(request, text))
    return findings


def _finding(code: str, message: str) -> Finding:
    return Finding(code, FACTUAL_FINDING, message)


def _normalise(text: str) -> str:
    return " ".join(text.replace("\u2019", "'").replace("\u2212", "-").lower().split())


def _skill_pattern(skill_id: str) -> str:
    return rf"(?<![\w-]){re.escape(skill_id.lower())}(?![\w-])"


def _skill_ids(request: RecommendationRequest) -> list[str]:
    ids = [skill.skill_id for skill in request.admitted_skills]
    ids.extend(exclusion.skill_id for exclusion in request.exclusions)
    return ids


def _guarded(clause: str, start: int) -> bool:
    """True when a claim starting at ``start`` is negated or hypothetical."""

    words = re.findall(r"[a-z_']+", clause[:start])[-_GUARD_WINDOW:]
    return any(word in _GUARD_WORDS or word.endswith("n't") for word in words)


def _asserted(pattern: re.Pattern[str], clause: str) -> Iterable[re.Match[str]]:
    return (match for match in pattern.finditer(clause) if not _guarded(clause, match.start()))


def _decision_named(request: RecommendationRequest, text: str) -> list[Finding]:
    decision = request.subrep_decision
    if decision.abstain:
        if not any(term in text for term in _ABSTENTION_TERMS):
            return [
                _finding(
                    "ABSTENTION_NOT_STATED",
                    "abstention explanation must describe the lack of a recommendation",
                )
            ]
        return []
    if not re.search(_skill_pattern(decision.selected_skill_id), text):
        return [_finding("SELECTED_SKILL_NOT_NAMED", "explanation must mention the selected skill ID")]
    return []


def _role_claims(request: RecommendationRequest, text: str) -> list[Finding]:
    decision = request.subrep_decision
    excluded = {exclusion.skill_id for exclusion in request.exclusions}
    findings: list[Finding] = []
    for skill_id in _skill_ids(request):
        skill = _skill_pattern(skill_id)
        if skill_id != decision.selected_skill_id:
            selected = (
                re.compile(skill + _SELECTED_AFTER_ID),
                re.compile(_SELECTING_VERB + skill),
            )
            if any(_asserted_in_a_sentence(pattern, text) for pattern in selected):
                findings.append(
                    _finding(
                        "WRONG_SKILL_ROLE",
                        f"explanation describes {skill_id!r} as selected, but SubRep "
                        + (
                            "abstained"
                            if decision.abstain
                            else f"selected {decision.selected_skill_id!r}"
                        ),
                    )
                )
        if skill_id == decision.runner_up_skill_id:
            continue
        runner_forms = [
            rf"{skill}\s+(?:was|is|remains|ranked\s+as|served\s+as)\s+(?:the\s+)?{_RUNNER_UP}",
            rf"{_RUNNER_UP}\s+(?:was|is)\s+{skill}",
        ]
        if skill_id in excluded:
            # "excluded_skill, the runner-up" is only unambiguous for a skill that cannot
            # be one; after the selected skill the phrase usually introduces the next name.
            runner_forms.append(rf"{skill}\s*,\s*(?:the\s+)?{_RUNNER_UP}")
        runner = re.compile("|".join(f"(?:{form})" for form in runner_forms))
        if _asserted_in_a_sentence(runner, text):
            if skill_id in excluded:
                message = (
                    f"explanation must not describe excluded skill {skill_id!r} "
                    "as an admitted runner-up"
                )
            elif decision.runner_up_skill_id is None:
                message = (
                    f"explanation describes {skill_id!r} as the runner-up, but SubRep "
                    "has no admitted runner-up"
                )
            else:
                message = (
                    f"explanation describes {skill_id!r} as the runner-up, but SubRep's "
                    f"runner-up is {decision.runner_up_skill_id!r}"
                )
            findings.append(_finding("WRONG_SKILL_ROLE", message))
    return findings


def _asserted_in_a_sentence(pattern: re.Pattern[str], text: str) -> bool:
    """Role claims span commas ("excluded_skill, the runner-up"), so check whole sentences."""

    return any(
        next(_asserted(pattern, sentence), None) is not None
        for sentence in _SENTENCE_BREAK.split(text)
    )


def _clauses(sentence: str) -> list[str]:
    return [clause for clause in _CLAUSE_BREAK.split(sentence) if clause.strip()]


def _exclusion_stage_claims(request: RecommendationRequest, text: str) -> list[Finding]:
    stages = {exclusion.skill_id: exclusion.stage for exclusion in request.exclusions}
    if not stages:
        return []
    ids = _skill_ids(request)
    wrong: dict[str, set[str]] = {}
    previous_sentence_subject: str | None = None
    for sentence in _SENTENCE_BREAK.split(text):
        clauses = _clauses(sentence)
        mentions = [_mentions(clause, ids) for clause in clauses]
        if not any(mentions) and previous_sentence_subject and _CARRY_SUBJECT.match(sentence):
            mentions = [[(0, previous_sentence_subject)] for _ in clauses]
        for index, clause in enumerate(clauses):
            for stage, patterns in _STAGE_CLAIMS.items():
                for pattern in patterns:
                    for match in _asserted(pattern, clause):
                        subject = _subject(mentions, index, match.start())
                        actual = stages.get(subject)
                        if actual is not None and stage != actual:
                            wrong.setdefault(subject, set()).add(stage)
        sentence_ids = [skill_id for clause_mentions in mentions for _, skill_id in clause_mentions]
        if sentence_ids:
            previous_sentence_subject = sentence_ids[-1]
    return [
        _finding(
            "WRONG_EXCLUSION_STAGE",
            f"explanation describes excluded skill {skill_id!r} as "
            f"{' and '.join(_STAGE_NAMES[stage] for stage in sorted(claimed))}, but SubRep "
            f"excluded it at stage {stages[skill_id]} ({_STAGE_NAMES[stages[skill_id]]})",
        )
        for skill_id, claimed in wrong.items()
    ]


def _mentions(clause: str, ids: list[str]) -> list[tuple[int, str]]:
    found = [
        (match.start(), skill_id)
        for skill_id in ids
        for match in re.finditer(_skill_pattern(skill_id), clause)
    ]
    return sorted(found)


def _subject(mentions: list[list[tuple[int, str]]], index: int, position: int) -> str | None:
    """The skill a claim is about: the nearest mention in its clause, else a neighbour."""

    own = mentions[index]
    if own:
        return min(own, key=lambda item: (abs(item[0] - position), item[0] > position))[1]
    for earlier in reversed(mentions[:index]):
        if earlier:
            return earlier[-1][1]
    for later in mentions[index + 1 :]:
        if later:
            return later[0][1]
    return None


def _score_claims(request: RecommendationRequest, text: str) -> list[Finding]:
    scores = {skill.skill_id: skill.score for skill in request.admitted_skills}
    ids = _skill_ids(request)
    wrong: dict[str, set[str]] = {}
    for sentence in _SENTENCE_BREAK.split(text):
        for clause in _clauses(sentence):
            claimed: list[tuple[str, str]] = []
            for skill_id in ids:
                pattern = re.compile(_skill_pattern(skill_id) + _SCORE_AFTER_ID)
                claimed.extend((skill_id, match["number"]) for match in pattern.finditer(clause))
            mentions = _mentions(clause, ids)
            for match in _SCORE_WORD.finditer(clause):
                owner = [skill_id for start, skill_id in mentions if start < match.start()]
                if owner and match.start() - _last_mention_end(clause, owner[-1], match.start()) <= 80:
                    claimed.append((owner[-1], match["number"]))
            for skill_id, number in claimed:
                if skill_id not in scores:
                    wrong.setdefault(skill_id, set()).add(number)
                elif not _matches(number, scores[skill_id]):
                    wrong.setdefault(skill_id, set()).add(number)
    findings = []
    for skill_id, numbers in wrong.items():
        if skill_id in scores:
            message = (
                f"explanation gives {skill_id!r} score {', '.join(sorted(numbers))}, but its "
                f"SubRep score is {scores[skill_id]:.6g}"
            )
        else:
            message = (
                f"explanation gives excluded skill {skill_id!r} a score, but excluded skills "
                "have no score in the request"
            )
        findings.append(_finding("SCORE_MISMATCH", message))
    return findings


def _last_mention_end(clause: str, skill_id: str, before: int) -> int:
    ends = [
        match.end()
        for match in re.finditer(_skill_pattern(skill_id), clause)
        if match.end() <= before
    ]
    return ends[-1] if ends else before


def _matches(written: str, value: float) -> bool:
    """True when ``written`` is ``value`` rounded or truncated to the shown decimals."""

    number = abs(float(written))
    decimals = len(written.split(".", 1)[1]) if "." in written else 0
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9
    return abs(number - abs(value)) <= tolerance


def _unsupported_numbers(request: RecommendationRequest, text: str) -> list[Finding]:
    stripped = text
    for skill_id in _skill_ids(request) + [request.request_id]:
        stripped = re.sub(_skill_pattern(skill_id), " ", stripped)
    allowed = _allowed_numbers(request)
    unsupported: list[str] = []
    for match in _PERCENT.finditer(stripped):
        written = match.group(1)
        decimals = len(written.split(".", 1)[1]) if "." in written else 0
        value = float(written) / 100.0
        tolerance = 0.5 * 10 ** (-(decimals + 2)) + 1e-9
        if not any(abs(value - candidate) <= tolerance for candidate in allowed):
            unsupported.append(match.group(0))
    for match in _DECIMAL.finditer(_PERCENT.sub(" ", stripped)):
        if not any(_matches(match.group(0), candidate) for candidate in allowed):
            unsupported.append(match.group(0))
    if not unsupported:
        return []
    return [
        _finding(
            "UNSUPPORTED_NUMBER",
            f"explanation quotes numbers that are not in the request: {sorted(set(unsupported))}",
        )
    ]


def _allowed_numbers(request: RecommendationRequest) -> list[float]:
    """Every number in the request, plus the products and gaps a score explanation uses."""

    values = [abs(value) for value in _numeric_leaves(request.to_dict())]
    weights = {item.objective_id: item.weight for item in request.objective_weights}
    for skill in request.admitted_skills:
        values.extend(
            abs(weights[name] * delta) for name, delta in skill.objective_deltas.items()
        )
    scores = [skill.score for skill in request.admitted_skills]
    values.extend(abs(a - b) for index, a in enumerate(scores) for b in scores[index + 1 :])
    values.append(1.0)
    return values


def _numeric_leaves(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _numeric_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _numeric_leaves(child)


def _execution_claims(request: RecommendationRequest, text: str) -> list[Finding]:
    ids = "|".join(_skill_pattern(skill_id) for skill_id in _skill_ids(request))
    subject = (
        rf"(?:{ids + '|' if ids else ''}(?:the\s+)?(?:selected|recommended|chosen)\s+skill|"
        r"this\s+skill|the\s+skill|\bit)"
    )
    patterns = _EXECUTION_GENERIC + (re.compile(subject + _EXECUTION_BY_SUBJECT),)
    claims = sorted(
        {
            match.group(0)
            for sentence in _SENTENCE_BREAK.split(text)
            for clause in _clauses(sentence)
            for pattern in patterns
            for match in _asserted(pattern, clause)
        }
    )
    if not claims:
        return []
    return [
        _finding(
            "EXECUTION_CLAIM",
            f"explanation claims execution or its results, but no skill was executed: {claims}",
        )
    ]


def _evidence_label_claims(request: RecommendationRequest, text: str) -> list[Finding]:
    if request.evidence_label != "SYNTHETIC":
        return []
    findings = []
    if "synthetic" not in text:
        findings.append(
            _finding(
                "SYNTHETIC_NOT_DISCLOSED",
                "explanation must say that the evidence is synthetic",
            )
        )
    claims = sorted(
        {
            match.group(0)
            for sentence in _SENTENCE_BREAK.split(text)
            for clause in _clauses(sentence)
            for pattern in _REAL_EVIDENCE
            for match in _asserted(pattern, clause)
        }
    )
    if claims:
        findings.append(
            _finding(
                "SYNTHETIC_PRESENTED_AS_REAL",
                f"explanation presents synthetic evidence as real: {claims}",
            )
        )
    return findings
