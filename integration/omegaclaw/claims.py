"""Check what an explanation claims against the SubRep evidence it was given.

The checks are deliberately narrow. Each pattern matches a specific wrong claim
(for example an operator exclusion described as a certification failure) in one
clause, and a claim that is negated or hypothetical ("not a certification
failure", "would have selected") is not counted. A paraphrase the patterns do not
cover passes, so an explanation without findings has no errors detected by the
checks; it is not proven accurate. A person reviews live responses for that.
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
# A reason word after a negation starts a new claim: in "not ranked because it failed
# certification" the "not" covers "ranked" only, while "not because it failed" denies the reason.
_REASON_WORDS = frozenset({"because", "since", "therefore", "hence", "thus", "after", "which", "who"})
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
# A decision field pasted into the prose, as in '"runner_up_skill_id": "skill_safety"'.
_DECISION_FIELD = re.compile(
    r"(?<![\w.])(?:subrep_decision\.)?(?P<field>selected_skill_id|runner_up_skill_id|abstain|"
    r"selected_score|runner_up_score|score_gap|reason_code)\b"
    r"[\"'\s\\]*(?P<separator>[:=]|\bis\b|\bwas\b|\bequals\b)(?P<before_value>[\"'\s\\]*)"
    r"(?P<value>null|none|true|false|[-+]?(?:\d+\.\d+|\.\d+|\d+)|[a-z0-9][\w-]*)"
)
_SKILL_ID_FIELDS = ("selected_skill_id", "runner_up_skill_id")
_SCORE_FIELDS = ("selected_score", "runner_up_score", "score_gap")
_NULL_WORDS = frozenset({"null", "none"})
# Any word can be a skill ID ("ghost", "unknown-skill"). A role sentence names one
# where an ID stands, so every name found there is compared with the decision,
# except ordinary words that also stand there: "was selected with score 0.82",
# "the selected skill shows", "the admitted runner-up named by the decision".
_NAME = r"(?<![\w-])(?P<name>[a-z0-9](?:[\w-]*[a-z0-9])?)(?![\w-])"
_NOT_A_SKILL_ID = frozenset(
    """
    a an the this that these those it its they them their which what who whom whose
    one ones none nothing no any each every either neither both all some another other
    others such same there here
    is was be been being are were has have had do does did exists exist existed named
    listed given shown reported admitted excluded selected chosen recommended picked
    defined set left required needed kept made found considered ranked scored used
    removed rejected filtered ruled also then now still only not never already again
    instead therefore thus hence so finally ultimately clearly correctly
    deterministically automatically explicitly simply just even
    skill skills option options candidate candidates policy policies choice choices
    decision decisions result results outcome outcomes answer explanation
    recommendation selection winner score scores rule value field fields entry item id
    position slot place request evidence subrep omega omegaclaw model system agent
    operator certificate certification stage budget weight weights objective objectives
    with by under for from as based using on at in of to into onto because since over
    above below among between according due per via after before and or nor but than
    while when where whereas if though although unless without within against despite
    through throughout toward towards upon
    runner runner-up second-best best top first second third last highest lowest
    highest-scoring top-scoring lower-scoring higher-scoring next sole single final
    eligible null status role rank ranking spot title standing
    """.split()
)
# After "the runner-up" a verb can stand where a name could ("the runner-up trails by
# 0.54", "the runner-up shows a delta"); a name is not followed by these words.
_NOT_AFTER_A_NAME = (
    r"(?!\s+(?:by|a|an|the|its|their|this|that|these|those|on|in|of|to|from|over|behind|"
    r"for|than|as)\b)"
)
_ROLE_CLAIMS = (
    ("selected", _NAME + r"\s+(?:was|is|has\s+been|got|gets|will\s+be|remains)\s+(?:the\s+)?"
     r"(?:selected|chosen|recommended|picked)\b"),
    ("selected", _NAME + r"\s+(?:was|is|remains)\s+(?:the\s+)?(?:selection|recommendation|winner|"
     r"top\s+choice)\b"),
    ("selected", r"\b(?:selected|selects|chose|chooses|chosen|recommends|recommended|picked|picks)"
     r"\s+(?:the\s+skill\s+)?" + _NAME),
    ("selected", r"\b(?:selected|chosen|recommended)\s+skill(?:\s+id)?\s+(?:is|was)\s+(?:the\s+)?"
     + _NAME),
    ("runner-up", _NAME + r"\s+(?:was|is|remains|ranked\s+as|served\s+as)\s+(?:the\s+)?"
     r"(?:admitted\s+)?" + _RUNNER_UP),
    ("runner-up", rf"\b{_RUNNER_UP}(?:\s+skill)?(?:\s+id)?\s+(?:is|was)\s+(?:the\s+)?(?:skill\s+)?"
     + _NAME),
    ("runner-up", rf"\b{_RUNNER_UP}(?:\s+skill)?\s+" + _NAME + _NOT_AFTER_A_NAME),
)
_ROLE_CLAIM_PATTERNS = tuple((role, re.compile(pattern)) for role, pattern in _ROLE_CLAIMS)
_SNAKE_CASE_NAME = r"(?<![\w-])(?P<name>[a-z][a-z0-9-]*_[a-z0-9_-]*[a-z0-9])(?![\w-])"
# A whole number counts only when no word follows it, so "(2 objectives)" is not a score.
_NUMBER = r"(?P<number>[-+]?(?:\d+\.\d+|\.\d+|\d+(?![\d.]*\s*[a-z%])))"
_SCORE_AFTER_ID = (
    r"\s*(?:\(\s*(?:score\s*(?:of|=|:)?\s*)?|\s+at\s+|\s*[=:]\s*|,?\s+(?:with\s+)?"
    r"(?:a\s+|its\s+)?(?:final\s+)?(?:selection\s+)?score\s*(?:of|=|:)?\s*|\s+"
    r"(?:scored|scores|scoring)\s+(?:of\s+)?)" + _NUMBER
)
_SCORE_WORD = re.compile(
    r"\b(?:final\s+)?(?:selection\s+)?(?:score|scored)\s*(?:of|=|:|was|is|at)?\s*"
    r"(?:about\s+|approximately\s+|roughly\s+|~\s*)?" + _NUMBER
)
_DECIMAL = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+|\.\d+)(?![\w])")
_TEXT_NUMBER = re.compile(r"(?<![\w.])-?(?:\d+\.\d+|\.\d+|\d+)(?![\w.])")
# Identity and timestamp fields hold digits that are not evidence.
_NUMBER_FREE_FIELDS = frozenset({"request_id", "created_at", "schema_version"})
# "not real, observed, or measured data": the negation covers every item of the list,
# and so does "rather than" or "instead of". A list ends with "or", "and" or "nor", so
# "rather than synthetic, these are real-world results" is not one.
_NEGATED_LIST = re.compile(
    r"\b(not|never|nor|rather\s+than|instead\s+of)\s+"
    r"([a-z-]+(?:\s*,\s*[a-z-]+)*\s*,?\s+(?:or|and|nor)\s+[a-z-]+)"
)
_PERCENT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*%")
# "weights" and "deltas" also arrive as field names: objective_weights, objective_deltas.
_WEIGHT_WORD = re.compile(r"(?:\b|(?<=_))weights?\b")
_DELTA_WORD = re.compile(r"(?:\b|(?<=_))deltas?\b")
# A change written as a direction and a size: "efficiency fell by 0.1" is a delta of -0.1.
_DECLINE_WORD = (
    r"(?:decreas\w*|declin\w*|drop\w*|fall\w*|fell|lower\w*|reduc\w*|lose|loses|losing|lost|"
    r"loss(?:es)?|worsen\w*|costs?|costing|penali[sz]\w*)"
)
_GAIN_WORD = r"(?:increas\w*|rises?|rising|rose|gain\w*|improv\w*|rais\w*|boost\w*)"
_DIRECTION = re.compile(rf"\b(?:(?P<decline>{_DECLINE_WORD})|(?P<gain>{_GAIN_WORD}))\b")
# A weight times a delta is a contribution to the score, not a delta.
_CONTRIBUTION_WORD = re.compile(r"\bweighted\b|\bcontribut\w*|\bproducts?\b|\btimes\b|×")

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
        # "The evidence is real", "it is not synthetic", "non-synthetic data".
        r"\b(?:is|are|was|were)\s+real\b",
        r"\b(?:is|are|was|were)(?:\s+not|n't)\s+synthetic\b",
        r"\bnon-synthetic\b",
    )
)
# Ways a correct abstention is worded: "SubRep abstained", "an abstention",
# "no admissible skill", "no recommendation", "no skill is recommended".
_ABSTENTION_STATED = re.compile(
    r"\babst(?:ain\w*|ention)\b|\bno_admissible_skills\b|\bno\s+admissible\b"
    r"|\bno\s+recommendation\b|\black\s+of\s+(?:a\s+)?recommendation\b"
    r"|\bno\s+(?:skill|candidate|option)\s+(?:is|was|has\s+been)\s+(?:selected|recommended|chosen)\b"
)

# Other roles an explanation can give a skill the request does not contain. Selected
# and runner-up claims are compared with the decision itself in _role_claims.
_UNKNOWN_SKILL_ROLES = (
    (
        "excluded",
        re.compile(
            _NAME
            + r"\s+(?:was|is|has\s+been|got)\s+(?:also\s+)?"
            r"(?:excluded|removed|rejected|filtered\s+out|ruled\s+out)\b"
        ),
    ),
    ("scored", re.compile(_NAME + r"\s+(?:scored|scores)\s+(?:of\s+)?" + _NUMBER)),
    (
        "scored",
        re.compile(
            _NAME + r"\s+(?:has|had)\s+(?:a\s+)?(?:final\s+)?(?:selection\s+)?score\s+"
            r"(?:of\s+)?" + _NUMBER
        ),
    ),
    # A snake_case name is a skill ID, so the looser score forms ("x at 0.5") apply to it.
    ("scored", re.compile(_SNAKE_CASE_NAME + _SCORE_AFTER_ID)),
)


def explanation_findings(request: RecommendationRequest, explanation: str) -> list[Finding]:
    """Return a factual finding for every wrong claim the checks recognise.

    An explanation must also name the decision and, for synthetic evidence, say so.
    """

    text = _normalise(explanation)
    findings: list[Finding] = []
    findings.extend(_decision_named(request, text))
    findings.extend(_synthetic_not_disclosed(request, text))
    findings.extend(_claim_findings(request, text))
    return findings


def claim_findings(request: RecommendationRequest, text: str) -> list[Finding]:
    """Wrong claims in other response text, such as an advisory concern message."""

    return _claim_findings(request, _normalise(text))


def _claim_findings(request: RecommendationRequest, text: str) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(_role_claims(request, text))
    findings.extend(_unknown_skill_claims(request, text))
    findings.extend(_decision_field_claims(request, text))
    findings.extend(_exclusion_stage_claims(request, text))
    findings.extend(_score_claims(request, text))
    findings.extend(_weight_claims(request, text))
    findings.extend(_delta_claims(request, text))
    findings.extend(_unsupported_numbers(request, text))
    findings.extend(_execution_claims(request, text))
    findings.extend(_evidence_label_claims(request, text))
    return findings


def _finding(code: str, message: str) -> Finding:
    return Finding(code, FACTUAL_FINDING, message)


def _normalise(text: str) -> str:
    # An en or em dash right before a number is a minus sign ("\u20130.1").
    text = re.sub(r"[\u2013\u2014](?=\.?\d)", "-", text)
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
    return any(
        (word in _GUARD_WORDS or word.endswith("n't"))
        and not _REASON_WORDS.intersection(words[index + 2 :])
        for index, word in enumerate(words)
    )


def _asserted(pattern: re.Pattern[str], clause: str) -> Iterable[re.Match[str]]:
    return (match for match in pattern.finditer(clause) if not _guarded(clause, match.start()))


def _decision_named(request: RecommendationRequest, text: str) -> list[Finding]:
    decision = request.subrep_decision
    if decision.abstain:
        if not _ABSTENTION_STATED.search(text):
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
    """Compare every skill an explanation names as selected or runner-up with the decision."""

    decision = request.subrep_decision
    ids = {skill_id.lower(): skill_id for skill_id in _skill_ids(request)}
    excluded = {exclusion.skill_id.lower() for exclusion in request.exclusions}
    named: dict[str, set[str]] = {"selected": set(), "runner-up": set()}
    for sentence in _SENTENCE_BREAK.split(text):
        for role, pattern in _ROLE_CLAIM_PATTERNS:
            for match in _asserted(pattern, sentence):
                if _could_be_skill_id(match["name"], request):
                    named[role].add(match["name"])
        # "excluded_skill, the runner-up" is only unambiguous for a skill that cannot be
        # one; after the selected skill the phrase usually introduces the next name.
        for skill_id in excluded:
            appositive = re.compile(rf"{_skill_pattern(skill_id)}\s*,\s*(?:the\s+)?{_RUNNER_UP}")
            if next(_asserted(appositive, sentence), None) is not None:
                named["runner-up"].add(skill_id)

    selected = decision.selected_skill_id.lower() if decision.selected_skill_id else None
    runner_up = decision.runner_up_skill_id.lower() if decision.runner_up_skill_id else None
    actual = "abstained" if decision.abstain else f"selected {decision.selected_skill_id!r}"
    findings: list[Finding] = []
    for name in sorted(named["selected"] - {selected}):
        if name not in ids:
            findings.append(_unknown_skill(name, "selected"))
            continue
        findings.append(
            _finding(
                "WRONG_SKILL_ROLE",
                f"explanation describes {ids[name]!r} as selected, but SubRep {actual}",
            )
        )
    for name in sorted(named["runner-up"] - {runner_up}):
        if name not in ids:
            findings.append(_unknown_skill(name, "the runner-up"))
        elif name in excluded:
            findings.append(
                _finding(
                    "WRONG_SKILL_ROLE",
                    f"explanation must not describe excluded skill {ids[name]!r} as an "
                    "admitted runner-up",
                )
            )
        elif runner_up is None:
            findings.append(
                _finding(
                    "WRONG_SKILL_ROLE",
                    f"explanation describes {ids[name]!r} as the runner-up, but SubRep has no "
                    "admitted runner-up",
                )
            )
        else:
            findings.append(
                _finding(
                    "WRONG_SKILL_ROLE",
                    f"explanation describes {ids[name]!r} as the runner-up, but SubRep's "
                    f"runner-up is {decision.runner_up_skill_id!r}",
                )
            )
    return findings


def _could_be_skill_id(name: str, request: RecommendationRequest) -> bool:
    """False for ordinary words and request field names found where an ID could stand."""

    if name in {skill_id.lower() for skill_id in _skill_ids(request)}:
        return True
    if name in _NOT_A_SKILL_ID or name.isdigit():
        return False
    # Adverbs stand where an ID could: "selected primarily because", "chosen solely on".
    if len(name) > 3 and name.endswith("ly") and not re.search(r"[_\d-]", name):
        return False
    return name not in _request_field_names(request)


def _request_field_names(request: RecommendationRequest) -> set[str]:
    return {
        segment.lower()
        for reference in request.valid_evidence_refs()
        for segment in reference.split(".")
    }


def _unknown_skill(name: str, role: str) -> Finding:
    return _finding(
        "UNKNOWN_SKILL",
        f"explanation describes {name!r} as {role}, but no skill with that ID is in the request",
    )


def _unknown_skill_claims(request: RecommendationRequest, text: str) -> list[Finding]:
    """Skill-like names (snake_case) given a role, though the request does not contain them.

    Omega replays earlier requests from its history, so a model can name a skill
    from another request; request field names such as selected_skill_id are not skills.
    """

    known = {skill_id.lower() for skill_id in _skill_ids(request)}
    roles: dict[str, set[str]] = {}
    for sentence in _SENTENCE_BREAK.split(text):
        for role, pattern in _UNKNOWN_SKILL_ROLES:
            for match in _asserted(pattern, sentence):
                name = match["name"]
                if name not in known and _could_be_skill_id(name, request):
                    roles.setdefault(name, set()).add(role)
    return [_unknown_skill(name, " and ".join(sorted(described))) for name, described in sorted(roles.items())]


def _decision_field_claims(request: RecommendationRequest, text: str) -> list[Finding]:
    """Compare decision fields pasted into the explanation with SubRep's own decision."""

    decision = request.subrep_decision
    known = {skill_id.lower() for skill_id in _skill_ids(request)}
    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()
    for match in _DECISION_FIELD.finditer(text):
        field, value = match["field"], match["value"]
        if (field, value) in seen:
            continue
        seen.add((field, value))
        # "selected_skill_id: ghost" or '"selected_skill_id" is "ghost"' states a value;
        # "selected_skill_id is the field" is prose about the field.
        explicit = match["separator"] in {":", "="} or any(
            quote in match["before_value"] for quote in "\"'"
        )
        finding = _decision_field_problem(field, value, decision, known, explicit=explicit)
        if finding is not None:
            findings.append(finding)
    return findings


def _decision_field_problem(
    field: str, value: str, decision: Any, known: set[str], *, explicit: bool
) -> Finding | None:
    expected = getattr(decision, field)
    is_null = value in _NULL_WORDS
    if field in _SKILL_ID_FIELDS:
        if not is_null and not explicit and value in _NOT_A_SKILL_ID:
            return None
        stated = None if is_null else value
        if stated == (None if expected is None else expected.lower()):
            return None
        unknown = "" if stated is None or stated in known else ", which is not a skill in the request"
        return _finding(
            "WRONG_SKILL_ROLE",
            f"explanation states {field} {value!r}{unknown}, but SubRep's decision has "
            f"{expected!r}",
        )
    if field == "abstain":
        if value not in {"true", "false"} or (value == "true") == decision.abstain:
            return None
        return _finding(
            "DECISION_MISMATCH",
            f"explanation states abstain {value}, but SubRep's decision has abstain "
            f"{str(decision.abstain).lower()}",
        )
    if field in _SCORE_FIELDS:
        numeric = re.fullmatch(r"[-+]?(?:\d+\.\d+|\.\d+|\d+)", value) is not None
        if not numeric and not is_null:
            return None
        if is_null and expected is None:
            return None
        if numeric and expected is not None and _matches(value, expected):
            return None
        return _finding(
            "SCORE_MISMATCH",
            f"explanation states {field} {value}, but SubRep's decision has "
            f"{'null' if expected is None else format(expected, '.6g')}",
        )
    # reason_code
    if not is_null and not explicit and "_" not in value:
        return None  # prose such as "reason_code is the reason", not a pasted value
    if is_null and expected is None:
        return None
    if expected is not None and value == expected.lower():
        return None
    return _finding(
        "DECISION_MISMATCH",
        f"explanation states reason_code {value}, but SubRep's decision has {expected!r}",
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
    runner_up = request.subrep_decision.runner_up_skill_id
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
                if not owner:
                    continue
                owner_end = _last_mention_end(clause, owner[-1], match.start())
                if match.start() - owner_end > 80:
                    continue
                # "skill_safety was selected over the runner-up score of 0.28": the words
                # between give the score to the runner-up, not to the skill named before.
                if re.search(rf"{_RUNNER_UP}|runner_up", clause[owner_end : match.start()]):
                    if runner_up is not None and runner_up != owner[-1]:
                        claimed.append((runner_up, match["number"]))
                    continue
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


def _weight_claims(request: RecommendationRequest, text: str) -> list[Finding]:
    """Check that each weight is given to its own objective.

    A live answer wrote "objective weights (safety 0.2, efficiency 0.8)" for weights of
    0.8 and 0.2: every number exists in the request, so only the pairing is wrong.
    """

    weights = {item.objective_id.lower(): item.weight for item in request.objective_weights}
    names = "|".join(re.escape(name) for name in sorted(weights, key=len, reverse=True))
    pair = re.compile(
        rf"(?<![\w.])(?P<objective>{names})(?![\w-])\s*(?:of\s+|=|:|at\s+|is\s+)?\s*" + _NUMBER
    )
    named_weight = re.compile(
        rf"(?<![\w.])(?P<objective>{names})(?![\w-])\s+(?:(?:carries|has|had|with|at)\s+"
        r"(?:a\s+|an\s+)?)?(?:objective\s+)?weight\s*(?:of\s+|=|:|is\s+)?\s*" + _NUMBER
    )
    # Deltas and skills follow weights in most answers; stop reading a weight list there.
    skills = "|".join(_skill_pattern(skill_id) for skill_id in _skill_ids(request))
    list_end = re.compile(r"deltas?\b|delta_r|\bscores?\b|;" + (f"|{skills}" if skills else ""))
    stated: list[tuple[str, str]] = []
    for sentence in _SENTENCE_BREAK.split(text):
        for word in _WEIGHT_WORD.finditer(sentence):
            window = sentence[word.end() : word.end() + 120]
            end = list_end.search(window)
            window = window[: end.start()] if end else window
            stated.extend((match["objective"], match["number"]) for match in pair.finditer(window))
        stated.extend(
            (match["objective"], match["number"]) for match in named_weight.finditer(sentence)
        )
    wrong: dict[str, set[str]] = {}
    for objective, number in stated:
        if not _matches(number, weights[objective]):
            wrong.setdefault(objective, set()).add(number)
    return [
        _finding(
            "WEIGHT_MISMATCH",
            f"explanation gives objective {objective!r} weight {', '.join(sorted(numbers))}, but "
            f"its SubRep weight is {weights[objective]:.6g}",
        )
        for objective, numbers in sorted(wrong.items())
    ]


def _delta_claims(request: RecommendationRequest, text: str) -> list[Finding]:
    """Check that each objective delta is given to its own skill and objective, with its sign.

    A live answer wrote "both skills show objective_deltas of 0.8 for safety and 0.2 for
    efficiency"; each number is in the request, but no skill has those deltas. A delta
    stated as a value keeps its sign, so "an efficiency delta of 0.1" is wrong for -0.1,
    while "efficiency decreased by 0.1" states the same -0.1 in words.
    """

    deltas = {
        skill.skill_id.lower(): {name.lower(): value for name, value in skill.objective_deltas.items()}
        for skill in request.admitted_skills
    }
    if not deltas:
        return []
    names = "|".join(
        re.escape(item.objective_id.lower())
        for item in sorted(request.objective_weights, key=lambda item: -len(item.objective_id))
    )
    direction = rf"(?:{_DECLINE_WORD}|{_GAIN_WORD})"
    pairs = re.compile(
        rf"(?<![\w.])(?P<objective>{names})(?![\w-])\s*(?:objective\s+)?(?P<delta_word>deltas?\s*)?"
        rf"(?:(?P<verb>{direction})\s+)?(?:by\s+|of\s+|=|:)?\s*{_NUMBER}"
        rf"|(?<![\w.])(?P<number_first>[-+]?(?:\d+\.\d+|\.\d+))\s+(?:{direction}\s+)?(?:for|on|in)\s+"
        rf"(?P<objective_after>{names})(?![\w-])"
    )
    owners = re.compile(
        "|".join(_skill_pattern(skill_id) for skill_id in deltas)
        + r"|\b(?:both|each|all)\s+(?:admitted\s+)?skills?\b"
    )
    wrong: dict[tuple[str, str], set[str]] = {}
    for sentence in _SENTENCE_BREAK.split(text):
        delta_word = _DELTA_WORD.search(sentence)
        previous_end = 0
        for match in pairs.finditer(sentence):
            objective = match["objective"] or match["objective_after"]
            number = match["number"] or match["number_first"]
            own_word = bool(match["delta_word"] or match["verb"])
            # A pair is a delta claim when it says "delta" or a change itself, or follows
            # the word "delta" in its sentence with no "weight" in between.
            after_delta = (
                delta_word is not None
                and delta_word.end() <= match.start()
                and not _WEIGHT_WORD.search(sentence[delta_word.end() : match.start()])
            )
            owner = None
            for found in owners.finditer(sentence[: match.start()]):
                owner = found
            # Direction words count from the previous pair, so each change has its own.
            wording = sentence[max(previous_end, owner.end() if owner else 0) : match.end()]
            previous_end = match.end()
            if owner is None:
                continue
            # "gains 0.8 on safety and 0.1 on efficiency" states two changes with one verb.
            own_word = own_word or bool(
                match["number_first"] and _DIRECTION.search(sentence[owner.end() : match.end()])
            )
            if not (own_word or after_delta):
                continue
            if _CONTRIBUTION_WORD.search(sentence[owner.end() : match.start()]):
                continue
            stated = _stated_delta(number, wording)
            claimed_by = list(deltas) if owner.group(0) not in deltas else [owner.group(0)]
            for skill_id in claimed_by:
                if abs(stated - deltas[skill_id][objective]) > _tolerance(number):
                    wrong.setdefault((skill_id, objective), set()).add(number)
    return [
        _finding(
            "DELTA_MISMATCH",
            f"explanation gives {skill_id!r} the {objective} delta {', '.join(sorted(numbers))}, "
            f"but its SubRep {objective} delta is {deltas[skill_id][objective]:.6g}",
        )
        for (skill_id, objective), numbers in sorted(wrong.items())
    ]


def _stated_delta(number: str, wording: str) -> float:
    """The delta a number states: its own sign, else the direction words around it."""

    value = float(number)
    if number[0] in "+-":
        return value
    found = None
    for found in _DIRECTION.finditer(wording):
        pass
    return -value if found is not None and found["decline"] else value


def _last_mention_end(clause: str, skill_id: str, before: int) -> int:
    ends = [
        match.end()
        for match in re.finditer(_skill_pattern(skill_id), clause)
        if match.end() <= before
    ]
    return ends[-1] if ends else before


def _tolerance(written: str) -> float:
    """Half a unit of the last decimal shown, so rounding or truncation still matches."""

    decimals = len(written.split(".", 1)[1]) if "." in written else 0
    return 0.5 * 10 ** (-decimals) + 1e-9


def _matches(written: str, value: float) -> bool:
    """True when ``written`` is ``value`` to the shown decimals, sign included.

    Used for scores: a score of -0.4 written as 0.4 is a wrong score.
    """

    return abs(float(written) - value) <= _tolerance(written)


def _matches_number(written: str, value: float) -> bool:
    """Like ``_matches`` when a sign is written; otherwise compare sizes only.

    An unsigned number may describe a decline ("safety fell by 0.1" for a delta of
    -0.1), so only an explicit sign is held to the sign of the evidence.
    """

    if written[0] in "+-":
        return _matches(written, value)
    return abs(float(written) - abs(value)) <= _tolerance(written)


def _unsupported_numbers(request: RecommendationRequest, text: str) -> list[Finding]:
    stripped = text
    for skill_id in _skill_ids(request) + [request.request_id]:
        stripped = re.sub(_skill_pattern(skill_id), " ", stripped)
    allowed = _allowed_numbers(request)
    unsupported: list[str] = []
    for match in _PERCENT.finditer(stripped):
        written = match.group(1)
        value = float(written) / 100.0
        tolerance = _tolerance(written) / 100.0
        if not any(abs(value - abs(candidate)) <= tolerance for candidate in allowed):
            unsupported.append(match.group(0))
    for match in _DECIMAL.finditer(_PERCENT.sub(" ", stripped)):
        if not any(_matches_number(match.group(0), candidate) for candidate in allowed):
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
    """Every number in the request, with its sign, plus the products and gaps a score
    explanation uses. Gaps between scores are sizes, so they are positive."""

    payload = request.to_dict()
    values = list(_numeric_leaves(payload))
    # Numbers written inside the request text count too, such as the epsilon in an
    # exclusion reason ("Certificate epsilon 0.05 exceeds the local maximum 0").
    values.extend(
        float(number)
        for key, value in payload.items()
        if key not in _NUMBER_FREE_FIELDS
        for text in _text_leaves(value)
        for number in _TEXT_NUMBER.findall(text)
    )
    weights = {item.objective_id: item.weight for item in request.objective_weights}
    for skill in request.admitted_skills:
        values.extend(weights[name] * delta for name, delta in skill.objective_deltas.items())
    scores = [skill.score for skill in request.admitted_skills]
    values.extend(abs(a - b) for index, a in enumerate(scores) for b in scores[index + 1 :])
    values.append(1.0)
    return values


def _text_leaves(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _text_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _text_leaves(child)


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


def _spread_negation(text: str) -> str:
    """Repeat a list's negation before each item, so splitting at commas keeps it.

    "not real, observed, or measured data" becomes
    "not real, not observed, or not measured data".
    """

    def spread(match: re.Match[str]) -> str:
        word = match.group(1)
        items = re.split(r"(\s*,\s*(?:(?:or|and|nor)\s+)?|\s+(?:or|and|nor)\s+)", match.group(2))
        return word + " " + "".join(
            part if index % 2 else f"{'' if index == 0 else 'not '}{part}"
            for index, part in enumerate(items)
        )

    return _NEGATED_LIST.sub(spread, text)


def _synthetic_not_disclosed(request: RecommendationRequest, text: str) -> list[Finding]:
    if request.evidence_label != "SYNTHETIC" or "synthetic" in text:
        return []
    return [_finding("SYNTHETIC_NOT_DISCLOSED", "explanation must say that the evidence is synthetic")]


def _evidence_label_claims(request: RecommendationRequest, text: str) -> list[Finding]:
    if request.evidence_label != "SYNTHETIC":
        return []
    findings = []
    claims = sorted(
        {
            match.group(0)
            for sentence in _SENTENCE_BREAK.split(_spread_negation(text))
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
