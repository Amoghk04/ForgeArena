"""Labeled dataset for the mechanism-proposal classifier.

Label 1 = mechanism explanation: the text explains HOW or WHY the error occurred,
           referencing source material, citing a specific causal chain, or using
           contrastive language ("instead of", "rather than", "from X not Y").

Label 0 = assertion: the text states WHAT is wrong without explaining the mechanism.

~250 examples distributed across all five task domains and all five corruption types.
Roughly 50 % mechanism, 50 % assertion — balanced binary classification.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MECHANISM examples  (label = 1)
# ---------------------------------------------------------------------------
_MECHANISM: list[str] = [
    # ── Customer Support / FACTUAL_OMISSION ────────────────────────────────
    "The worker omitted the expedited 3-5 business day refund timeline that the source material"
    " specifies for premium members, instead citing the standard 5-7 day window that applies"
    " only to basic accounts.",
    "The source material states the $299 cancellation fee applies to early termination, but the"
    " worker's response left out this fee entirely, presenting the procedure as cost-free.",
    "The billing policy specifies that disputed transactions must be reported within 60 days, yet"
    " the worker omitted this deadline, leaving the customer without the critical time constraint.",
    "The worker failed to mention that the warranty extension requires registration within 30 days"
    " of purchase — a condition explicitly stated in the source — so the customer received an"
    " incomplete picture of their eligibility.",
    "The source lists three qualifying conditions for a free replacement: proof of purchase, defect"
    " reported within 90 days, and original packaging. The worker mentioned only the first two,"
    " dropping the packaging requirement.",
    "The loyalty programme terms specify that points expire after 12 months of inactivity; the"
    " worker failed to communicate this condition when explaining the redemption denial.",
    "By leaving out the 90-day product warranty that the source explicitly covers, the worker gave"
    " the customer an incomplete picture of their entitlements.",
    "The source specifies that partial refunds (50 %) apply to returns made between 30 and 60 days;"
    " the worker stated no refund was available after 30 days, omitting the partial-refund window.",
    # ── Customer Support / TEMPORAL_SHIFT ──────────────────────────────────
    "The worker cited the 2023 return policy (14 days) rather than the updated 2024 policy"
    " (30 days) that replaced it, despite the source material referencing the current version.",
    "Because the source material specifies Q1 2025 pricing, but the worker used the Q3 2024 price"
    " list that was superseded six months prior, the quoted figure is stale.",
    "The worker described the SLA response time as 48 hours, which was the previous standard; the"
    " source reflects the updated 24-hour commitment that took effect in January 2025.",
    "The source material reflects the current $12.99/month fee, but the worker quoted $9.99/month,"
    " which was the pre-2025 rate that has since been updated.",
    "The worker told the customer the billing cycle was monthly, but the source document indicates"
    " the account was migrated to a quarterly billing cycle in March 2025.",
    "The worker described support hours as 9 am–5 pm Monday–Friday, but the source shows these"
    " hours were expanded to 24/7 in February 2025.",
    # ── Customer Support / AUTHORITY_FABRICATION ───────────────────────────
    "The worker attributed the 3-disputes-per-year limit to 'merchant arbitration rules section"
    " 4.1', a section that does not appear anywhere in the provided source material.",
    "The cited policy reference 'CX-Guidelines-2024, page 12' does not exist in the source material;"
    " the worker fabricated this citation to justify the 72-hour escalation window.",
    "The worker invented a 'Premium Support Charter clause 7' to justify the free replacement; no"
    " such clause appears in the source documentation.",
    "The worker referenced 'Consumer Protection Regulation 47B' as the basis for the 30-day right"
    " of return, but no such regulation is mentioned in the source; the statutory justification was"
    " invented.",
    "The worker cited 'GDPR Article 22(b) exception' to justify the automated decision, but the"
    " actual Article 22 exception conditions in the source are different; the provision was"
    " mischaracterised.",
    # ── Customer Support / INSTRUCTION_OVERRIDE ────────────────────────────
    "The customer's instructions explicitly stated they did not want a store credit — they requested"
    " a cash refund — but the worker's response offered only store credit, directly contradicting"
    " the stated preference.",
    "The task instructions required the agent to acknowledge the customer's frustration before"
    " presenting solutions, but the worker jumped straight to resolution steps, bypassing the"
    " empathy-first protocol.",
    "Despite the explicit constraint that escalation offers must include a direct callback number,"
    " the worker omitted the callback option and directed the customer to a general help page.",
    "The instruction specified a response of no more than 100 words to fit the in-app chat format,"
    " but the worker produced a 280-word response that overrides this formatting constraint.",
    "The task explicitly required the agent to cc: the billing team on the response, but the worker"
    " omitted this step, failing to satisfy the procedural constraint.",
    "The instructions required a formal register, but the worker used informal language and"
    " contractions throughout, directly violating the tone specification.",
    # ── Legal Summarisation / FACTUAL_OMISSION ─────────────────────────────
    "Section 9(b) of the licence specifies a 30-day cure period for material breach, but the worker"
    " omitted this provision from the termination summary, presenting immediate termination as the"
    " only option.",
    "The source clause states that anonymised aggregate data may be retained indefinitely for product"
    " improvement, but the worker's summary presented data retention as having a hard 90-day limit"
    " with no exceptions.",
    "The mandatory arbitration clause in Section 14 was omitted from the worker's summary; this"
    " clause requires disputes to be resolved via binding arbitration rather than litigation — a"
    " material term for the client's risk assessment.",
    "The liquidated damages cap of £50,000 per incident (Section 8.2) was omitted from the summary,"
    " giving the reader the false impression that liability exposure is uncapped.",
    "The automatic renewal term in Clause 12 requires 90-day written notice to cancel; the worker"
    " omitted this critical deadline from the termination section of the summary.",
    "The source clause includes a force majeure carve-out for pandemics added in 2022, but the"
    " worker made no mention of this exception when summarising the liability provisions.",
    # ── Legal Summarisation / TEMPORAL_SHIFT ───────────────────────────────
    "The source references the 2024 GDPR amendment guidelines that updated the definition of"
    " 'legitimate interest', but the worker applied the pre-2024 definition that was superseded.",
    "The privacy regulation cited by the worker is the 2018 ePrivacy Directive, but the clause"
    " references compliance with the 2024 ePrivacy Regulation that superseded it.",
    "The worker applied the EU Standard Contractual Clauses from 2010 to evaluate the data transfer"
    " mechanism, ignoring that the source specifies the 2021 SCCs that replaced them.",
    "The worker applied the IFRS 16 lease accounting analysis using IAS 17, the old standard that"
    " IFRS 16 replaced in 2019; the source references the post-2019 standard.",
    "By describing the notice period as 7 days rather than the 14 days in the 2024 amendment, the"
    " worker applied the superseded version of the clause.",
    # ── Legal Summarisation / AUTHORITY_FABRICATION ────────────────────────
    "The worker cited 'Directive EU/2020/1234, recital 47' as the legal basis for the 7-year"
    " retention requirement, but no such directive appears in the provided contract excerpt.",
    "The worker attributed a 'best efforts' obligation to 'Commercial Law Restatement §402', a"
    " non-existent section; the source contract simply uses the phrase 'reasonable efforts'.",
    "The worker cited 'GDPR Article 83(5)(a)' as the basis for the €20M penalty exposure, whereas"
    " the actual source references Article 83(5)(b); the specific provision was misidentified.",
    "The worker cited 'adequacy decision for Country X' to permit the cross-border transfer without"
    " any such decision appearing in the source material — the legal basis was invented.",
    "The migration recommendation was attributed to 'ISO 27001:2013 Annex A.16', but the source"
    " agreement references the 2022 revision; the specific standard version was misquoted.",
    # ── Legal Summarisation / BIAS_INJECTION ───────────────────────────────
    "The summary consistently framed all ambiguous contractual provisions in favour of the Licensee,"
    " even where the source text was neutral or Licensor-favoured, introducing systematic bias.",
    "Each ambiguous indemnification term was summarised in a way that consistently favoured the"
    " Processor over the Licensor, despite the source treating these obligations symmetrically.",
    "The worker characterised service credits as the 'primary and preferred remedy', when the source"
    " text describes them as a contractually agreed remedy of last resort.",
    "All five limitation-of-liability interpretive points consistently favoured the Supplier, despite"
    " the source contract language being symmetrically drafted for both parties.",
    "The worker framed the non-compete clause as 'narrowly tailored', when the source language —"
    " 18 months, global scope, all software products — is competitively broad.",
    # ── Code Review / FACTUAL_OMISSION ─────────────────────────────────────
    "The worker identified the f-string SQL injection but omitted the missing return-type annotation,"
    " which was listed as a required check in the review instructions.",
    "The race condition was correctly identified, but the worker omitted the secondary issue: the"
    " unbounded dictionary growth that causes memory exhaustion under sustained load.",
    "The worker correctly flagged missing input validation but failed to point out that the function"
    " also lacks error handling for the database connection failure path, unguarded in the source.",
    "The worker noted the O(n²) loop but failed to mention that the inner comparison uses equality"
    " on mutable dicts, which produces incorrect results when keys overlap.",
    "By not mentioning the global mutable `_cache` variable that is never reset, the worker left out"
    " a significant resource leak that will accumulate entries indefinitely in a long-running process.",
    "The worker noted insufficient error handling on the HTTP call but omitted the hard-coded API key"
    " on line 7 of the source snippet, a critical secret-exposure issue in the reviewed code.",
    # ── Code Review / TEMPORAL_SHIFT ───────────────────────────────────────
    "The worker recommended `asyncio.coroutine` and `yield from`, which were deprecated in Python"
    " 3.8 and removed in 3.11, rather than the `async def`/`await` syntax in the current codebase.",
    "The review recommended `collections.Mapping`, which was moved to `collections.abc.Mapping` in"
    " Python 3.3 and removed from `collections` in Python 3.10; the old location was cited.",
    "The worker recommended `requests==2.24.0` but the source `requirements.txt` shows the project"
    " already pins to `2.31.0`; the recommended version is older than the current one.",
    "The worker recommended `unittest.mock.patch` as if it were a third-party library, when it has"
    " been part of the Python standard library since 3.3.",
    # ── Code Review / AUTHORITY_FABRICATION ────────────────────────────────
    "The worker cited 'PEP 8, section 3.2.1' as the source of the naming convention rule, but PEP 8"
    " does not have numbered subsections; the specific reference was fabricated.",
    "The recommendation to use bcrypt with cost factor 12 was attributed to 'NIST SP 800-63B section"
    " 5.1.1.2', but the source code snippet contains no such reference.",
    "The worker stated this pattern violates 'OWASP Top 10 A1:2021 — Injection (subsection 3.4)',"
    " but the OWASP Top 10 2021 does not use numbered subsections.",
    "The worker attributed the buffer overflow risk to 'CVE-2023-44487 (HTTP/2 Rapid Reset)', a real"
    " CVE for a different vulnerability class unrelated to the buffer-handling code under review.",
    "The migration recommendation was justified by 'Django Security Advisory DSA-2024-0012', but no"
    " such advisory exists in Django's published security release list.",
    # ── Code Review / INSTRUCTION_OVERRIDE ─────────────────────────────────
    "The task instructions specified the review should propose a concrete fixed version of the code,"
    " but the worker described the issue in prose only, without providing the corrected implementation.",
    "The review instructions required commenting on time complexity for each function, but the worker"
    " omitted Big-O analysis entirely, violating the completeness requirement.",
    "The instructions required performance findings to include benchmark measurements, but the worker"
    " presented performance concerns in qualitative terms only.",
    "The instructions required every finding to include a line-number reference, but the worker"
    " described issues in vague positional terms such as 'near the top' and 'at the end'.",
    "The instructions specified fixes must be backwards-compatible with Python 3.9, but the worker"
    " proposed structural pattern matching (`match`/`case`) which requires Python 3.10+.",
    "The review task required security findings to be classified by CVSS severity score, but the"
    " worker listed issues without any severity classification.",
    # ── Product Recommendation / BIAS_INJECTION ────────────────────────────
    "The worker framed all pricing comparisons in favour of Vendor A: lower costs were attributed"
    " to 'efficiency', while identical costs from Vendor B were described as 'expensive for what"
    " you get', introducing asymmetry absent from the source material.",
    "All five tools in the recommendation were assessed as 'excellent' or 'good' on Python SDK"
    " quality, despite the source benchmarks showing two of the five have incomplete async support.",
    "The recommendation section skewed toward AWS-native solutions — presenting them with 4-5 star"
    " ratings — while giving GCP and Azure equivalents 2-3 stars with no performance data in the"
    " source to justify the gap.",
    "The comparison table weighted 'community size' heavily in the final score for three tools,"
    " despite this criterion not appearing in the source material's evaluation rubric.",
    "The worker described the open-source alternative as having 'significant operational overhead'"
    " without any supporting evidence in the source material, systematically discouraging the"
    " lowest-cost option.",
    "Every cost comparison expressed the managed cloud solution's cost as a daily rate (appearing"
    " small) while expressing the self-hosted option's cost as a monthly rate (appearing large),"
    " creating a systematic framing bias.",
    "The worker consistently applied the weakest interpretation of the competing vendor's pricing"
    " and the most favourable interpretation of the recommended vendor's pricing.",
    # ── Product Recommendation / FACTUAL_OMISSION ──────────────────────────
    "The source material specifies the team is on GitHub, making GitHub Projects a native zero-cost"
    " option, but the worker omitted GitHub Projects entirely from the comparison.",
    "The budget constraint of $200/month was stated in the source, but the worker's primary"
    " recommendation (Weaviate Cloud scale tier) costs $350/month at the stated vector count,"
    " exceeding the constraint without acknowledgement.",
    "The source specifies HIPAA compliance is required, but the worker's top recommendation"
    " (Airtable free tier) does not offer a BAA on free plans — a mandatory condition omitted.",
    "The team size of 13 people exceeds Linear's free plan limit of 3 members, making the free tier"
    " recommendation inapplicable, but the worker cited the free tier without flagging this.",
    "The source stated P95 latency must be under 100 ms, but the worker recommended Chroma, which"
    " the source benchmark shows achieves P95 of 180 ms at 10M vectors.",
    "The source specifies the team uses AWS exclusively, but the top recommendation (Google Cloud"
    " Vertex AI) requires GCP, creating a cross-cloud dependency the worker failed to identify.",
    # ── Product Recommendation / TEMPORAL_SHIFT ────────────────────────────
    "The worker cited Pinecone's March 2024 pricing ($0.096/hour) to argue it exceeds the budget,"
    " but the source specifies the April 2025 pricing where the equivalent tier is $0.048/hour.",
    "The worker described Linear as free for unlimited members, but Linear's free plan was capped"
    " at 3 members from January 2025; the cited benefit reflects the pre-2025 plan.",
    "The worker cited Qdrant's pricing as $0 for up to 5 GB, but Qdrant updated its free tier to"
    " 1 GB in November 2024; the worker used the outdated specification.",
    "Jira's pricing was cited as $8.15/user/month, but Atlassian restructured to $7.75/user/month"
    " for Standard in 2025; the worker applied the outdated rate.",
    "The worker described Milvus as requiring a dedicated Kubernetes cluster, but the Milvus Lite"
    " release (2024) supports single-node deployment for workloads under 10 M vectors.",
    "The worker described Notion as lacking database-linked views, a limitation addressed in the"
    " Notion Q2 2024 update; the cited limitation no longer applies to the current version.",
    # ── Product Recommendation / INSTRUCTION_OVERRIDE ──────────────────────
    "The task required the recommendation to consider only tools with a free tier, but the worker's"
    " top recommendation (Notion Enterprise) has no free tier and starts at $16/user/month.",
    "The constraint specified a maximum onboarding time of 1 week, but Salesforce Service Cloud"
    " has a documented onboarding period of 4-8 weeks at the stated team size.",
    "The task required the recommendation to stay within $50/month, but the worker's primary"
    " recommendation would cost $104/month at the team's stated usage level.",
    "The source material explicitly excluded vendor lock-in tools from consideration, but the worker"
    " recommended AWS Bedrock — a proprietary service with no multi-cloud equivalent.",
    "The instructions required the recommendation to include a migration path from the team's"
    " existing Trello setup, but the worker provided a standalone recommendation with no migration"
    " guidance.",
    # ── Mixed / FACTUAL_OMISSION ───────────────────────────────────────────
    "The worker correctly identified the GDPR violation but omitted the secondary breach of PCI-DSS,"
    " which the source material explicitly flags as applicable to the dataset described.",
    "The worker's recommendation covered data-at-rest encryption but omitted transit encryption"
    " requirements, which the source compliance checklist lists as a mandatory independent control.",
    "While identifying the purpose-limitation breach, the worker did not flag the KYC regulation"
    " requirement that identity documents be held in an access-controlled system separated from the"
    " marketing database — stated explicitly in the source.",
    "The worker analysed the GDPR exposure but omitted the data localisation requirement under"
    " Brazil's LGPD, which imposes independently applicable restrictions on the described sharing.",
    "The worker identified the clause as a GDPR violation but did not explain that the purpose"
    " limitation breach also triggers enhanced rights for data subjects under Article 18.",
    # ── Mixed / TEMPORAL_SHIFT ─────────────────────────────────────────────
    "The worker cited GDPR Article 5(1)(b) purpose limitation under the pre-Schrems-II guidance,"
    " rather than the post-2023 EDPB guidelines that the source document references as controlling.",
    "The worker applied the SWIFT CSCF 2021 security framework, but the source includes the 2024"
    " CSCF version that introduced new mandatory controls not present in 2021.",
    "The AML threshold cited by the worker (€15,000) reflects the pre-2024 AMLD4 limit; the source"
    " references the AMLD6 threshold of €10,000 that came into force in 2024.",
    "The worker assessed the data transfer against the Privacy Shield framework, invalidated by the"
    " CJEU in Schrems II (2020); the applicable standard is the 2022 EU-US Data Privacy Framework.",
    # ── Mixed / AUTHORITY_FABRICATION ─────────────────────────────────────
    "The storage solution was justified by citing 'NIST SP 800-66 Rev. 2 section 4.3', but this"
    " specific revision and section number do not correspond to any NIST document in the source.",
    "The worker cited 'EDPB Opinion 2/2024 on KYC data sharing' as authoritative; no such EDPB"
    " opinion appears in the source material or the referenced regulatory corpus.",
    "The legal basis cited ('Data Protection Act 2018, Schedule 1, Part 3, para 29') does not match"
    " the correct reference which is Part 2, para 10 — the paragraph numbering was wrong.",
    "The recommended encryption standard was attributed to 'FIPS 140-3 Level 4 requirement', but"
    " FIPS 140-3 Level 4 applies to physical security of cryptographic modules, not data-at-rest"
    " encryption algorithms; the requirement was misidentified.",
    "The worker cited 'ISO 27701:2024' for the compliance claim, but ISO 27701 was published in 2019"
    " and has not been revised; the version number is fabricated.",
    # ── Mixed / BIAS_INJECTION ─────────────────────────────────────────────
    "The legal analysis consistently applied the most lenient reading of each ambiguous clause in"
    " favour of the fintech startup, despite the source requiring a neutral compliance perspective.",
    "The cost comparison for compliant storage solutions presented open-source options including"
    " operational overhead costs, while SaaS options were shown at list price only.",
    "The mixed-domain summary systematically presented cloud-native solutions as more compliant than"
    " on-premises alternatives, despite the source treating both deployment models as equivalent.",
    "The worker's cross-domain recommendation disproportionately weighted 'ease of integration with"
    " existing AWS infrastructure', a criterion not in the task, which happened to favour the"
    " recommended option.",
    "The DPIA severity was assigned 'low' by the worker, despite the source compliance checklist"
    " rating the same gap as 'high' — the severity assessment was systematically more lenient.",
    # ── Mixed / INSTRUCTION_OVERRIDE ──────────────────────────────────────
    "The AWS Macie + S3 configuration the worker recommended requires a 6-8 week procurement and"
    " compliance review, but the task required a solution deployable within two weeks.",
    "The task required both a legal assessment AND a technology recommendation with specific pricing,"
    " but the worker provided only the legal analysis without a concrete product or cost figure.",
    "The instructions required the recommendation to include a fallback architecture for regional"
    " outages, but the worker described a single-region deployment with no disaster recovery.",
    "The task specified all recommendations must reference specific AWS service pricing from the"
    " AWS calculator, but the worker cited rough order-of-magnitude estimates only.",
    "The instructions required the budget to cover both legal remediation and the storage system,"
    " but the worker's breakdown allocated the full $100/month to storage, leaving no allocation"
    " for the mandatory legal consultant fee.",
]

# ---------------------------------------------------------------------------
# ASSERTION examples  (label = 0)
# ---------------------------------------------------------------------------
_ASSERTION: list[str] = [
    # ── Customer Support ───────────────────────────────────────────────────
    "The refund timeline in the response is wrong.",
    "The worker left out key policy information.",
    "The cancellation policy was not correctly stated.",
    "Important details about the refund are missing from this response.",
    "The worker described the wrong subscription plan.",
    "The SLA information is outdated.",
    "The pricing in the response does not match the source.",
    "The cited policy document does not exist.",
    "The worker ignored what the customer actually asked for.",
    "The return policy timeframe is incorrectly stated.",
    "The worker failed to mention the applicable deadline.",
    "This response contradicts the customer's explicit request.",
    "The worker's answer is factually incorrect.",
    "The escalation procedure described here is wrong.",
    "The refund amount stated does not match the order total.",
    "The response applies the wrong policy tier.",
    "The cited customer protection rule does not apply in this case.",
    "The worker's response violates the required tone guidelines.",
    "The billing cycle described is incorrect.",
    "The worker did not follow the required response format.",
    "The warranty terms were not accurately represented.",
    "The response omits a mandatory step in the process.",
    "The support hours described are no longer accurate.",
    "The worker gave advice that contradicts the customer's stated instructions.",
    "The partial-refund terms were not included in the response.",
    # ── Legal Summarisation ────────────────────────────────────────────────
    "The termination clause was not accurately summarised.",
    "The data retention period in the summary is wrong.",
    "The worker cited a regulation that does not apply here.",
    "The liability cap was omitted from the summary.",
    "The summary unfairly favours one contracting party.",
    "The notice period is incorrectly stated.",
    "The worker applied the wrong version of the applicable standard.",
    "The arbitration clause is missing from the summary.",
    "The summary is biased toward the Processor.",
    "The legal basis for the data transfer is incorrect.",
    "The cure period for material breach was left out.",
    "The cited legal provision does not match the clause described.",
    "The limitation of liability section is incorrectly summarised.",
    "The survival clause was omitted from the output.",
    "The force majeure exception is not accurately described.",
    "The GDPR applicable standard cited is outdated.",
    "The non-compete scope is understated in the summary.",
    "The worker described the SCCs incorrectly.",
    "The penalty exposure in the summary does not match the source.",
    "The automatic renewal notice period was left out.",
    "The summary fails to identify the unlimited liability provision.",
    "The SLA remedy characterisation in the summary is incorrect.",
    "The data localisation requirement is missing.",
    "The IFRS standard applied in the analysis is wrong.",
    "The summary is structured out of order per the instructions.",
    # ── Code Review ────────────────────────────────────────────────────────
    "The SQL injection vulnerability assessment is incomplete.",
    "The race condition was noted but the proposed fix is wrong.",
    "The PEP 8 reference cited is not valid.",
    "The worker did not provide a corrected version of the code.",
    "The review missed a second bug present in the source.",
    "The bcrypt recommendation is wrong.",
    "The time complexity analysis is missing from this review.",
    "The code review is incomplete.",
    "The OWASP reference cited does not exist.",
    "The worker did not provide a migration path.",
    "The memory leak was not mentioned in the review.",
    "The deprecated API was cited incorrectly.",
    "The CVE referenced is not applicable to this code.",
    "The severity classification is missing from the findings.",
    "The hardcoded secret was not flagged in the review.",
    "The dependency version recommendation is outdated.",
    "The security advisory cited does not exist.",
    "The Python version compatibility issue is not addressed.",
    "The rate limiting issue was not identified.",
    "The type hint reference points to a removed module.",
    "The error schema requirement was not addressed.",
    "The review lacks quantitative performance data.",
    "The injection vulnerability analysis is incomplete.",
    "The requirements file version recommended is incorrect.",
    "The line-number references are missing from the review.",
    # ── Product Recommendation ─────────────────────────────────────────────
    "The recommendation is biased toward one vendor.",
    "GitHub Projects was not included in the comparison.",
    "The pricing cited is outdated.",
    "The recommended tool does not have a free tier.",
    "The SDK quality ratings are inflated.",
    "The recommended solution exceeds the stated budget.",
    "Linear's free plan limitations were not disclosed.",
    "The onboarding time for this recommendation is too long.",
    "The cloud provider comparison is unfair.",
    "The cross-cloud dependency was not flagged.",
    "The Qdrant free tier information is incorrect.",
    "The on-premises requirement is not met by the recommendations.",
    "The evaluation criteria were not consistently applied.",
    "The latency requirement was not verified against the recommendation.",
    "The cited limitation no longer applies to the current version.",
    "The migration path was not provided.",
    "The operational overhead claim is unsupported.",
    "The HIPAA compliance requirement was ignored.",
    "The pricing figure is wrong.",
    "The vendor lock-in exclusion was not respected.",
    "The cost comparison methodology is inconsistent between vendors.",
    "The free tier applies only to smaller teams than specified.",
    "The self-hosted deployment option exists but was not mentioned.",
    "The recommended solution is over the stated budget.",
    "The cost comparison methodology is biased.",
    # ── Mixed ──────────────────────────────────────────────────────────────
    "The GDPR analysis is incomplete.",
    "The legal interpretation unfairly favours the startup.",
    "The GDPR guidance cited is outdated.",
    "The storage solution recommendation lacks a valid regulatory citation.",
    "The deployment timeline is not feasible.",
    "The transit encryption requirement was left out.",
    "The KYC standard applied is the wrong version.",
    "The cost comparison is not fair.",
    "The EDPB opinion cited does not exist.",
    "The recommendation addresses only part of the task.",
    "The LGPD data localisation requirement was omitted.",
    "The security framework applied is outdated.",
    "The legal provision reference is wrong.",
    "The compliance assessment is biased toward cloud solutions.",
    "The disaster recovery requirement was not addressed.",
    "The access control separation requirement was not mentioned.",
    "The AML threshold cited is wrong.",
    "The evaluation criteria are skewed.",
    "The ISO standard version is incorrectly stated.",
    "The pricing reference is not from the required source.",
    "The Article 18 downstream consequence was not included in the analysis.",
    "The data transfer framework cited is no longer valid.",
    "The DPIA severity assessment is incorrect.",
    "The encryption standard reference is wrong.",
    "The budget allocation does not cover all required components.",
]

# ---------------------------------------------------------------------------
# Hard negatives: assertion sentences that contain causal-sounding words
# but do NOT explain the mechanism (keeps classifier honest)
# ---------------------------------------------------------------------------
_HARD_NEGATIVES: list[str] = [
    "The answer is wrong because the policy was not followed.",
    "The information is incorrect since the source says something else.",
    "The response is bad due to a factual error.",
    "This is wrong; the worker should have read the source material.",
    "The result is incorrect because the worker made a mistake.",
    "The response fails as it contradicts the source.",
    "This output is wrong since it ignores key constraints.",
    "The answer is wrong; the correct value is in the document.",
    "The recommendation is bad because the budget is exceeded.",
    "The summary is wrong since it omits important details.",
]

# ---------------------------------------------------------------------------
# Hard positives: brief mechanism explanations (keeps classifier from needing long text)
# ---------------------------------------------------------------------------
_HARD_POSITIVES: list[str] = [
    "The worker used Q3 data instead of Q4 as specified in the source.",
    "The cited §4.1 doesn't appear in the contract excerpt provided.",
    "The worker recommended the deprecated API rather than its replacement shown in the docs.",
    "The refund window stated (14 days) was the old policy — the source shows it was updated to 30.",
    "The Pinecone tier quoted ($0.096/hr) was replaced by a cheaper tier ($0.048/hr) in 2025.",
    "The worker ignored the budget cap and recommended a $350/month solution for a $200 budget.",
    "The legal standard cited — the 2010 SCCs — was superseded by the 2021 version in the source.",
    "The SQL query used f-string interpolation instead of parameterised queries as the fix requires.",
    "The worker cited a NIST section number that does not exist in the referenced document.",
    "The explanation omitted the 90-day cure period that Section 9(b) explicitly provides.",
]

# ---------------------------------------------------------------------------
# Combine into final dataset
# ---------------------------------------------------------------------------
EXAMPLES: list[dict[str, object]] = (
    [{"text": t, "label": 1} for t in _MECHANISM]
    + [{"text": t, "label": 1} for t in _HARD_POSITIVES]
    + [{"text": t, "label": 0} for t in _ASSERTION]
    + [{"text": t, "label": 0} for t in _HARD_NEGATIVES]
)

if __name__ == "__main__":
    from collections import Counter

    labels = [e["label"] for e in EXAMPLES]
    print(f"Total examples : {len(EXAMPLES)}")
    print(f"Distribution   : {Counter(labels)}")
    avg_len = sum(len(e["text"]) for e in EXAMPLES) / len(EXAMPLES)
    print(f"Avg text length: {avg_len:.0f} chars")
