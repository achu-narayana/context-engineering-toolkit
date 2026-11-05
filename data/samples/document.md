# Project Atlas: Q3 Platform Migration Review

## Background

Project Atlas is the internal effort to move the Northwind billing platform from the legacy monolith (codenamed "Ledger") to a set of containerised services running on Kubernetes. The programme was approved by the steering committee on 14 February 2025 with an initial budget of $1.8 million and a target completion date of 30 September 2025. The migration is led by Priya Raman (engineering) and Daniel Okafor (programme management), with Helena Fischer acting as executive sponsor.

As you may know, the legacy Ledger system has been in production since 2011 and handles roughly 4.2 million invoices per month across 38 countries. It is written mostly in Java 8 with a 1.1 TB Oracle database. Over the years, basically every team has added integrations to it, which is exactly why replacing it is so hard and honestly why previous attempts in 2019 and 2022 were abandoned.

## Goals

The goals of the programme have not changed since kickoff. First, reduce the monthly infrastructure cost of billing from $212,000 to under $140,000. Second, cut the median invoice processing latency from 3.4 seconds to under 800 ms. Third, make it possible to ship billing changes weekly instead of the current quarterly release train. Fourth, retire the Oracle licence, which costs $640,000 per year and renews on 1 December 2025.

It is worth noting that the fourth goal is the one with a hard external deadline. If the Oracle contract is not cancelled by 31 October 2025, the licence renews automatically for another three years.

## Progress to date

As of 29 August 2025, 11 of the 17 planned services are live in production. The invoice generation service, tax calculation service, currency conversion service and PDF rendering service have all been running for more than 60 days with no Sev-1 incidents. The payment reconciliation service went live on 18 August and has processed 1.9 million transactions with a mismatch rate of 0.003%, which is actually better than the legacy rate of 0.011%.

The remaining six services are: dunning (late payment reminders), credit notes, partner commission, revenue recognition, the reporting warehouse export, and the audit log. Of these, dunning and credit notes are in staging and expected to ship by 12 September. Partner commission is blocked on a data model decision described below. Revenue recognition and the reporting export are the most complex and are currently forecast for mid-October, which puts them after the original 30 September date.

Infrastructure cost for August was $168,000, down from $212,000 in January. Median latency for the migrated flows is 610 ms. For the flows still on Ledger it remains 3.4 seconds.

## Key decisions

The steering committee made three decisions in the 26 August meeting. One: the programme deadline is extended to 31 October 2025 to accommodate revenue recognition, but no further extension will be granted because of the Oracle renewal. Two: the partner commission service will use the new event-sourced data model rather than mirroring the legacy tables; Priya Raman estimated this adds two weeks but avoids a second migration later. Three: the team will hire two additional contract engineers for eight weeks at a cost of $96,000, to be funded from the programme contingency, which falls from $240,000 to $144,000.

The committee also agreed that the Oracle cancellation notice must be sent by Daniel Okafor no later than 15 October 2025, regardless of the state of the reporting export, because the export can fall back to a nightly file transfer for up to 90 days.

## Risks

The biggest risk is the revenue recognition service. It encodes 14 years of accounting rules, several of which are undocumented, and the finance team has only one person, Marta Lindqvist, who understands all of them. She is on leave from 22 September to 6 October. The mitigation is to complete her knowledge-transfer sessions by 19 September and to record them.

The second risk is performance of the reporting warehouse export under month-end load. The load test on 21 August reached 71% of expected peak before the export fell behind. The team believes partitioning the export by region will fix it, but this has not been proven yet.

The third risk is organisational: the support team has not been trained on the new runbooks, and 6 of the 9 runbooks are still drafts. Training is scheduled for the week of 22 September.

## Open questions

Should the audit log service be descoped from the migration and kept on a small Postgres instance attached to the legacy system until Q1 2026? This would remove roughly three weeks of work but would mean the Oracle database cannot be fully decommissioned until the audit log moves too, which may complicate the licence cancellation.

Who owns the decision on whether to pause feature work in the partner team during the final four weeks? Daniel Okafor raised this but no owner was assigned.

## Next steps

- Dunning and credit notes to production by 12 September (owner: Priya Raman).
- Knowledge-transfer sessions with Marta Lindqvist recorded by 19 September (owner: Daniel Okafor).
- Partitioned export load test by 26 September; result to be reported to the steering committee on 30 September.
- Oracle cancellation notice sent by 15 October (owner: Daniel Okafor).
- Full cutover and Ledger read-only mode by 31 October 2025.

For reference, the live dashboard is at https://atlas.northwind.internal/dashboard and the runbooks are in the repository under docs/runbooks/.
