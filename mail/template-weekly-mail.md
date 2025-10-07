---
status: draft         # later: sent
from: "{{ from_iso }}"   # e.g., 2025-09-30
to:   "{{ to_iso }}"     # e.g., 2025-10-07
action_needed: {{ action_needed }}   # true/false
subject: "Weekly Report — Jörn Stöhler — {{ from_iso }}–{{ to_iso }}{{ action_tag }}"
---

# Executive Summary
- {{ bullet1 }}
- {{ bullet2 }}
- {{ bullet3 }}
- {{ bullet4 }}
- {{ bullet5 }}

{{#if action_needed}}
## Actions for Kai
- {{ explicit_question_1 }}
- {{ explicit_decision_2 }}
{{/if}}

## Weekly Update
### {{ takeaway_heading_1 }}
{{ one_paragraph_takeaway_1 }}  
Evidence: {{ link_1 }}

### {{ takeaway_heading_2 }}
{{ one_paragraph_takeaway_2 }}  
Evidence: {{ link_2 }}

## Plan Adjustments
- {{ adjustment_1 }}
- {{ adjustment_2 }}

## Next Up
- {{ next_item_1 }}
- {{ next_item_2 }}

## Appendix (links)
- {{ label }} — {{ url_or_path }}
