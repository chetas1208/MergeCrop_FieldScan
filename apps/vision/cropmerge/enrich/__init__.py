"""Optional local-LLM prose enrichment for Inspection Area descriptions.

Off by default (CROP_MERGE_LLM_ENRICHMENT=false). Never fakes output: if the
model isn't enabled, isn't loaded, or generation fails, callers get None and
must keep showing the existing heuristic `reasons` text — never silently
replace real evidence with a placeholder.
"""
