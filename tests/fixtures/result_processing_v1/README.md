# Exact result-processing source admission

This candidate delta admits only SimResult.to_dict's one-operation, hook-free
full-history detailed projection. It is a private review candidate, not a
release approval or native-runtime qualification. The immutable historical
API, result identity and spring provenance fixtures remain unchanged.

The test guard pins this delta separately, verifies its exact prior and result
SHA256, and compares complete approved methods. Mutation refusal, source
checkout byte preservation, all state/result validators and byte limits remain
in force. No atomic-save, solver, physics, Graph or other backend rewrite.
