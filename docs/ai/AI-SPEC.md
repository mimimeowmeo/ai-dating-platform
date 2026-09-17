# AI Specification — MVP Direction

## AI subsystems

1. Face verification
2. Image embedding
3. Structured profile feature extraction
4. Text embedding
5. Conversation feature extraction
6. User feature aggregation
7. Clustering
8. Candidate ranking
9. AI evaluation

## Face Verification

Pipeline:

```text
Selfie
 -> image validation
 -> face detection
 -> liveness / anti-spoofing
 -> face embedding
 -> identity verification
 -> policy decision
```

Do not treat detection as identity proof.

## User Feature Composition

```text
Basic
+ Preferences
+ Interests
+ Hobbies
+ Foods
+ Image features
+ Behavior
+ Conversation features
+ Geographic context
```

Persist model and feature versions.

## Recommendation

```text
Hard filter
 -> candidate generation
 -> vector similarity
 -> behavioral features
 -> ranking
 -> safety / diversity rules
 -> final candidates
```

## Conversation Analysis

Never overwrite raw messages.

Store AI-derived features separately with:
- source
- confidence
- model_version
- feature_version
- timestamp

## Model Governance

Any change to:
- model
- prompt
- embedding dimensions
- ranking formula
- verification threshold
- feature schema

requires appropriate tests and version updates. Threshold/policy changes require human approval.
