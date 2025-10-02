# Software Design Document (SDD) v4.1
## Image Segmentation Platform - Technical Specification

**Version**: 4.1  
**Last Updated**: 2025-01-20  
**Status**: Production-Ready  
**Audience**: Engineers, Architects, Contributors

---

## ⚠️ IMPORTANT: Choose the Right Document

**This SDD is the technical specification for builders, not users.**

### Are you a **USER** trying to segment images?
➡️ **START HERE**: [README_STREAMLINED.md](README_STREAMLINED.md) (2 minutes)  
➡️ **DETAILED GUIDE**: [GOLDEN_PATH.md](GOLDEN_PATH.md) (15 minutes)  
➡️ **MAKING DECISIONS**: [DECISION_GUIDES/](DECISION_GUIDES/) (5 min each)

### Are you an **ENGINEER** implementing features?
➡️ **START HERE**: [FOR_ENGINEERS.md](FOR_ENGINEERS.md) (10 minutes)  
➡️ **THEN READ**: This SDD (Sections 1-5, then relevant sections as needed)  
➡️ **CONTRACTS**: [CONTRACTS/](CONTRACTS/) (specific interfaces)

### Are you an **ARCHITECT/REVIEWER**?
➡️ **READ**: This entire SDD (full specification)  
➡️ **FOCUS ON**: Sections 4 (Contracts), 6 (Policies), 13 (Cross-Validation)

---

## Document Purpose

This SDD specifies:
1. **What** the platform does (scope) and doesn't do (out of scope)
2. **Stable contracts** that must not break (Section 4)
3. **Policies** for fallbacks, testing, validation (Sections 5-9)
4. **Patterns** for extension and contribution (Section 4.6, Appendix A)

This SDD does NOT:
- ❌ Teach you how to use the platform (see GOLDEN_PATH.md)
- ❌ Help you choose parameters (see DECISION_GUIDES/)
- ❌ Provide API documentation (see docstrings)
- ❌ Explain deep learning concepts (external resources)

---

## Version History

### v4.1 (Current) - 2025-01-20
**Changes**:
- Added factory patterns for models and checkpoints
- Centralized construction to eliminate redundancy
- Added experiment identity tracking
- Documented loss function exceptions

**Key Additions**:
- Section 4.6: Construction Factories
- Appendix A: Critical Refinements
- Tests for import violations

### v4.0 - 2025-01-15
- Added Golden Path concept
- Three-tier testing strategy
- Effective settings logging
- Auto-tuning safety gates

### v3.x - Earlier
- Initial architecture
- Basic contracts
- Configuration system

---

## Quick Navigation for Engineers

| I need to... | Read Section... | Time |
|--------------|----------------|------|
| Understand scope and goals | 1-2 | 10 min |
| Know what can't change | 4.1-4.5 | 15 min |
| Add a model architecture | 4.6, then factory code | 10 min |
| Add a loss function | 4.3, then losses/ code | 10 min |
| Write tests | 9 | 15 min |
| Handle dependencies | 6 | 10 min |
| Understand validation modes | 5 | 10 min |

**Total for typical feature work**: ~1 hour of SDD reading

---

## Critical Rules (From Painful Experience)

These rules exist because we got burned:

1. **No `import segmentation_models_pytorch` in scripts/**
   - Pain: Scripts and library code diverged
   - Fix: Factory pattern (Section 4.6)

2. **No hand-rolled model architectures**
   - Pain: Bugs, inconsistency, maintenance burden
   - Fix: Use SMP exclusively (except losses - see below)

3. **Loss functions are exception to "no custom implementations"**
   - Why: SMP doesn't provide losses, they're math formulas not architectures
   - Location: `src/losses/` with tests

4. **One source for checkpoints**
   - Pain: Different checkpoint configs in different places
   - Fix: `src/core/checkpoints.build_checkpoint_callback()`

5. **Fallbacks are test-only**
   - Pain: Accidentally deployed without full dependencies
   - Fix: Validation modes and guards (Section 6)

6. **Contract interfaces are frozen**
   - Pain: Breaking changes broke user code
   - Fix: Section 4 contracts stable through v5.0

---

## Reading Guide by Role

### Junior Engineer (Implementing Well-Defined Task)
1. Read FOR_ENGINEERS.md
2. Skim Section 4 (contracts)
3. Deep-read the section relevant to your task
4. Follow existing code patterns

### Senior Engineer (Designing Features)
1. Read Sections 1-5 (purpose, scope, architecture, policies)
2. Read Section 9 (testing strategy)
3. Read relevant advanced sections
4. Consider impact on contracts

### Architect / Tech Lead
1. Read entire SDD
2. Focus on Sections 4 (contracts), 13 (future extensions)
3. Review Appendix A (refinements)
4. Ensure new features align with philosophy

### New Contributor
1. Start with GOLDEN_PATH.md (understand what users do)
2. Read FOR_ENGINEERS.md (understand builder perspective)
3. Read SDD Sections 1-4 (understand architecture)
4. Pick a small issue and follow patterns

---

## Philosophy: Simple for Users, Structured for Builders

**User Perspective**: "I just want to segment images"
- Give them: README → Config → Train → Results
- Hide: Validation modes, fallbacks, factories, policies

**Engineer Perspective**: "I need to add a feature without breaking things"
- Give them: Stable contracts, clear patterns, comprehensive tests
- Document: What can change, what can't, how to extend

**This SDD enables the engineer perspective.**  
**Other docs enable the user perspective.**

---

## Scope Reminder (From Section 2)

### ✅ IN SCOPE
- Binary image segmentation (one target class)
- Training on local PNG/JPEG images
- U-Net family models from SMP
- Common loss functions (Dice, Focal, Tversky)
- Single GPU or Apple Silicon (MPS)
- Configuration-driven workflows

### ❌ OUT OF SCOPE (Not v4.1)
- Multi-class segmentation (future)
- Object detection, classification
- Custom architecture implementation
- Real-time inference optimization
- Multi-GPU training
- Cloud deployment automation
- Web UI / REST API

If your task is out of scope, discuss scope change before implementing.

---

## The Rest of This Document

Below this header, the SDD continues with full technical details:
- **Section 1**: Executive Summary (why this exists)
- **Section 2**: Purpose & Scope (what we build)
- **Section 3**: Tech Stack (dependencies)
- **Section 4**: Architecture Contracts ⭐ CRITICAL
- **Section 5**: Configuration Policy
- **Section 6**: Fallback Policy
- **Section 7**: Effective Settings Logging
- **Section 8**: Auto-Tuning (opt-in)
- **Section 9**: Testing Strategy ⭐ IMPORTANT
- **Section 10**: Repository Structure
- **Section 11-15**: Implementation guides
- **Appendix A**: Critical Refinements

**For most engineering tasks, you'll read Sections 1-5, then reference others as needed.**

---

[Original SDD content continues below...]

---
