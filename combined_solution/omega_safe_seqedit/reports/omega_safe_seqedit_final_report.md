# Precision-first long-read error correction with support-conditioned local edit evidence

**Shane Jayasundera and Eesha Kurella**  
Department of Computer Science, University of Maryland  
CMSC701 Final Project Report

## Abstract

Long-read sequencing enables assembly, phasing, and structural-variant discovery across repetitive genomic regions, but Oxford Nanopore Technologies (ONT) reads remain error-prone enough that read-level error correction can substantially affect downstream analyses. The central challenge is not simply to maximize the number of edits. In diploid and repeat-rich regions, aggressive correction can erase true haplotypes or repeat-copy differences and thereby create biological artifacts. We developed **Omega Safe SeqEdit**, a compact research codebase for conservative long-read correction as support-conditioned edit prediction. The system predicts local edit actions aligned to a target read, uses support-read pileup and rule features, and evaluates not only sequence identity but also overcorrection, hard-edit false positives, and edit-type precision/recall. Starting from an interim transformer-based sequence-to-sequence correction model, we rebuilt the pipeline around explicit COPY/SUB/INS/DEL decisions, support-rule baselines, hybrid conservative decoding, real-data abstention, and candidate-level forensic diagnostics. On small synthetic benchmarks, support-conditioned hybrid decoding corrected most designed edits with zero false edits. On a curated noisy synthetic benchmark, the final hybrid setting corrected 11 edits with zero false edits, matching conservative consensus safety while improving over no-edit. On HG002 chr20 real-data windows, however, naive support-rule and neural correction produced many false positives: support-rule corrected 1001 edits but introduced 1528 false edits, and neural-only decoding introduced 6255 false edits. Strict hybrid decoding avoided all false edits but collapsed to no-edit behavior. A ranked-substitution experiment demonstrated that one apparently safe candidate can be recovered only when the allowlist is verified; newer audited runs showed the top-ranked candidate was false, motivating a shift from per-site pileup scoring to local-window, whole-read evidence. These results show that safe long-read correction requires local haplotype- and repeat-aware evidence before scaling model capacity.

**Keywords:** long-read sequencing; error correction; Oxford Nanopore; overcorrection; support reads; haplotype-aware correction

## Introduction

Long-read sequencing has changed genomics by making it possible to observe long-range structure, repeats, and haplotype phase directly in individual reads. These advantages are especially important for human genome assembly and medically relevant structural variation. However, long reads can contain substitutions, insertions, and deletions at rates high enough to complicate downstream analysis. Error correction attempts to transform each noisy read into a more accurate representation of the molecule that was sequenced.

The problem is deceptively close to sequence-to-sequence translation: given a noisy DNA string, output a corrected DNA string. Our interim project began from that framing, using a transformer-style architecture that could attend to overlapping support reads. This is analogous to correcting a misspelled sentence by using language context, except the alphabet is {A, C, G, T} and the “grammar” is determined by local read evidence and genome biology. The interim system established an end-to-end pipeline for ONT data, window generation, target/support/truth examples, edit-token training, and baseline comparison. It also revealed a critical failure mode: naive support attention can be worse than target-only baselines because support reads are not uniformly trustworthy.

That observation reframed the final project. In long-read correction, false-positive hard edits are often worse than missed edits. A missed correction leaves an existing sequencing error, but a false correction can overwrite real biological variation. This is particularly dangerous in heterozygous, repeat-rich, or locally ambiguous alignments where support reads may come from different haplotypes, shifted indel alignments, or paralogous/repetitive loci. Recent systems motivate this caution. DeepConsensus demonstrates the value of gap-aware alignment modeling for read correction; HERRO and DeChat emphasize haplotype- and repeat-aware correction; hifieval formalizes evaluation in terms of correct corrections, undercorrections, and overcorrections rather than identity alone.

The final project therefore asks a narrower research question: **Can a support-conditioned, deletion-aware, precision-first edit model beat conservative consensus on small long-read benchmarks without increasing overcorrection?** The answer from our current implementation is mixed. The approach is successful as a debugging and research framework and on controlled synthetic tasks, but the real HG002 benchmark shows that the present evidence representation is still insufficient for safe nontrivial correction. This negative result is useful: it identifies the exact granularity at which the next method must operate, namely local correction hypotheses supported by whole reads rather than isolated pileup columns.

## Methods

### Data representation and benchmarks

We implemented three benchmark levels. The smallest, `mac_debug`, is a synthetic smoke test containing designed substitutions, insertions, deletions, homopolymer cases, and copy-only controls. A noisier synthetic slice adds shallow support, false-support cases, neighboring edits, homopolymer deletions, and support-rule failures. The real benchmark uses HG002 chr20 windows derived from aligned ONT reads with reference, truth VCF, and confident-region annotations when available. Each example is stored as inspectable JSONL with a target sequence, support-aligned sequences, truth sequence, edit labels, support pileup features, repeat/homopolymer flags, and optional variant/confidence masks.

The main real-data preset in the combined codebase is `hg002_old_12x_mac`, which uses 12x HG002 chr20-derived windows and evaluates 200 test examples. This preset is intentionally Mac-runnable and small enough to support rapid iteration. It is not a full SOTA-scale benchmark; rather, it is an early real-data gate to determine whether the method can safely beat no-edit and conservative consensus before scaling.

### Model architecture

Omega Safe SeqEdit uses explicit edit prediction rather than free-form generation. The target read is encoded with token embeddings and a small transformer encoder. Support evidence is represented as per-position features: base counts, insertion/deletion support, depth, agreement, entropy, strand and repeat/context indicators, and support-rule features. The model predicts factorized edit outputs: a main edit type head for COPY/SUB/DEL, a substitution payload head, an insertion payload head, and auxiliary allow/confidence outputs. This factorization makes it possible to diagnose whether errors arise from edit-family selection, payload selection, or decoding.

The original architecture was larger and more seq2seq-like, but the final system deliberately deemphasizes open-ended generation. In practice, the safest behavior came from hybrid decoding: support rules propose candidate edits, neural outputs provide payload and confidence information, and a conservative decoder abstains unless evidence passes strict filters. Trust gating was demoted from a core fusion bottleneck to a diagnostic/abstention concept because early experiments showed that a weak or flat gate could suppress useful edit learning.

### Conservative decoding and candidate evidence

The decoder is precision-first. Rule-negative positions veto neural hard edits by default. For insertions and deletions, real-data policies require strong support, non-repeat context, and allow-gate approval. Substitutions are particularly risky on HG002, so support-rule substitution is treated as a candidate rather than a forced edit. Candidate recovery is evaluated through explicit allowlists whose SHA-256 hashes and true/false counts are asserted against the evaluated summaries to prevent stale results.

The most recent implementation adds a candidate-evidence module that computes local-window evidence for every support-rule SUB candidate. For each candidate, it exports the target window, candidate-corrected window, support-read snippets, COPY-vs-SUB window scores, counts of reads better explained by COPY or SUB, ambiguous-read counts, cluster margin, flank consistency, strand balance, mapping-quality summaries, nearby indel/mismatch density, repeat flags, and variant-context features. This is the methodological transition from per-site scoring to local hypothesis scoring.

### Evaluation metrics

We report identity and normalized edit distance, but use them as secondary metrics. The primary safety metrics are hard-edit false-positive rate, overcorrection rate, total false edits, corrected edits, and a usable score:

`usable_score = identity - 0.5 * overcorrection - 0.5 * hard_edit_false_positive_rate`.

This score intentionally penalizes overcorrection and false hard edits. We also report edit-type precision/recall for substitutions, insertions, and deletions; per-base recall for SUB and INS payloads; support-rule false-positive counts; and diagnostic tables for false edits and missed true edits. This evaluation design is directly motivated by the interim finding that models can appear strong by identity while making biologically unsafe edits.

## Results

### Synthetic benchmarks confirm end-to-end learnability

On `mac_debug`, no-edit achieved identity 0.9897 with zero corrected edits. The conservative consensus baseline improved identity to 0.9966 and corrected 0.667 edits per example with zero false edits. The support-rule baseline performed best on this clean synthetic setting, reaching identity 0.9996, corrected edits 0.958 per example, hard-edit precision 0.917, and zero false edits. The learned full hybrid model reached identity 0.9987 and corrected 0.875 edits per example with zero false edits. Neural-only decoding collapsed to no-edit behavior. This shows that the full pipeline can learn and apply designed edits, but also that deterministic support evidence is very strong when synthetic examples are clean and well aligned.

The curated noisy synthetic benchmark was a more useful safety gate. No-edit reached usable score 0.9922. Support-rule improved identity but introduced five false edits, reducing usable score to 0.9121. The final conservative hybrid model matched consensus with usable score 0.9970, corrected 11 total edits across 24 examples, and introduced zero false edits. This is the intended behavior: support is used only when it is safe, and noisy support-rule false positives are vetoed.

### Real HG002 exposes the support ambiguity bottleneck

The HG002 chr20 benchmark changes the picture. No-edit achieved usable score 0.9663, identity 0.9663, and zero false edits. Conservative consensus improved identity to 0.9682 and corrected 960 true edits, but introduced 1278 false edits, yielding a much lower usable score of 0.7773. The support-rule baseline corrected 1001 true edits but introduced 1528 false edits, reducing usable score further to 0.7282. Neural-only decoding corrected 959 true edits but introduced 6255 false edits and scored only 0.5447. Strict hybrid decoding avoided all false edits but corrected zero edits, matching no-edit exactly.

These results show that raw support majority is not enough on real data. The evidence at a single pileup column can be extremely confident while still representing a haplotype switch, repeat-copy artifact, shifted indel alignment, or low-confidence truth region. The model also learned edit priors that were unsafe when decoded without rule-negative vetoes. In other words, the limiting factor is not merely model capacity; it is evidence calibration.

### Ranked substitution recovery is fragile

We attempted to recover a tiny number of ultra-safe substitutions from the strict no-recovery policy. Early unaudited outputs suggested that `ranked_sub_top_1` recovered one true substitution with zero false edits. After adding allowlist hashing and count assertions, the audited run showed that the current top-ranked candidate was false: one allowlisted candidate, zero true candidates, and one false candidate. This result invalidates the earlier top-1 claim and demonstrates why stale-output protection is essential.

The subsequent forensic tooling compares top candidates by the older scalar score, by local-window score, by false-positive history, and by vetoed true positives. The emerging pattern is that many false SUB candidates look strong by support fraction, entropy, payload confidence, and even local-window pileup delta. Therefore, safe recovery likely requires whole-read clustering and haplotype-aware local hypothesis scoring. The new `candidate_evidence.py` module is designed to expose exactly these features for the next iteration.

## Discussion

The main scientific result is that conservative local edit prediction is a better framing than unconstrained seq2seq correction, but the current evidence representation is still too local for real human ONT correction. Synthetic data validated the codebase and showed that the architecture can learn edit operations. However, real HG002 results show that support reads cannot be treated as independent votes over a pileup column. In regions with repeats, heterozygosity, local indels, or mapping uncertainty, majority support can point toward an edit that is unsafe relative to the target molecule and truth annotation.

This explains why larger models and lower thresholds did not solve the problem. Increasing model capacity produced more edits but not safer edits. Lowering thresholds recovered recall on synthetic cases but admitted false positives on real HG002. Pairwise ranking over scalar features also failed because the features themselves were not discriminative enough: false candidates often had high support fraction, low entropy, and high payload confidence. The next unit of evidence must be a local correction hypothesis: does the corrected local sequence explain whole support reads better than COPY, and is that support consistent with strand, mapping quality, repeat status, and known variant context?

The final codebase is therefore best understood as a research instrument. It provides a small, inspectable architecture; reproducible synthetic and real-data presets; precision-first metrics; explicit safety gates; and forensic candidate reports. It also contains an important negative result: real-data safe correction has not yet surpassed no-edit except under untrusted or fragile ranked outputs. That result is scientifically meaningful because it prevents overstating identity improvements that come from unsafe edits.

Limitations remain substantial. The HG002 benchmark is small and not sufficient for SOTA claims. External tools such as HERRO and DeChat were not run end-to-end on the same data in the final evaluation. The truth construction depends on reference, VCF, and confident-region assumptions. Candidate evidence currently uses aligned support snippets rather than a full haplotype-aware local realignment. Finally, the model is not yet calibrated well enough to use neural-only predictions safely on real data.

## Conclusion

Omega Safe SeqEdit advances the project from a naive transformer seq2seq correction model to a precision-first, support-conditioned edit-correction framework. It demonstrates end-to-end learnability on synthetic benchmarks and exposes why real ONT correction is harder: the dominant problem is not finding edits, but identifying which locally plausible edits are biologically safe. The most important next step is to rank and decode candidate corrections using whole-read local-window evidence, haplotype/variant context, and repeat-aware abstention. In its current form, the method should not be claimed as SOTA-competitive, but it provides a principled and editable foundation for reaching that goal.

## Data and code availability

All code, configuration files, notebooks, and saved summaries used in this report are located in the local `LRS-Error-Correction` repository. The primary final implementation is `combined_solution/omega_safe_seqedit`; the earlier rebuild is `new_solution/omega_lr_rebuild`; and the original prototype is `old_solution`.

## References

1. Baid, G. et al. DeepConsensus improves the accuracy of sequences with a gap-aware sequence transformer. *Nature Biotechnology* 41, 232–238 (2023). https://www.nature.com/articles/s41587-022-01435-7  
2. Oxford Nanopore Technologies. HERRO: haplotype-aware error correction of ultra-long nanopore reads. https://nanoporetech.com/resource-centre/herro-haplotype-aware-error-correction-of-ultra-long-nanopore-reads  
3. Liu, Q. et al. Repeat and haplotype aware error correction in nanopore sequencing reads with DeChat. *Communications Biology* (2024). https://www.nature.com/articles/s42003-024-07376-y  
4. hifieval: Evaluation of haplotype-aware long-read error correction. *Bioinformatics* 39, btad631 (2023). https://academic.oup.com/bioinformatics/article/39/10/btad631/7321114  
5. Salmela, L. and Rivals, E. LoRDEC: accurate and efficient long read error correction. *Bioinformatics* 30, 3506–3514 (2014). https://doi.org/10.1093/bioinformatics/btu538

