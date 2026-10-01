"""Baselines for the WaveLM-vs-Transformer comparison (E1/E2).

Contains a from-scratch byte-level decoder-only Transformer (`byte_gpt.ByteGPT`)
used as the head-to-head and scaling-law baseline. Kept separate from the `wave/`
package so the comparison is clearly an external baseline, not part of the model.
"""
