# AEROSENTINEL 2.9.2 architecture

Satellite catalog search and per-scene optical/SAR/thermal processing feed quality and temporal scene analysis. Optional foundation embeddings are accepted by the research pipeline when actually generated. The Aircraft Awareness interface runs a published optical aircraft checkpoint on georeferenced high-resolution RGB imagery and can also accept outputs from other satellite aircraft detectors, gates resolution and quality, fuses coincident evidence, then screens nearby candidates between acquisition times. Human review follows all candidate outputs. See [SATELLITE_AWARENESS.md](SATELLITE_AWARENESS.md) for the input contract and limitations.

The existing environmental risk and airfield resilience screens are retained as ancillary research context; they do not assert aircraft presence. The optical checkpoint downloads on first use; no real-time stream or local validation set is bundled.
