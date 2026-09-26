# AEROSENTINEL 2.9.2: satellite-only aircraft awareness

The application collects optical, SAR and thermal satellite scenes and retains acquisition time, processing quality and cross-sensor uncertainty. Its optional foundation model workflow supplies scene representations when configured. These environmental products are context; they do not by themselves detect aircraft.

## Automatic high-resolution collection

See [AUTO_HIGHRES_SETUP.md](AUTO_HIGHRES_SETUP.md). The background SkySat archive collector uses the saved location and an entitled PL_API_KEY to download and crop georeferenced RGB imagery, then calls the optical detector automatically. Missing coverage is explicit.

## Built-in optical inference

The **Aircraft Awareness** page accepts a georeferenced, high-resolution, three-band RGB optical satellite GeoTIFF. On first use, it downloads the ITU Remote Sensing Laboratory's YOLOv8x experiment 12 transfer-learning checkpoint from `iturslab/Efficient-YOLO-RS-Airplane-Detection` at checkpoint revision `38fc6ae`. The checkpoint file is `transfer-learning/experiment-12/best.pt`. The model publisher reports evaluation on HRPlanes and CORS-ADD high-resolution optical imagery; these numbers are not validation for a new site. Model and image SHA-256 digests are recorded in the output. The checkpoint is fetched from the publisher at runtime and is not redistributed in this ZIP. Review its terms and the imagery license before use. See the [publisher's model card](https://huggingface.co/iturslab/Efficient-YOLO-RS-Airplane-Detection) and [research repository](https://github.com/RSandAI/Efficient-YOLO-RS-Airplane-Detection).

The app checks the raster CRS, native GSD (at most 2 m/pixel), three bands and actual acquisition timestamp. It reads overlapping tiles at native resolution, percentile-stretches each band for inference, applies the single-class airplane model, removes duplicate overlapping boxes, and converts box centers to WGS84 using the raster transform. This is **optical-only** inference. It cannot classify SAR or thermal pixels with an optical model. Tiles with insufficient valid pixels are skipped; a missing detection is `NO DETECTION / ABSENCE UNDETERMINED`.

## Fusion and temporal screening

You may also import externally produced georeferenced candidate rows using the CSV template. Fields: `site_id`, `scene_id`, `acquired_at` (ISO-8601 with timezone), `modality` (`optical`, `sar`, `thermal`, `foundation`), `latitude`, `longitude`, `confidence`, `gsd_m`, `object_length_m`, `model_id`; optional `registration_error_m`, `quality`, `cloud_fraction`. A site groups acquisitions; a scene identifies one acquisition. Use actual measurement provenance and a model validated for its sensor domain.

Nearby candidates in the same scene/time are fused. The screening score is a model score reduced by scene quality/cloud fraction, not a calibrated probability. A presence candidate requires optical or SAR evidence, at least three native pixels across the submitted object length, and a screening score of at least 0.5. Thermal and foundation outputs add context but cannot independently confirm aircraft. The published optical checkpoint does not automatically validate SAR, thermal or foundation classifications.

Across different acquisitions of a site, nearby candidates within 2 km may be marked as possible movement if displacement exceeds the combined positional uncertainty. Position change is an anomaly **candidate**. This is an association hypothesis: a different aircraft at the later location is an alternative explanation. Sparse revisits cannot establish identity, continuous route, speed or what happened between acquisitions. A negative detector output never establishes an empty sky.

## Evaluation before deployment

Benchmark on independently labeled imagery from the actual sensor and sites, including empty scenes, hard negatives, aircraft sizes, cloud/season variation and held-out locations. Measure precision, recall, false positives per square kilometer, calibration, stratified performance and geolocation error. Inspect original raster boxes and alternate explanations. This research system does not issue flight clearances or autonomous aviation decisions.
