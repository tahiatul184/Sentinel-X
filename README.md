# Sentinal X 1.0.0

Satellite-only aviation research application, upgraded from AEROSENTINEL 2.9.2 while retaining the 2.9.1 aircraft detector and automatic high-resolution imagery collector.

Run `python start_dashboard.py` (Windows: `RUN_ALL.bat`), then open the displayed local dashboard URL. Existing setup instructions remain in `START_HERE.txt` and `AUTO_HIGHRES_SETUP.md`. Automatic SkySat collection requires your entitled Planet API key and a saved location; it does not purchase imagery or provide live satellite coverage. No location or API credential is bundled.

## Try it yourself

This repository is public. Each tester runs an independent copy with their own data and credentials.

### In your browser (GitHub account required)

[Open in GitHub Codespaces](https://codespaces.new/tahiatul184/Sentinel-X)

Create a codespace and wait for dependency setup. In its terminal run:

```sh
python -m streamlit run app/dashboard.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Open port **8501** from the **Ports** tab. This starts the dashboard only; automatic background collection is not running. Try **Research & Validation** with the bundled synthetic examples. The lightweight setup excludes pretrained detector weights and optional foundation models. Codespaces availability and usage charges depend on your GitHub account; stop the codespace when finished. Keep its forwarded port private because the app stores shared settings and has no multi-user authentication.

### On your computer (Python 3.12)

```sh
git clone https://github.com/tahiatul184/Sentinel-X.git
cd Sentinel-X
python start_dashboard.py
```

The launcher installs the full dashboard dependencies and starts background workers. Open http://127.0.0.1:8501. Windows users can also double-click `RUN_ALL.bat`. First setup needs internet access and may take several minutes. Provider imagery requires your own access; model downloads may be large.

### Run the complete test suite

In a virtual environment, or the configured codespace:

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s app -v
```

[Test results on GitHub Actions](https://github.com/tahiatul184/Sentinel-X/actions/workflows/tests.yml)

The suite uses synthetic inputs and mocked providers. It does not establish real-world aircraft detection accuracy or validate live imagery entitlements.

## Aircraft Map dashboard

**Aircraft Map** is the default landing page. When the manual detector, imported satellite observations, or automatic SkySat worker saves an assessment, its presence candidates appear here. The page reads the latest saved assessment every 10 seconds while open.

- Click an aircraft marker or choose an observation ID to inspect its coordinates, capture time, site, scene, screening score, model confidence (new assessments), model provenance, resolution, and validation status.
- Filter by site, minimum screening score, and latest candidate capture per site or all candidate captures in the assessment.
- Download the selected observation as JSON or the filtered table as CSV.
- Turn on **Preview synthetic example** to try the map without imagery or credentials. Its invented points are labeled synthetic and are never saved to your evidence.
- Real detections can be generated on **Aircraft Awareness**, then viewed using **Open Aircraft Map**.

This is a map of satellite observations, not a live flight feed. A marker is an imagery-derived aircraft candidate at its acquisition time. Repeated observations are not unique aircraft; older candidates may remain in the history after a newer scene has no detections. Flight registration, speed, heading, altitude, and transponder status are unknown. Scores are not calibrated probabilities. The map shows the latest saved assessment, so an automatic assessment can replace a manual/imported assessment as the displayed source; check the source label and capture times.

Map tiles need internet access. The table and exports remain available if the basemap is unavailable. No tracking-service API key is required.

## What changed

- Sentinal X dashboard branding and **Research & Validation** page.
- Direct optical/SAR evidence determines presence candidates; thermal/foundation context cannot inflate them. Exact duplicate imports are deduplicated.
- Documented native GSD constrains enhanced-product screening; missing native resolution is labeled unverified. Detector product pixel spacing is no longer mislabeled as native resolution.
- Adjacent-epoch reciprocal temporal matching, ambiguity abstention, 72-hour gap cap and explicit registration-error gates. Missing uncertainty cannot produce a movement anomaly.
- Offline object-detection evaluator with grouped leakage checks, equal test footprints across methods, per-domain results and calibration diagnostics. User annotations and honest group definitions remain necessary.
- Checkpoint-, band-, footprint- and alignment-constrained foundation feature-pair comparison. This is contextual change, not an aircraft head.
- Twenty-paper primary-source traceability and a Bangladesh validation protocol in `research/RESEARCH_TRACEABILITY.md`.

## Use the validation workflow

Open Research & Validation and download the example JSON. Examples are explicitly synthetic: replace them with actual reviewed imagery annotations and model predictions. Boxes are `[x_min, y_min, x_max, y_max]` in each original scene's shared pixel grid. Ground truth covers the declared reviewed area. Every method is evaluated on the same test manifest, so omitted predictions count as misses. Use independent calibration data to select the threshold before opening test results.

The foundation example accepts `before`/`after` embeddings with matching model/checkpoint/bands/site/footprint, acquisition timestamps, verified alignment and valid coverage. It rejects incompatible vectors rather than averaging unrelated latent spaces.

## Validation status

138 tests passed on Python 3.12 on 2026-09-26, including Streamlit AppTest rendering of all nine dashboard pages, synthetic GeoTIFF processing, collection/transfer logic and validation accounting. Python compilation also passed. Two stale test fixtures were updated for the current page names and high-resolution worker. Live provider collection, pretrained model inference and a real Codespaces launch were not exercised. No Bangladesh performance measurement or new model training was performed.

Run tests from `app`:

```sh
python -m unittest test_sentinal_x test_highres_collection test_aircraft_detector test_aircraft_awareness test_dashboard_contract test_aerosentinel test_resilience_ops test_multimodal_pipeline -q
```

Research ideas are selectively applied; the limitations of all cited papers are not solved. Existing foundation runners remain optional. SatMAE/CROMA/STANet/DSen2 model implementations are not added by this release. Legacy module/file names remain for compatibility.

Aircraft Map update (2026-09-27): 144 offline tests passed, including candidate filtering, coordinate/time validation, map data, synthetic-preview isolation, saved-result refresh, and all ten dashboard pages. Python compilation passed. Real provider/model accuracy remains unverified.
