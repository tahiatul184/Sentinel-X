"""Sentinal X research workspace; no simulated performance is presented as measured."""
import json
from pathlib import Path
from sentinal_validation import evaluate, representation_change


def render():
    import streamlit as st
    root = Path(__file__).resolve().parent.parent
    st.header('Sentinal X · Research & Validation')
    st.info('Research candidate system. Local aircraft accuracy is unestablished until independent, geographically held-out evaluation is complete.')
    st.caption('Automatic imagery collection remains on Aircraft Awareness. Native resolution, detector validation, registration and revisit gaps constrain every interpretation.')
    tab1, tab2, tab3 = st.tabs(['Local validation', 'Foundation comparison', 'Research traceability'])
    with tab1:
        st.write('Compare optical baselines and fusion ablations on the same fully reviewed test scenes. Declare source-scene and spatial groups to catch split leakage. Include reviewed scenes with no aircraft.')
        st.download_button('Download evaluation schema example', (root/'research/evaluation_example.json').read_bytes(), 'evaluation_example.json', 'application/json')
        st.warning('The example is synthetic and demonstrates the schema only. Replace every scene and prediction with measured data.')
        threshold = st.number_input('Decision threshold fixed using calibration data', 0.0, 1.0, .5, .05)
        overlap = st.number_input('Matching IoU', .01, 1.0, .5, .05)
        uploaded = st.file_uploader('Evaluation JSON', type=['json'], key='sentinal_eval')
        if uploaded and st.button('Evaluate held-out scenes'):
            try:
                if uploaded.size > 10_000_000:
                    raise ValueError('Maximum evaluation upload is 10 MB')
                report = evaluate(json.loads(uploaded.getvalue()), threshold, overlap)
                st.dataframe([dict(method=k, **v['overall']) for k,v in report['methods'].items()])
                st.json(report)
                st.download_button('Download evaluation report', json.dumps(report, indent=2), 'sentinal_evaluation.json', 'application/json')
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                st.error(f'Invalid evaluation input: {exc}')
    with tab2:
        st.write('Compare two exported embeddings from the same model checkpoint, band schema and aligned footprint. Prithvi, TerraMind, SatMAE or CROMA records may be imported; this does not install or train those architectures.')
        st.download_button('Download feature-pair schema', (root/'research/feature_pair_example.json').read_bytes(), 'feature_pair_example.json', 'application/json')
        uploaded = st.file_uploader('Feature pair JSON with before and after records', type=['json'], key='sentinal_features')
        if uploaded and st.button('Compare feature pair'):
            try:
                if uploaded.size > 2_000_000:
                    raise ValueError('Maximum feature upload is 2 MB')
                pair = json.loads(uploaded.getvalue())
                report = representation_change(pair['before'], pair['after'])
                st.json(report)
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                st.error(f'Invalid feature pair: {exc}')
    with tab3:
        st.markdown((root/'research/RESEARCH_TRACEABILITY.md').read_text())
