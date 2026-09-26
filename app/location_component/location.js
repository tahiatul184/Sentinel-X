// Self-contained Streamlit component for browser-authorized device location.
// On first setup it can request location automatically; the visible button
// remains available as a retry path. No IP geolocation or external scripts.
(() => {
  'use strict';
  const button = document.getElementById('locate');
  const status = document.getElementById('status');
  const parentOrigin = window.location.origin;
  let requestedAutomatically = false;
  let locating = false;
  const send = (type, payload = {}) => window.parent.postMessage(
    { isStreamlitMessage: true, type, ...payload }, parentOrigin);
  const resize = () => send('streamlit:setFrameHeight', { height: document.body.scrollHeight + 12 });
  const report = (value) => send('streamlit:setComponentValue', { value, dataType: 'json' });
  const eventId = () => `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const fail = (message) => {
    locating = false;
    button.disabled = false;
    status.textContent = message;
    resize();
    report({ status: 'error', event_id: eventId(), message });
  };
  const locate = () => {
    if (locating) return;
    if (!window.isSecureContext || !navigator.geolocation) {
      fail('Device location is unavailable here. Open the local dashboard at http://127.0.0.1:8501 or enter coordinates manually.');
      return;
    }
    locating = true;
    button.disabled = true;
    status.textContent = 'Finding your location… allow the browser location request.';
    resize();
    navigator.geolocation.getCurrentPosition((position) => {
      locating = false;
      button.disabled = false;
      status.textContent = `Location found (reported accuracy ±${Math.round(position.coords.accuracy)} m). Saving your monitoring area…`;
      resize();
      report({ status: 'ok', event_id: eventId(), latitude: position.coords.latitude,
        longitude: position.coords.longitude, accuracy_m: position.coords.accuracy,
        timestamp_ms: position.timestamp });
    }, (error) => {
      const messages = {
        1: 'Location permission was denied. Allow location for this dashboard in browser site settings, or enter coordinates manually.',
        2: 'Your device could not determine its location. Enable Windows Location services, or enter coordinates manually.',
        3: 'Location lookup timed out. Try again, or enter coordinates manually.'
      };
      fail(messages[error.code] || 'Location lookup failed. Try again or enter coordinates manually.');
    }, { enableHighAccuracy: true, maximumAge: 0, timeout: 20000 });
  };
  button.addEventListener('click', locate);
  window.addEventListener('message', (event) => {
    if (event.source !== window.parent || event.origin !== parentOrigin) return;
    if (event.data?.type === 'streamlit:render') {
      resize();
      const autoRequest = Boolean(event.data?.args?.auto_request);
      if (autoRequest && !requestedAutomatically) {
        requestedAutomatically = true;
        status.textContent = 'Requesting your device location for first-time setup…';
        resize();
        // Small delay lets Streamlit finish mounting before the permission prompt.
        setTimeout(locate, 250);
      }
    }
  });
  send('streamlit:componentReady', { apiVersion: 1 });
  resize();
})();
