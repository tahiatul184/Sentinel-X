"""Persistent HTTP transfers for public TIFF assets (one downloader process).

Keep partial bytes across network errors, pause and application restarts. Only
append after the server confirms the byte offset and the original strong ETag.
"""
from pathlib import Path
import json
import os
import re
import shutil
import time

from collection_store import atomic_json

TIFF_HEADERS = (b'II\x2a\x00', b'MM\x00\x2a', b'II\x2b\x00', b'MM\x00\x2b')


class TransferPaused(RuntimeError):
    pass


class InvalidTransfer(RuntimeError):
    pass


def metadata_path(destination):
    return destination.with_name(destination.name + '.json')


def discard_transfer(destination):
    destination.unlink(missing_ok=True)
    metadata_path(destination).unlink(missing_ok=True)


def strong_etag(headers):
    tag = headers.get('ETag', '')
    return tag if tag.startswith('"') and tag.endswith('"') else None


def load_transfer(destination, url, max_bytes):
    try:
        meta = json.loads(metadata_path(destination).read_text(encoding='utf-8'))
        size = destination.stat().st_size
        total = meta.get('total')
        with destination.open('rb') as source:
            header = source.read(4)
        valid_header = header in TIFF_HEADERS if size >= 4 else any(h.startswith(header) for h in TIFF_HEADERS)
        if (meta.get('url') != url or not valid_header or size > max_bytes or
                (total is not None and (not isinstance(total, int) or total < size or total > max_bytes))):
            raise ValueError('Invalid saved transfer')
        return meta, size
    except (OSError, ValueError, TypeError, AttributeError):
        discard_transfer(destination)
        return {}, 0


def transfer_tiff(url, destination, cfg, session, progress, cancelled,
                  *, max_bytes=1024**3, clock=time.monotonic):
    """Download/resume one TIFF. No elapsed-time limit on a progressing transfer.

    Read timeouts still detect a stalled connection. The caller schedules retries;
    all safely written bytes survive. Unknown-size or unversioned servers cannot
    safely resume and explicitly restart a fresh response instead of appending.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    meta, offset = load_transfer(destination, url, max_bytes)
    total = meta.get('total')
    attempts = int(meta.get('attempts', 0))
    started = clock()
    last_report = started
    rate_start, rate_bytes = started, offset
    speed, eta = 0.0, None
    received = offset

    def report(message, *, phase='download', force=False, **fields):
        nonlocal last_report, speed, eta, rate_start, rate_bytes
        now = clock()
        if not force and now - last_report < 1:
            return
        elapsed = now - rate_start
        if elapsed > 0:
            speed = max(0.0, (received - rate_bytes) / elapsed)
            eta = max(0.0, (total - received) / speed) if total and speed > 0 else None
        if elapsed >= 10:
            rate_start, rate_bytes = now, received
        progress(message=message, downloaded_bytes=received, total_bytes=total,
                 speed_bps=speed, eta_seconds=eta, transfer_attempt=attempts,
                 resumed_bytes=offset, transfer_phase=phase,
                 transfer_elapsed_seconds=now-started, **fields)
        last_report = now

    if cancelled():
        raise TransferPaused('Transfer paused; saved bytes will be used when collection resumes.')
    if offset >= 8 and total == offset:
        report('Image already downloaded. Creating your area crop locally.', phase='crop', force=True)
        return

    headers = {'Accept': 'image/tiff,application/octet-stream', 'Accept-Encoding': 'identity'}
    can_resume = offset > 0 and meta.get('etag') and total is not None
    if can_resume:
        headers.update(Range=f'bytes={offset}-', **{'If-Range': meta['etag']})
    attempts += 1
    # Persist attempts before connecting, so connection failures are also counted.
    meta.update(url=url, attempts=attempts)
    if not destination.exists():
        destination.touch()
    atomic_json(metadata_path(destination), meta)
    report('Reconnecting to continue the saved image download.' if can_resume else
           'Connecting to download the satellite image.', force=True)
    try:
        with session.get(url, stream=True, timeout=(10, 30), allow_redirects=False,
                         headers=headers) as response:
            if response.status_code == 416:
                raise InvalidTransfer('The saved byte range is no longer valid; a fresh transfer will be tried.')
            response.raise_for_status()
            if response.status_code not in (200, 206):
                raise RuntimeError(f'Image server returned HTTP {response.status_code}.')
            if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                raise InvalidTransfer('Image server changed the transfer encoding; cannot safely resume.')
            length_raw = response.headers.get('Content-Length')
            try:
                length = int(length_raw) if length_raw is not None else None
            except (ValueError, TypeError) as exc:
                raise InvalidTransfer('Image server returned an invalid Content-Length.') from exc
            if length is not None and length < 0:
                raise InvalidTransfer('Image server returned a negative content size.')
            tag = strong_etag(response.headers)
            if response.status_code == 206:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                if not can_resume or not match:
                    raise InvalidTransfer('Image server returned an unexpected or invalid partial response.')
                first, last, reported_total = map(int, match.groups())
                if (first != offset or last != reported_total-1 or reported_total != total or
                        last < first or (length is not None and length != last-first+1) or
                        (tag is not None and tag != meta['etag'])):
                    raise InvalidTransfer('Image server returned a mismatched range or changed image; refusing to combine bytes.')
                message = 'Resuming the saved image download.'
                mode = 'ab'
            else:
                message = ('The image changed or the server declined resume; starting a fresh image.' if can_resume else
                           'Downloading the image. Progress is saved for retries and restarts.')
                if offset and not can_resume:
                    message = 'The server did not provide a versioned image size; this transfer must restart.'
                offset, received, total = 0, 0, length
                rate_start, rate_bytes = clock(), 0
                mode = 'wb'
            if total is not None and (total < 8 or total > max_bytes):
                raise InvalidTransfer('Provider TIFF is empty or exceeds the 1 GiB download limit.')
            reserve = float(cfg['min_free_gb']) * 1024**3 + 800_000_000
            remaining = (total or max_bytes) - received
            if shutil.disk_usage(destination.parent).free < reserve + remaining:
                raise RuntimeError('Not enough free storage to finish downloading and crop the image. Saved bytes are retained.')
            # For a 200 response discard the old content BEFORE publishing its new
            # validator. A crash must never label old bytes as the new image.
            with destination.open(mode, buffering=0) as out:
                meta = dict(url=url, etag=tag or (meta.get('etag') if mode=='ab' else None),
                            total=total, attempts=attempts)
                atomic_json(metadata_path(destination), meta)
                with destination.open('rb') as source:
                    header = source.read(4)
                report(message, force=True)
                for chunk in response.iter_content(chunk_size=16*1024):
                    if cancelled():
                        raise TransferPaused('Transfer paused; downloaded bytes are saved.')
                    if not chunk:
                        continue
                    if received + len(chunk) > max_bytes or (total and received + len(chunk) > total):
                        raise InvalidTransfer('Image response exceeded its declared size or the 1 GiB limit.')
                    if shutil.disk_usage(destination.parent).free < reserve + len(chunk):
                        raise RuntimeError('Storage became too low. Downloaded bytes are saved.')
                    header = (header + chunk)[:4]
                    if len(header) >= 4 and header not in TIFF_HEADERS:
                        raise InvalidTransfer('Provider returned a non-TIFF response instead of imagery.')
                    out.write(chunk)
                    received += len(chunk)
                    report(message)
                os.fsync(out.fileno())
            if received < 8 or (total is not None and received != total):
                raise RuntimeError('Connection ended before the image completed. Downloaded bytes are saved for retry.')
            total = received
            meta.update(total=total)
            atomic_json(metadata_path(destination), meta)
            report('Download complete. Creating your area crop locally.', phase='crop', force=True)
    except InvalidTransfer:
        discard_transfer(destination)
        received = 0
        report('The server response could not be verified. A fresh download is needed.', phase='error', force=True)
        raise
    except BaseException:
        # Disk and HTTP errors, user stop and process interruption keep partials.
        # Unbuffered writes limit restart loss to a currently arriving 16 KiB chunk.
        received = destination.stat().st_size if destination.exists() else 0
        report('Download interrupted. Saved progress will be used on the next attempt.', phase='waiting', force=True)
        raise
