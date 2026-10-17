"""Tiny key/value store: Cloudflare R2 (S3 API) when configured, else a local folder.

Keys look like paths, e.g. 'odds/CFB/2026-10-16T13-00.json'.  Values are JSON
(or raw bytes for cached downloads).  Every write is a whole-object put, so the
same code works on R2 and on disk.
"""
from __future__ import annotations
import gzip
import io
import json
import os
from pathlib import Path


class Store:
    def __init__(self, secrets: dict | None = None, local_dir: str | None = None):
        s = secrets or {}
        self.bucket = s.get('R2_BUCKET')
        self.kind = 'local'
        self._s3 = None
        if s.get('R2_ACCOUNT_ID') and s.get('R2_ACCESS_KEY_ID') and s.get('R2_SECRET_ACCESS_KEY') and self.bucket:
            try:
                import boto3
                self._s3 = boto3.client(
                    's3', endpoint_url=f"https://{s['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                    aws_access_key_id=s['R2_ACCESS_KEY_ID'], aws_secret_access_key=s['R2_SECRET_ACCESS_KEY'],
                    region_name='auto')
                self.kind = 'r2'
            except Exception as exc:          # boto3 missing or bad config -> local fallback
                self.error = str(exc)
        base = local_dir or s.get('MODEL_DATA_DIR') or os.environ.get('BTC_DATA_DIR') or '.btc_data'
        self.root = Path(base)
        if self.kind == 'local':
            self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ raw bytes
    def put_bytes(self, key: str, data: bytes):
        if self.kind == 'r2':
            self._s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        else:
            p = self.root / key
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + '.tmp')
            tmp.write_bytes(data)
            tmp.replace(p)

    def get_bytes(self, key: str) -> bytes | None:
        if self.kind == 'r2':
            try:
                return self._s3.get_object(Bucket=self.bucket, Key=key)['Body'].read()
            except Exception:
                return None
        p = self.root / key
        return p.read_bytes() if p.exists() else None

    def exists(self, key: str) -> bool:
        return self.get_bytes(key) is not None if self.kind == 'r2' else (self.root / key).exists()

    def list(self, prefix: str) -> list[str]:
        if self.kind == 'r2':
            keys, token = [], None
            while True:
                kw = dict(Bucket=self.bucket, Prefix=prefix)
                if token:
                    kw['ContinuationToken'] = token
                resp = self._s3.list_objects_v2(**kw)
                keys += [o['Key'] for o in resp.get('Contents', [])]
                if not resp.get('IsTruncated'):
                    break
                token = resp.get('NextContinuationToken')
            return sorted(keys)
        base = self.root / prefix
        folder = base if base.is_dir() else base.parent
        if not folder.exists():
            return []
        return sorted(str(p.relative_to(self.root)).replace('\\', '/') for p in folder.rglob('*')
                      if p.is_file() and not p.name.endswith('.tmp')
                      and str(p.relative_to(self.root)).replace('\\', '/').startswith(prefix))

    def delete(self, key: str):
        if self.kind == 'r2':
            self._s3.delete_object(Bucket=self.bucket, Key=key)
        else:
            p = self.root / key
            if p.exists():
                p.unlink()

    # ------------------------------------------------------------ JSON
    def put_json(self, key: str, value, compress: bool | None = None):
        raw = json.dumps(value, default=str).encode()
        if compress or (compress is None and key.endswith('.gz')):
            raw = gzip.compress(raw)
        self.put_bytes(key, raw)

    def get_json(self, key: str, default=None):
        raw = self.get_bytes(key)
        if raw is None:
            return default
        if raw[:2] == b'\x1f\x8b':
            raw = gzip.decompress(raw)
        try:
            return json.loads(raw)
        except ValueError:
            return default

    def describe(self) -> str:
        return f'Cloudflare R2 bucket "{self.bucket}"' if self.kind == 'r2' else f'local folder {self.root.resolve()}'


def read_frame(store: Store, key: str):
    import pandas as pd
    raw = store.get_bytes(key)
    if raw is None:
        return None
    return pd.read_csv(io.BytesIO(raw), compression='gzip' if raw[:2] == b'\x1f\x8b' else None)


def write_frame(store: Store, key: str, frame):
    buf = io.BytesIO()
    frame.to_csv(buf, index=False)
    store.put_bytes(key, gzip.compress(buf.getvalue()))
