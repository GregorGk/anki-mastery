"""Cloudflare R2 client — S3-compatible upload via boto3.

Reads from environment:
    R2_ACCOUNT_ID         — Cloudflare account ID
    R2_ACCESS_KEY_ID      — R2 token: Object Read & Write, scoped to bucket
    R2_SECRET_ACCESS_KEY  — secret for the above token
    R2_BUCKET             — bucket name
    R2_PUBLIC_BASE        — base URL of the bucket's public access (custom domain
                            or *.r2.dev). Used to compose object URLs.

R2 is S3-compatible; the only non-S3 wrinkle is the endpoint URL is
`https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com`.

Uploads use Cache-Control: public, max-age=31536000, immutable since each
versioned filename is content-addressable (a regen produces a new filename).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

DEFAULT_CACHE_CONTROL = "public, max-age=31536000, immutable"
DEFAULT_CONTENT_TYPE_MP3 = "audio/mpeg"


@dataclass
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    public_base: str  # e.g. "https://<id>.r2.dev" or custom domain

    @classmethod
    def from_env(cls) -> R2Config:
        missing = []
        env = os.environ
        for var in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_PUBLIC_BASE"):
            if not env.get(var):
                missing.append(var)
        if missing:
            raise RuntimeError(f"R2 env vars missing: {', '.join(missing)}")
        return cls(
            account_id=env["R2_ACCOUNT_ID"],
            access_key_id=env["R2_ACCESS_KEY_ID"],
            secret_access_key=env["R2_SECRET_ACCESS_KEY"],
            bucket=env["R2_BUCKET"],
            public_base=env["R2_PUBLIC_BASE"].rstrip("/"),
        )

    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


@dataclass
class UploadResult:
    object_key: str
    url: str
    bucket: str
    content_md5: str  # hex
    size_bytes: int


class R2Client:
    """Thin wrapper around boto3 S3 client targeting Cloudflare R2."""

    def __init__(self, config: R2Config | None = None) -> None:
        self.config = config or R2Config.from_env()
        self._s3 = boto3.client(
            "s3",
            endpoint_url=self.config.endpoint_url(),
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 5, "mode": "standard"},
                # virtual-host style not supported on R2; force path-style
                s3={"addressing_style": "path"},
            ),
            region_name="auto",
        )

    def public_url(self, object_key: str) -> str:
        return f"{self.config.public_base}/{object_key.lstrip('/')}"

    def head(self, object_key: str) -> dict | None:
        """HEAD probe; returns response dict if present, None on 404."""
        try:
            return self._s3.head_object(Bucket=self.config.bucket, Key=object_key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                return None
            raise

    def upload_bytes(
        self,
        body: bytes,
        object_key: str,
        *,
        content_type: str = DEFAULT_CONTENT_TYPE_MP3,
        cache_control: str = DEFAULT_CACHE_CONTROL,
        extra_metadata: dict[str, str] | None = None,
    ) -> UploadResult:
        """Upload bytes to R2; returns UploadResult with public URL + md5."""
        md5_hex = hashlib.md5(body, usedforsecurity=False).hexdigest()
        kwargs: dict = {
            "Bucket": self.config.bucket,
            "Key": object_key.lstrip("/"),
            "Body": body,
            "ContentType": content_type,
            "CacheControl": cache_control,
        }
        if extra_metadata:
            kwargs["Metadata"] = {k: str(v) for k, v in extra_metadata.items()}
        try:
            self._s3.put_object(**kwargs)
        except (ClientError, BotoCoreError) as exc:
            raise RuntimeError(f"R2 upload failed for {object_key}: {exc}") from exc
        return UploadResult(
            object_key=object_key.lstrip("/"),
            url=self.public_url(object_key),
            bucket=self.config.bucket,
            content_md5=md5_hex,
            size_bytes=len(body),
        )

    def delete(self, object_key: str) -> None:
        try:
            self._s3.delete_object(Bucket=self.config.bucket, Key=object_key.lstrip("/"))
        except ClientError as exc:
            raise RuntimeError(f"R2 delete failed for {object_key}: {exc}") from exc
