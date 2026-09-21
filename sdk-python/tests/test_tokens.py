"""Deterministic bearer-token and RFC 8693 exchange contract tests."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.hashes import SHA256

from conducto.security import (
    AcquiredToken,
    AudiencePolicy,
    ClockSkewPolicy,
    InvalidActorChainError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidScopeError,
    InvalidSignatureTokenError,
    IssuerPolicy,
    JWTBearerTokenValidator,
    MalformedTokenError,
    ScopeAttenuationError,
    ScopePolicy,
    StaticJWKSResolver,
    TokenCache,
    TokenExpiredError,
    TokenNotYetValidError,
    TrustPolicy,
    UnknownKeyError,
    UnsupportedAlgorithmError,
    attenuate_scopes,
)

ISSUER = "https://issuer.example"
AUDIENCE = "agent-b"


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _sign_es256(private_key: ec.EllipticCurvePrivateKey, header: dict, payload: dict) -> str:
    protected = _b64url(json.dumps(header, separators=(",", ":")).encode())
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{protected}.{body}".encode("ascii")

    der = private_key.sign(signing_input, ec.ECDSA(SHA256()))
    r, s = decode_dss_signature(der)
    sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return f"{protected}.{body}.{_b64url(sig)}"


def _sign_rs256(private_key: rsa.RSAPrivateKey, header: dict, payload: dict) -> str:
    protected = _b64url(json.dumps(header, separators=(",", ":")).encode())
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{protected}.{body}".encode("ascii")
    sig = private_key.sign(signing_input, padding.PKCS1v15(), SHA256())
    return f"{protected}.{body}.{_b64url(sig)}"


class FixedClock:
    def __init__(self, now: float) -> None:
        self._now = now

    def now(self) -> float:
        return self._now


def _policy(**overrides: object) -> TrustPolicy:
    defaults: dict[str, object] = dict(
        version="policy-1",
        issuer=IssuerPolicy(issuer=ISSUER, allowed_algorithms=frozenset({"ES256", "RS256"})),
        audience=AudiencePolicy(audiences=frozenset({AUDIENCE})),
        scopes=ScopePolicy(allowed_scopes=frozenset({"read", "write"})),
        clock_skew=ClockSkewPolicy(leeway_seconds=5),
    )
    defaults.update(overrides)
    return TrustPolicy(**defaults)  # type: ignore[arg-type]


def _claims(**overrides: object) -> dict:
    base = dict(
        iss=ISSUER,
        aud=AUDIENCE,
        sub="workload-a",
        iat=1_000,
        nbf=1_000,
        exp=1_900,
        scope="read write",
    )
    base.update(overrides)
    return base


def test_validate_es256_token_succeeds() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver({(ISSUER, "key-1"): key.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(1_500))
    token = _sign_es256(key, {"alg": "ES256", "kid": "key-1"}, _claims())
    identity = validator.validate(token, policy=_policy())
    assert identity.subject == "workload-a"
    assert identity.issuer == ISSUER
    assert identity.scopes == frozenset({"read", "write"})
    assert identity.policy_version == "policy-1"


def test_validate_rs256_token_succeeds() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    resolver = StaticJWKSResolver({(ISSUER, "rsa-1"): key.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(1_500))
    token = _sign_rs256(key, {"alg": "RS256", "kid": "rsa-1"}, _claims())
    identity = validator.validate(token, policy=_policy())
    assert identity.subject == "workload-a"


def test_unknown_key_id_is_rejected() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver({(ISSUER, "key-1"): key.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(1_500))
    token = _sign_es256(key, {"alg": "ES256", "kid": "rotated-away"}, _claims())
    with pytest.raises(UnknownKeyError):
        validator.validate(token, policy=_policy())


def test_wrong_key_rejects_signature() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    other = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver({(ISSUER, "key-1"): other.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(1_500))
    token = _sign_es256(key, {"alg": "ES256", "kid": "key-1"}, _claims())
    with pytest.raises(InvalidSignatureTokenError):
        validator.validate(token, policy=_policy())


def test_algorithm_not_permitted_by_policy_is_rejected() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver({(ISSUER, "key-1"): key.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(1_500))
    token = _sign_es256(key, {"alg": "ES256", "kid": "key-1"}, _claims())
    policy = _policy(issuer=IssuerPolicy(issuer=ISSUER, allowed_algorithms=frozenset({"RS256"})))
    with pytest.raises(UnsupportedAlgorithmError):
        validator.validate(token, policy=policy)


def test_invalid_issuer_is_rejected() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver({(ISSUER, "key-1"): key.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(1_500))
    token = _sign_es256(key, {"alg": "ES256", "kid": "key-1"}, _claims(iss="https://other.example"))
    with pytest.raises(InvalidIssuerError):
        validator.validate(token, policy=_policy())


def test_invalid_audience_is_rejected() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver({(ISSUER, "key-1"): key.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(1_500))
    token = _sign_es256(key, {"alg": "ES256", "kid": "key-1"}, _claims(aud="other-audience"))
    with pytest.raises(InvalidAudienceError):
        validator.validate(token, policy=_policy())


def test_expired_token_is_rejected() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver({(ISSUER, "key-1"): key.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(2_000))
    token = _sign_es256(key, {"alg": "ES256", "kid": "key-1"}, _claims())
    with pytest.raises(TokenExpiredError):
        validator.validate(token, policy=_policy())


def test_not_yet_valid_token_is_rejected() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver({(ISSUER, "key-1"): key.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(500))
    token = _sign_es256(key, {"alg": "ES256", "kid": "key-1"}, _claims())
    with pytest.raises(TokenNotYetValidError):
        validator.validate(token, policy=_policy())


def test_insufficient_scope_is_rejected() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver({(ISSUER, "key-1"): key.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(1_500))
    token = _sign_es256(key, {"alg": "ES256", "kid": "key-1"}, _claims(scope="read"))
    policy = _policy(
        scopes=ScopePolicy(
            allowed_scopes=frozenset({"read", "write"}), required_scopes=frozenset({"write"})
        )
    )
    with pytest.raises(InvalidScopeError):
        validator.validate(token, policy=policy)


def test_scope_outside_destination_policy_is_rejected() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver({(ISSUER, "key-1"): key.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(1_500))
    token = _sign_es256(key, {"alg": "ES256", "kid": "key-1"}, _claims(scope="read admin"))
    with pytest.raises(InvalidScopeError):
        validator.validate(token, policy=_policy())


def test_revoked_key_is_rejected() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver(
        {(ISSUER, "key-1"): key.public_key()}, revoked=frozenset({(ISSUER, "key-1")})
    )
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(1_500))
    token = _sign_es256(key, {"alg": "ES256", "kid": "key-1"}, _claims())
    with pytest.raises(UnknownKeyError):
        validator.validate(token, policy=_policy())


def test_actor_chain_is_parsed() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver({(ISSUER, "key-1"): key.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(1_500))
    token = _sign_es256(
        key,
        {"alg": "ES256", "kid": "key-1"},
        _claims(act={"sub": "gateway-1", "act": {"sub": "user-1"}}),
    )
    identity = validator.validate(token, policy=_policy())
    assert identity.actor_chain == ("gateway-1", "user-1")


def test_malformed_actor_chain_is_rejected() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver({(ISSUER, "key-1"): key.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(1_500))
    token = _sign_es256(key, {"alg": "ES256", "kid": "key-1"}, _claims(act="not-an-object"))
    with pytest.raises(InvalidActorChainError):
        validator.validate(token, policy=_policy())


def test_malformed_token_is_rejected() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    resolver = StaticJWKSResolver({(ISSUER, "key-1"): key.public_key()})
    validator = JWTBearerTokenValidator(resolver, clock=FixedClock(1_500))
    with pytest.raises(MalformedTokenError):
        validator.validate("not-a-jwt", policy=_policy())


def test_attenuate_scopes_enforces_subset_of_incoming_and_destination() -> None:
    scopes = attenuate_scopes(
        incoming_scopes=frozenset({"read", "write"}),
        requested_scopes=frozenset({"read"}),
        destination_allowed_scopes=frozenset({"read", "write"}),
    )
    assert scopes == frozenset({"read"})

    with pytest.raises(ScopeAttenuationError):
        attenuate_scopes(
            incoming_scopes=frozenset({"read"}),
            requested_scopes=frozenset({"write"}),
            destination_allowed_scopes=frozenset({"read", "write"}),
        )

    with pytest.raises(ScopeAttenuationError):
        attenuate_scopes(
            incoming_scopes=frozenset({"read", "write"}),
            requested_scopes=frozenset({"write"}),
            destination_allowed_scopes=frozenset({"read"}),
        )


def test_token_cache_returns_cached_token_within_ttl() -> None:
    calls = 0

    async def acquire() -> AcquiredToken:
        nonlocal calls
        calls += 1
        return AcquiredToken(access_token="tok", issued_token_type="jwt", expires_at=1_100)

    cache = TokenCache(refresh_skew_seconds=10)
    key = TokenCache.make_key(
        issuer=ISSUER,
        client_id="client-a",
        subject="sub-a",
        actor_chain=(),
        audience=AUDIENCE,
        scopes=frozenset({"read"}),
        policy_version="policy-1",
    )

    async def run() -> None:
        first = await cache.get_or_acquire(key, acquire, clock=FixedClock(1_000))
        second = await cache.get_or_acquire(key, acquire, clock=FixedClock(1_010))
        assert first is second
        assert calls == 1

    asyncio.run(run())


def test_token_cache_refreshes_after_expiry_and_isolates_keys() -> None:
    calls = 0

    async def acquire() -> AcquiredToken:
        nonlocal calls
        calls += 1
        return AcquiredToken(access_token=f"tok-{calls}", issued_token_type="jwt", expires_at=2_000)

    cache = TokenCache(refresh_skew_seconds=10)
    key_a = TokenCache.make_key(
        issuer=ISSUER,
        client_id="client-a",
        subject="sub-a",
        actor_chain=(),
        audience=AUDIENCE,
        scopes=frozenset({"read"}),
        policy_version="policy-1",
    )
    key_b = TokenCache.make_key(
        issuer=ISSUER,
        client_id="client-a",
        subject="sub-b",
        actor_chain=(),
        audience=AUDIENCE,
        scopes=frozenset({"read"}),
        policy_version="policy-1",
    )

    async def run() -> None:
        first = await cache.get_or_acquire(key_a, acquire, clock=FixedClock(1_000))
        expired = await cache.get_or_acquire(key_a, acquire, clock=FixedClock(1_995))
        assert first is not expired
        other = await cache.get_or_acquire(key_b, acquire, clock=FixedClock(1_000))
        assert other is not expired
        assert calls == 3

    asyncio.run(run())


def test_token_cache_prevents_refresh_stampede() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def acquire() -> AcquiredToken:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return AcquiredToken(access_token="tok", issued_token_type="jwt", expires_at=2_000)

    cache = TokenCache(refresh_skew_seconds=10)
    key = TokenCache.make_key(
        issuer=ISSUER,
        client_id="client-a",
        subject="sub-a",
        actor_chain=(),
        audience=AUDIENCE,
        scopes=frozenset({"read"}),
        policy_version="policy-1",
    )

    async def run() -> None:
        task_a = asyncio.create_task(cache.get_or_acquire(key, acquire, clock=FixedClock(1_000)))
        await started.wait()
        task_b = asyncio.create_task(cache.get_or_acquire(key, acquire, clock=FixedClock(1_000)))
        # Cooperative scheduling is deterministic here: task_b's cache lookup never
        # suspends until it blocks on the already-held per-key lock, so yielding
        # control back to the event loop a bounded number of times is sufficient to
        # guarantee task_b has reached that blocking wait before we release it.
        for _ in range(10):
            await asyncio.sleep(0)
        release.set()
        result_a, result_b = await asyncio.gather(task_a, task_b)
        assert result_a is result_b
        assert calls == 1

    asyncio.run(run())


def test_token_cache_bounds_size() -> None:
    async def acquire(idx: int) -> AcquiredToken:
        return AcquiredToken(access_token=f"tok-{idx}", issued_token_type="jwt", expires_at=10_000)

    cache = TokenCache(max_entries=2, refresh_skew_seconds=0)

    async def run() -> None:
        for i in range(3):
            key = TokenCache.make_key(
                issuer=ISSUER,
                client_id="client-a",
                subject=f"sub-{i}",
                actor_chain=(),
                audience=AUDIENCE,
                scopes=frozenset(),
                policy_version="policy-1",
            )
            await cache.get_or_acquire(key, lambda i=i: acquire(i), clock=FixedClock(0))
        assert len(cache._entries) == 2

    asyncio.run(run())
