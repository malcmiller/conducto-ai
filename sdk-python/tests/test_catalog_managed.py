"""Atomic managed catalog admission, replay fencing, and lease regressions."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from conducto.core.catalog import (
    AgentCatalog,
    CatalogEntry,
    CatalogLifecycleState,
    CatalogManagedResult,
    CatalogValidationError,
    DeploymentType,
)
from conducto.core.catalog import (
    CatalogInstanceState as State,
)
from conducto.core.catalog import (
    CatalogManagedCode as Code,
)
from conducto.core.catalog import (
    CatalogManagedCommand as Command,
)


class Clock:
    """Deterministic injectable lease clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        """Return the controlled timestamp."""
        return self.now


def entry(instance_id: str = "one", *, endpoint: str = "https://example.org/a2a") -> CatalogEntry:
    """Construct a valid remote instance with a transport-specific endpoint."""
    card: dict[str, Any] = {
        "name": "demo",
        "description": "Demo agent",
        "version": "1.0.0",
        "supportedInterfaces": [
            {"url": endpoint, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
        ],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [{"id": "echo", "name": "Echo", "description": "Echo", "tags": []}],
    }
    return CatalogEntry(
        "org.demo",
        instance_id,
        "org",
        DeploymentType.REMOTE_CONTAINER,
        endpoint + "/card.json",
        card,
    )


def registration(instance_id: str = "one", **changes: Any) -> Command:
    """Construct an authenticated registration command."""
    return replace(
        Command(
            operation="register",
            agent_id="org.demo",
            instance_id=instance_id,
            owner="org",
            environment="test",
            subject_id="principal",
            issuer="issuer",
            idempotency_key="create-" + instance_id,
            fingerprint="register-" + instance_id,
            expected_generation=0,
            lease_seconds=30,
            entry=entry(instance_id),
            deployment_id="deployment",
            provenance="attestation",
        ),
        **changes,
    )


def followup(command: Command, handle: str, **changes: Any) -> Command:
    """Construct a renewal command with fresh operation identity."""
    return replace(
        replace(
            command,
            operation="renew",
            idempotency_key="renew",
            fingerprint="renew",
            expected_generation=1,
            lease_handle=handle,
            entry=None,
        ),
        **changes,
    )


def concurrently(
    catalog: AgentCatalog, commands: list[Command]
) -> tuple[CatalogManagedResult, ...]:
    """Submit competing commands without sleeps or network dependencies."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        return tuple(pool.map(catalog.manage_instance, commands))


def test_registration_replay_and_immutable_metadata() -> None:
    catalog = AgentCatalog(clock=Clock())
    command = registration()
    result = catalog.manage_instance(command)
    assert result.code is Code.OK
    assert result.generation == 1
    assert result.state is State.ACTIVE
    assert result.lease_handle
    assert catalog.manage_instance(command) == result
    assert result.lease_handle not in repr(result)
    assert result.lease_handle not in repr(replace(command, lease_handle=result.lease_handle))
    instance = catalog.snapshot().agents[0].instances[0]
    assert (instance.environment, instance.deployment_id, instance.provenance) == (
        "test",
        "deployment",
        "attestation",
    )
    assert (instance.subject_id, instance.issuer) == ("principal", "issuer")
    with pytest.raises(FrozenInstanceError):
        for attribute in ("environment",):
            setattr(instance, attribute, "other")


def test_generation_fences_old_receipts_and_conflicting_keys() -> None:
    clock = Clock()
    catalog = AgentCatalog(clock=clock)
    command = registration()
    result = catalog.manage_instance(command)
    assert (
        catalog.manage_instance(replace(command, fingerprint="changed")).code
        is Code.IDEMPOTENCY_CONFLICT
    )
    renew = followup(command, result.lease_handle)
    clock.now = 10
    renewed = catalog.manage_instance(renew)
    assert renewed.code is Code.OK
    assert renewed.generation == 2
    assert renewed.lease_expires_at == 40
    assert renewed.lease_handle == ""
    assert catalog.manage_instance(renew) == renewed
    assert catalog.manage_instance(command).code is Code.STALE_GENERATION
    assert (
        catalog.manage_instance(replace(renew, idempotency_key="other")).code
        is Code.STALE_GENERATION
    )
    status = replace(
        renew,
        operation="status",
        expected_generation=2,
        idempotency_key="status",
        fingerprint="status",
    )
    assert catalog.manage_instance(status).generation == 2


def test_receipt_preflight_needs_no_card_and_does_not_create_or_extend_authority() -> None:
    clock = Clock()
    catalog = AgentCatalog(clock=clock)
    command = registration()
    preflight = replace(command, entry=None)
    assert catalog.lookup_managed_request(preflight) is None
    assert catalog.revision == 0
    assert catalog.get(command.agent_id) is None
    admitted = catalog.manage_instance(command)
    revision = catalog.revision
    clock.now = 10
    assert catalog.lookup_managed_request(preflight) == admitted
    assert catalog.revision == revision
    conflict = catalog.lookup_managed_request(replace(preflight, fingerprint="changed"))
    assert conflict is not None and conflict.code is Code.IDEMPOTENCY_CONFLICT
    denied = catalog.lookup_managed_request(replace(preflight, environment="other"))
    assert denied is not None and denied.code is Code.IDENTITY_CONFLICT
    assert catalog.lookup_managed_request(replace(preflight, subject_id="other")) is None
    clock.now = 30
    expired = catalog.lookup_managed_request(preflight)
    assert expired is not None
    assert (expired.code, expired.generation, expired.state) == (Code.EXPIRED, 2, State.EXPIRED)
    assert expired.lease_handle == ""
    assert catalog.snapshot().agents == ()


def test_receipt_preflight_cannot_replay_superseded_or_revoked_admission() -> None:
    catalog = AgentCatalog(clock=Clock())
    command = registration()
    admitted = catalog.manage_instance(command)
    renew = followup(command, admitted.lease_handle)
    assert catalog.lookup_managed_request(renew) is None
    renewed = catalog.manage_instance(renew)
    stale = catalog.lookup_managed_request(replace(command, entry=None))
    assert stale is not None and stale.code is Code.STALE_GENERATION
    assert catalog.lookup_managed_request(renew) == renewed
    catalog.revoke(command.agent_id)
    inactive = catalog.lookup_managed_request(replace(command, entry=None))
    assert inactive is not None and inactive.code is Code.INACTIVE
    assert inactive.lease_handle == ""


def test_renewal_does_not_shorten_an_existing_lease() -> None:
    clock = Clock()
    catalog = AgentCatalog(clock=clock)
    command = registration()
    admitted = catalog.manage_instance(command)
    clock.now = 10
    renewed = catalog.manage_instance(followup(command, admitted.lease_handle, lease_seconds=1))
    assert renewed.code is Code.OK
    assert renewed.generation == 2
    assert renewed.lease_expires_at == admitted.lease_expires_at
    assert catalog.snapshot().agents[0].instances[0].last_heartbeat_at == 10


def test_handle_authorization_precedes_conflicts_and_failures_do_not_reserve_keys() -> None:
    catalog = AgentCatalog(clock=Clock())
    command = registration()
    admitted = catalog.manage_instance(command)
    renewal = followup(command, admitted.lease_handle)
    invalid = replace(renewal, lease_handle="incorrect")
    assert catalog.manage_instance(invalid).code is Code.INVALID_HANDLE
    assert catalog.lookup_managed_request(renewal) is None
    renewed = catalog.manage_instance(renewal)
    assert renewed.code is Code.OK
    conflicted = replace(invalid, fingerprint="different", expected_generation=2)
    for result in (
        catalog.manage_instance(conflicted),
        catalog.lookup_managed_request(conflicted),
    ):
        assert result is not None and result.code is Code.INVALID_HANDLE
    authorized_conflict = replace(conflicted, lease_handle=admitted.lease_handle)
    assert catalog.manage_instance(authorized_conflict).code is Code.IDEMPOTENCY_CONFLICT
    assert catalog.manage_instance(renewal) == renewed


@pytest.mark.parametrize("field", ["owner", "environment", "subject_id", "issuer"])
def test_handles_are_bound_to_full_authenticated_identity(field: str) -> None:
    catalog = AgentCatalog(clock=Clock())
    command = registration()
    result = catalog.manage_instance(command)
    renew = followup(command, result.lease_handle, **{field: "substituted"})
    assert catalog.manage_instance(renew).code is Code.IDENTITY_CONFLICT
    wrong_handle = followup(command, "different")
    assert catalog.manage_instance(wrong_handle).code is Code.INVALID_HANDLE
    assert (
        catalog.manage_instance(followup(command, "invalid-\N{SNOWMAN}")).code
        is Code.INVALID_HANDLE
    )


def test_drain_and_admin_revoke_preserve_healthy_siblings() -> None:
    catalog = AgentCatalog(clock=Clock())
    command = registration()
    admitted = catalog.manage_instance(command)
    sibling = registration("two", entry=entry("two", endpoint="https://other.example.org/a2a"))
    assert catalog.manage_instance(sibling).code is Code.OK
    drain = followup(command, admitted.lease_handle, operation="drain", fingerprint="drain")
    drained = catalog.manage_instance(drain)
    assert (drained.state, drained.generation) == (State.DRAINING, 2)
    assert [item.instance_id for item in catalog.snapshot().agents[0].instances] == ["two"]
    assert catalog.manage_instance(drain) == drained
    assert (
        catalog.manage_instance(
            followup(
                command, admitted.lease_handle, expected_generation=2, idempotency_key="heartbeat"
            )
        ).code
        is Code.INACTIVE
    )
    revoke = followup(
        command,
        "",
        operation="revoke",
        expected_generation=2,
        subject_id="admin",
        issuer="admin-issuer",
        fingerprint="revoke",
    )
    revoked = catalog.manage_instance(revoke)
    assert (revoked.state, revoked.generation) == (State.REVOKED, 3)
    assert catalog.manage_instance(revoke) == revoked
    assert [item.instance_id for item in catalog.snapshot().agents[0].instances] == ["two"]
    assert catalog.manage_instance(command).code is Code.INACTIVE


def test_deregister_retains_identity_tombstone() -> None:
    catalog = AgentCatalog(clock=Clock())
    command = registration()
    admitted = catalog.manage_instance(command)
    remove = followup(command, admitted.lease_handle, operation="deregister", fingerprint="remove")
    result = catalog.manage_instance(remove)
    assert (result.state, result.generation) == (State.REMOVED, 2)
    assert catalog.manage_instance(remove) == result
    record = catalog.get(command.agent_id)
    assert record is not None and record.instances == ()
    assert catalog.manage_instance(command).code is Code.INACTIVE
    assert (
        catalog.manage_instance(replace(command, idempotency_key="restart")).code is Code.INACTIVE
    )
    assert catalog.manage_instance(registration("new-instance")).code is Code.OK


@pytest.mark.parametrize(
    "operation", ["quarantine", "disable", "revoke", "remove", "register", "renew"]
)
def test_direct_mutations_permanently_fence_managed_authority(operation: str) -> None:
    catalog = AgentCatalog(clock=Clock())
    command = registration()
    admitted = catalog.manage_instance(command)
    if operation in ("register", "renew"):
        with pytest.raises(CatalogValidationError, match="Managed"):
            if operation == "register":
                catalog.register_instance(entry())
            else:
                catalog.renew_lease(command.agent_id, command.instance_id)
    elif operation == "remove":
        catalog.remove(command.agent_id)
    else:
        lifecycle = {
            "quarantine": CatalogLifecycleState.QUARANTINED,
            "disable": CatalogLifecycleState.DISABLED,
            "revoke": CatalogLifecycleState.REVOKED,
        }[operation]
        catalog.set_lifecycle(command.agent_id, lifecycle)
        catalog.reactivate(command.agent_id)
    assert catalog.manage_instance(command).code is Code.INACTIVE
    assert catalog.manage_instance(followup(command, admitted.lease_handle)).code is Code.INACTIVE
    assert catalog.snapshot().agents == ()


@pytest.mark.parametrize("sweep", [False, True])
def test_expiry_fences_replays_and_has_original_principal_attribution(sweep: bool) -> None:
    clock = Clock()
    catalog = AgentCatalog(clock=clock)
    command = registration()
    admitted = catalog.manage_instance(command)
    assert catalog.manage_instance(registration("two", lease_seconds=100)).code is Code.OK
    clock.now = 30
    if sweep:
        ((agent_id, instance_id, expired),) = catalog.expire_managed_instances()
        assert (agent_id, instance_id) == ("org.demo", "one")
        assert (expired.generation, expired.state) == (2, State.EXPIRED)
        assert (expired.subject_id, expired.issuer) == ("principal", "issuer")
        assert expired.lease_handle == ""
        assert catalog.expire_managed_instances() == ()
    assert catalog.manage_instance(command).code is Code.EXPIRED
    assert catalog.manage_instance(followup(command, admitted.lease_handle)).code is Code.EXPIRED
    assert [item.instance_id for item in catalog.snapshot().agents[0].instances] == ["two"]


def test_concurrent_create_and_renew_are_atomic() -> None:
    catalog = AgentCatalog(clock=Clock())
    command = registration()
    results = concurrently(catalog, [command] * 16)
    assert all(result == results[0] for result in results)
    handle = results[0].lease_handle
    commands = [
        followup(command, handle, idempotency_key=f"renew-{index}", fingerprint=f"renew-{index}")
        for index in range(16)
    ]
    results = concurrently(catalog, commands)
    assert sum(result.code is Code.OK for result in results) == 1
    assert sum(result.code is Code.STALE_GENERATION for result in results) == 15


def test_concurrent_distinct_creates_cannot_claim_one_instance_twice() -> None:
    catalog = AgentCatalog(clock=Clock())
    commands = [
        registration(idempotency_key=f"create-{index}", fingerprint=f"create-{index}")
        for index in range(16)
    ]
    results = concurrently(catalog, commands)
    assert sum(result.code is Code.OK for result in results) == 1
    assert sum(result.code is Code.IDENTITY_CONFLICT for result in results) == 15


def test_legacy_expiration_sweep_preserves_managed_generation_and_drain_expiry() -> None:
    clock = Clock()
    catalog = AgentCatalog(clock=clock)
    command = registration()
    admitted = catalog.manage_instance(command)
    drain = followup(command, admitted.lease_handle, operation="drain", fingerprint="drain")
    assert catalog.manage_instance(drain).generation == 2
    clock.now = 30
    assert catalog.expire_instances() == (("org.demo", "one"),)
    assert catalog.expire_instances() == ()
    result = catalog.manage_instance(drain)
    assert (result.code, result.generation, result.state) == (Code.EXPIRED, 3, State.EXPIRED)
    assert catalog.expire_managed_instances() == (("org.demo", "one", result),)
    assert catalog.expire_managed_instances() == ()


@pytest.mark.parametrize("observe", ["lookup", "manage"])
def test_lazily_observed_expiry_is_delivered_once_with_original_attribution(observe: str) -> None:
    clock = Clock()
    catalog = AgentCatalog(clock=clock)
    command = registration()
    catalog.manage_instance(command)
    clock.now = 30
    if observe == "lookup":
        expired = catalog.lookup_managed_request(replace(command, entry=None))
    else:
        expired = catalog.manage_instance(command)
    assert expired is not None and expired.code is Code.EXPIRED
    assert (expired.subject_id, expired.issuer) == ("principal", "issuer")
    assert expired.lease_handle == ""
    catalog.remove(command.agent_id)
    assert catalog.expire_managed_instances() == (("org.demo", "one", expired),)
    assert catalog.expire_managed_instances() == ()
    assert catalog.manage_instance(command).code is Code.EXPIRED
    assert catalog.expire_managed_instances() == ()


def test_admission_does_not_take_over_unmanaged_or_incompatible_siblings() -> None:
    catalog = AgentCatalog(clock=Clock())
    catalog.register_instance(entry())
    assert catalog.manage_instance(registration()).code is Code.IDENTITY_CONFLICT
    assert (
        catalog.manage_instance(
            registration("two", owner="impostor", entry=replace(entry("two"), owner="impostor"))
        ).code
        is Code.IDENTITY_CONFLICT
    )
    changed = dict(entry("two").agent_card)
    changed["version"] = "2.0.0"
    assert (
        catalog.manage_instance(
            registration("two", entry=replace(entry("two"), agent_card=changed))
        ).code
        is Code.IDENTITY_CONFLICT
    )
    assert catalog.manage_instance(registration("two")).code is Code.OK
    with pytest.raises(CatalogValidationError, match="sibling"):
        catalog.register_instance(replace(entry("three"), agent_card=changed))


def test_capacity_never_evicts_receipts_or_tombstones() -> None:
    catalog = AgentCatalog(clock=Clock(), managed_capacity=1)
    command = registration()
    admitted = catalog.manage_instance(command)
    assert catalog.manage_instance(command) == admitted
    assert (
        catalog.manage_instance(followup(command, admitted.lease_handle)).code
        is Code.CAPACITY_EXCEEDED
    )
    assert catalog.manage_instance(registration("two")).code is Code.CAPACITY_EXCEEDED
    catalog.remove(command.agent_id)
    assert catalog.manage_instance(command).code is Code.INACTIVE
    assert catalog.manage_instance(registration("two")).code is Code.CAPACITY_EXCEEDED


@pytest.mark.parametrize("operation", ["deregister", "revoke"])
def test_terminal_operations_have_reserved_capacity_and_replay_safely(operation: str) -> None:
    catalog = AgentCatalog(clock=Clock(), managed_capacity=1)
    command = registration()
    admitted = catalog.manage_instance(command)
    terminal = followup(command, admitted.lease_handle, operation=operation, fingerprint=operation)
    result = catalog.manage_instance(terminal)
    assert result.code is Code.OK
    assert catalog.manage_instance(terminal) == result
    assert catalog.snapshot().agents == ()
    assert catalog.manage_instance(command).code is Code.INACTIVE
    assert catalog.manage_instance(registration("two")).code is Code.CAPACITY_EXCEEDED


def test_status_can_reconcile_an_unknown_generation_without_extending_authority() -> None:
    catalog = AgentCatalog(clock=Clock())
    command = registration()
    admitted = catalog.manage_instance(command)
    renewed = catalog.manage_instance(followup(command, admitted.lease_handle))
    status = followup(
        command,
        admitted.lease_handle,
        operation="status",
        expected_generation=0,
        idempotency_key="status",
        fingerprint="status",
    )
    observed = catalog.manage_instance(status)
    assert observed.code is Code.OK
    assert observed.generation == renewed.generation
    assert observed.lease_expires_at == renewed.lease_expires_at


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("-inf"), float("nan")])
def test_nonfinite_or_nonpositive_leases_are_rejected(duration: float) -> None:
    catalog = AgentCatalog(clock=Clock())
    assert catalog.manage_instance(registration(lease_seconds=duration)).code is Code.INVALID_LEASE
    with pytest.raises(CatalogValidationError, match="finite and positive"):
        replace(entry(), lease_seconds=duration)
    catalog.register_instance(entry())
    with pytest.raises(CatalogValidationError, match="finite and positive"):
        catalog.renew_lease("org.demo", "one", lease_seconds=duration)
