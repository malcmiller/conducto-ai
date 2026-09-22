"""Pin registration wire JSON and confidential grant redaction."""

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from conducto.registration import RegistrationResult
from conducto.registration.models import REQUEST_ADAPTER, request_document

pytestmark = pytest.mark.golden


def test_registration_wire_contract_and_safe_result() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures" / "registration.v1.json").read_text())
    request = REQUEST_ADAPTER.validate_python(fixture["request"])
    assert request_document(request) == fixture["request"]
    result = RegistrationResult.model_validate(
        {**fixture["safe_result"], "lease_handle": SecretStr("confidential-grant")}
    )
    assert result.model_dump(mode="json") == fixture["safe_result"]
    assert "confidential-grant" not in repr(result)
    assert "confidential-grant" not in result.model_dump_json()
