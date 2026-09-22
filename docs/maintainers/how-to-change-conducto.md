# How to approach a change

## 1. Describe the behavior, not the file

Start with an observable statement:

> A remote caller with a stricter budget cannot receive the broader
> authenticated budget.

Avoid starting with:

> Modify `_attenuate_budget()`.

The first form identifies the contract and test. The second may patch one
location while missing other paths.

## 2. Find the owner

Use the [module map](../system-design/module-map.md). Search for existing public
contracts, tests, and prior art before adding a new abstraction.

If several modules appear to own the same behavior, stop and clarify the
boundary. Do not solve uncertainty by duplicating logic.

## 3. Trace one complete path

For execution behavior, follow:

```text
public API
  → adapter or gateway
  → runtime
  → security/model/capability
  → typed result
  → protocol mapping
```

Read tests alongside implementation. Golden fixtures describe intentional wire
behavior; unit tests often reveal concurrency and ownership assumptions that
types alone cannot show.

## 4. List affected invariants

Use the [invariant catalog](../system-design/invariants.md). Typical questions:

- Could authority increase?
- Could a request execute twice?
- Could state leak between concurrent runs?
- Could a raw exception or secret become public?
- Could an optional dependency become mandatory?
- Who closes newly created resources?

## 5. Make the narrowest complete change

A complete change updates all relevant surfaces:

- implementation;
- public exports;
- type contracts and docstrings;
- deterministic tests;
- golden fixtures for intentional wire changes;
- examples and conceptual documentation;
- package extras or release verification when applicable.

Avoid unrelated refactoring. If a prerequisite refactor is substantial, split
it into its own reviewable change.

## 6. Test the requirement, not a proxy

If the requirement says “never amplify authority,” test combinations of parent
and child authority, already consumed budgets, and concurrency. A test that
only verifies parsing does not prove attenuation.

Test:

- success;
- each typed failure;
- boundary values;
- cancellation and timeout;
- duplicate/racing calls;
- concurrent isolation; and
- cleanup.

## 7. Update the explanation

Behavioral changes should leave a human-readable trail:

- concept change → understanding or system-design page;
- public usage change → using guide;
- maintenance rule → invariant or change recipe;
- durable architecture choice → ADR;
- exact API change → reference and docstring.

## 8. Validate from source and wheel

Run the repository gate in
[validation and debugging](./validation-and-debugging.md). When public imports,
metadata, dependencies, examples, or installed behavior change, build the wheel
and run installed-package verification.
