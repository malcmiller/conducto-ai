# Maintainer guides

These guides answer a practical question: “Where should I make this change,
and what must remain true afterward?”

- [How to approach a change](./how-to-change-conducto.md)
- [Common change recipes](./change-recipes.md)
- [Validation and debugging](./validation-and-debugging.md)
- [Module ownership](../system-design/module-map.md)
- [System invariants](../system-design/invariants.md)
- [Architecture decisions](../decisions/README.md)

Before editing, identify:

1. the owning module;
2. the public contract;
3. the invariants affected;
4. the deterministic evidence that proves the change; and
5. whether installed package behavior changes.
