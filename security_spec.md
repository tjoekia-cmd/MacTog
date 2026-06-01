# Firestore Security Specification TDD for Toto Macau AI Engine

## 1. Data Invariants

1. **Draw Identification**: A draw must have a numeric string ID, and is stored under `draws/{drawId}`.
2. **Date Constraint**: The draw date must be a valid ISO string of format `YYYY-MM-DD` and must not exceed standard bounds.
3. **Digit Schema**: The digits must be an array of exactly 4 integers, each within the inclusive range $[0, 9]$.
4. **Sum Constraint**: The sum field must equal the sum of all elements in the `digits` array.
5. **Timestamp Immutability**: All records represent immutable physical lottery results. Deletion or updates should generally be restricted, but under our application scope, the server can manage adjustments.

## 2. The "Dirty Dozen" Malicious Payloads

We design these payloads designed to violate system constraints:

1. **Spoofed Author / Unauthenticated Write**: Attempting to write a draw with no auth.
2. **Missing Vital Fields**: Omitting `drawCodeString` in creation.
3. **Array Type Poisoning**: Inserting string "9" instead of number inside `digits`.
4. **Resource Poisoning (Overlength ID)**: Drawing with an ID longer than 128 characters.
5. **Negative Digits**: Writing digit `-2`.
6. **Out of Bound Digits**: Writing digit `15`.
7. **Invalid Sum Field**: Writing digits `[1, 1, 1, 1]` but specifying `sum: 99`.
8. **Shadow Field Injection**: Injecting an unrequested field `cheatCode: "admin"` inside the document.
9. **Invalid Date Format**: Writing `date: "not-a-date"`.
10. **Malicious ID Characters**: Attempting to set document ID to a path traversal pattern `../../hack`.
11. **Client-side Bulk Read Scrape**: Attempting a query without proper auth rules.
12. **Wrong List Size**: Providing an array of 5 elements inside `digits`.

## 3. Test Runner Design

We will write rule tests verifying that all unauthorized attempts fail with `permission_denied`.
