## Invisibility Cloak: Disappearance under Human Pose Estimation via Backdoor Attacks

In `IntC.py`, we provide the implementations of our IntC attacks, which can be applied to existing HPE techniques.

### Usage

1. Selecting a target HPE technique.
2. Config the environment following their instructions.
3. Involve our implementation codes `IntC.py`.
4. Call our function `InvisibilityCloak()` (as defined in `IntC.py`) after loading the clean data for data poisoning.
5. Train a backdoored HPE model based on the poisoned training data.
