# RL entry points and compatibility

`from fibernet import train_rl` retains its historical name and random-rollout
semantics. It does not learn or return a policy. Its built-in `FiberNetworkEnv`
is currently unavailable: every configuration raises `NotImplementedError`
before graph generation or solver execution.

The environment requires effective Young's modulus E* and Poisson's ratio ν*.
The spring-dynamics `SimResult` provides neither quantity. Its `energy` and
`deformed_positions` fields are not substitutes for a homogenization model.
Changing targets, `default_E` or `applied_strain`, or installing Gymnasium, does
not make those effective properties available. Failed evaluations are not
negative-reward design samples.

`from fibernet.rl import train_rl` is the separate optional policy trainer.
It requires an appropriate caller-managed Gymnasium/SB3 environment, including
a physically justified objective when used for mechanical design. The trainer
does not supply that evaluator. Its model/checkpoint interface differs from the
root function's historical reward-statistics JSON interface.

The root wrapper rejects boolean, noninteger and nonpositive episode counts.
If an environment is successfully constructed, reset/step errors and `info.error`
abort the rollout. Nonfinite rewards are rejected. The environment is closed
before any statistics are saved or success is printed; a cleanup failure does
not mask the original execution error. An unsuccessful rollout produces no
reward summary. A later validated effective-property evaluator would require a
separate physical contract, not additional fields guessed from existing output.

Statistics are staged in an exclusively created temporary file beside the target.
Serialization, writes and close must succeed before atomic replacement. A failed
save preserves the previous output and removes its staging file when cleanup is
possible. On POSIX, existing ordinary permission bits are retained and new files
use mode 0600. Windows chmod controls only the read-only flag; it does not enforce
POSIX rwx bits or an owner-only ACL. File ownership and ACLs are not preserved.
Symlinks, nonregular targets and special mode bits are rejected. This guarantees
replacement atomicity, not crash durability or concurrent-writer locking.
