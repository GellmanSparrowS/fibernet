# RL plot figures: export and interactive use

`plot_reward_curve`, `plot_convergence` and `plot_action_distribution` return a
Matplotlib `Figure`. Their default `show=False` creates a standalone figure with
an Agg canvas for headless export. It does not switch the application's plotting
backend, select a new pyplot current figure or register the returned figure with
pyplot. The `save_path` argument still saves the plot during the call.

## Save and edit the returned figure

```python
from fibernet.rl import plot_reward_curve

fig = plot_reward_curve([1.0, 2.0, 3.0], window=2,
                        save_path="reward.png")
fig.axes[0].set_title("Reward after three episodes")
fig.savefig("reward-edited.svg")
```

The same pattern applies to convergence and action-distribution plots. Use
`fig.axes` to edit labels, limits and artists, and `fig.savefig(...)` to export
the result. Saving does not require starting a GUI. Matplotlib remains a base
FiberNet dependency; this change adds no installation requirement.

## Interactive display

Choose an appropriate interactive backend in your application, then request a
pyplot-managed figure explicitly:

```python
from fibernet.rl import plot_convergence

fig = plot_convergence([3.0, 2.0, 1.0], show=True)
```

With `show=True`, the helper uses the caller's backend and calls `plt.show()`.
Whether that opens a window, blocks execution or displays inline depends on the
selected backend and application. The helper does not select or install an
interactive backend; the selected backend and its dependencies must be available.
As with ordinary pyplot use, `plt.show()` may
display other pyplot-managed figures as well.

## Migration from the previous default behavior

Previously, a `show=False` call selected the global Agg backend and registered
its result with pyplot. Code that implicitly selected the helper's plot through
`plt.gcf()` or `plt.savefig()` must now use the returned figure:

```python
# Before: relied on pyplot's current figure.
plot_reward_curve(rewards)
plt.savefig("reward.png")

# Now: select the intended figure directly.
fig = plot_reward_curve(rewards)
fig.savefig("reward.png")
```

A later `plt.show()` will not display a standalone `show=False` result. Use
`show=True` when you need the pyplot-managed path. Function names, parameters,
plot data and the returned `Figure` type are unchanged; pyplot ownership is an
intentional behavior change. Existing calls that pass `save_path` continue to
export without requiring this migration.
