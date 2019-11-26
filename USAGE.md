# Usage

torchelastic requires you to implement a `state` object and a `train_step` function.
For details on what these are refer to [how torch elastic works](README.md).

While going through the sections below, refer to the imagenet [example](examples/imagenet/main.py)
for more complete implementation details.

## Implement `state`
The `State` object has two categories of methods that need to be implemented: 
synchronization and persistence.

### `sync()`
Lets take a look at synchronization first. The `sync` method is responsible for
ensuring that all workers get a consistent view of `state`. It is called at 
startup as well as on each event that potentially leaves the workers out of sync,
for instance, on membership changes. Things you should consider doing in `sync` are:

1. Broadcasting global parameters/data from a particular worker (e.g. rank 0).
2. (re)Initializing data loaders based on markers (e.g. last known start index).
3. (re)Initializing the model.

> IMPORTANT: `state.sync()` is **not** meant for synchronizing steps in training. For instance
you should not be synchronizing weights (e.g .all-reduce model weights for synchronous SGD).
These type of collectives operations belong in the `train_step`.

All workers initially create the `state` object with the same constructor arguments.
We refer to this initial state as `S_0` and assume that any worker is able to create
`S_0` without needing any assistance from torchelastic. Essentially `S_0` is the **bootstrap**
state. This concept will become important in the next sections when talking about
state persistence (rollbacks and checkpoints).

### `snapshot()` and `apply(snapshot)`
torchelastic has the ability to rollback a state if a `train_step` fails to 
execute successfully, which may result in the `state` object being left partially
updated. It relies on a properly implemented `snapshot()` and `apply(snapshot)`
methods of the `state` to ensure that the `state` is restored to before the
faulty `train_step`.

The `snapshot()` method, as the name implies, takes a snapshot of the `state`
 and returns the necessary information to be able to restore
the `state` object. You may return **any** object from `snpshot()` so long as you
can use it in the `apply(snapshot)` method. A possible implementation of a 
rollback is:

```python
snapshot = state.snapshot()
try:
    train_step(state)
except RuntimeError:
    state.apply(snapshot)
    state.sync()
```

> NOTE: Since certain fields of the `state` may need to get re-initialized,
 torchelastic calls the `sync()` method. For instance, data loaders may need
 to be restarted as their iterators may end up in a corrupted state when the 
 `train_step` does not exit successfully.

Notice that the apply method is called on the existing `state` object, this implies
that an efficient implementation of `snapshot` should only return mutable, stateful
data. Immutable fields or fields that can be derived from other member variables or
restored in the `sync` method need not be included in the snapshot.
 
 By default the `snapshot()` method returns `None` and the `apply(snapshot)` method
 is a `pass`, which essentially means "rollback not supported".  

### `save(stream)` and `load(stream)`
Much like the `snapshot` and `rollback`, the `save` and `load` methods form a pair.
They are responsible for persisting and restoring the `state` object to and from 
a `stream` which is a *file-like* object 
that is compatible with [pytorch.save](https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save).
torchelastic relies on these methods to provide checkpoint functionality for your job.

> We encourage users to use `torch.save` and `torch.load` methods when implementing
`save` and `load` methods of their `state` class.



## Implement `train_step`

The `train_step` is a function that takes `state` as a single argument
and carries out a partition of the overall training job. 
This is your unit of work and it is up to you to define what
a *unit* is. When deciding what your unit of work should be, keep in mind the
following:

1. Rollbacks and checkpoints are done at `train_step` granularity. This means 
that torchelastic can only recover to the last successful `train_step` Any failures
**during** the train_step are not recoverable.
2. A `train_step` iteration in the `train_loop` has overhead due
to the work that goes in ensuring that your job is fault-tolerant and elastic. 
How much overhead depends on your configurations for rollbacks and checkpoints as well
as how expensive your `snapshot`, `apply`, `save` and `load` functions are.

> In most cases, your job naturally lends itself to an 
obvious `train_step`. The most canonical one for many training jobs is to map
the processing of a mini-batch of training data to a `train_step`.

There is a trade-off to be made between how much work you are 
willing to lose versus how much overhead you want to pay for that security.

## Write a `main.py`

Now that you have `state` and `train_step` implementations all that remains
is to bring everything together and implement a `main` that will execute your 
training. Your script should initialize torchelastic's `coordinator`, create
your `state` object, and call the `train_loop`. Below is a simple example:


```python
import torchelastic
from torchelastic.p2p import CoordinatorP2P

if __name__ == "__main__":
        min_workers = 1
        max_workers = 1
        run_id = 1234
        etcd_endpoint = "localhost:2379"

        state = MyState()

        coordinator = CoordinatorP2P(
            c10d_backend="gloo",
            init_method=f"etcd://{etcd_endpoint}/{run_id}?min_workers={min_workers}&max_workers={max_workers}",
            max_num_trainers=max_workers,
            process_group_timeout=60000,
        )

        torchelastic.train(coordinator, train_step, state)
```

## Configuring

### Metrics
See metrics [documentation](torchelastic/metrics/README.md).

### Checkpoint and Rollback
See checkpoint [documentation](torchelastic/checkpoint/README.md)

### Rendezvous

See rendezvous [documentation](torchelastic/rendezvous/README.md)
