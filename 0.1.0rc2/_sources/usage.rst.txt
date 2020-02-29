Usage
=====

torchelastic requires you to implement a `state` object and a `train_step` function.

For details on what these are refer to `how torch elastic works <README.md>`_.

While going through the sections below, refer to the imagenet `example <examples/imagenet/main.py>`_

for more complete implementation details.

# Implement `state`
===================

The `State` object has two categories of methods that need to be implemented: 

synchronization and persistence.

## `sync()`
===========

Lets take a look at synchronization first. The `sync` method is responsible for

ensuring that all workers get a consistent view of `state`. It is called at 

startup as well as on each event that potentially leaves the workers out of sync,

for instance, on membership changes and rollback events. Torchelastic relies on

the `sync()` method for `state` recovery from surviving workers (e.g. when

there are membership changes, either due to worker failure or elasticity,

the new workers receive the most up-to-date `state` from one of the surviving 

workers - usually the one that has the most recent `state` - we call this worker

the most **tenured** worker). 

Things you should consider doing in `sync` are:

1. Broadcasting global parameters/data from a particular worker (e.g. rank 0).

2. (re)Initializing data loaders based on markers (e.g. last known start index).

3. (re)Initializing the model.

> IMPORTANT: `state.sync()` is **not** meant for synchronizing steps in training. For instance

you should not be synchronizing weights (e.g .all-reduce model weights for synchronous SGD).

These type of collectives operations belong in the `train_step`.

All workers initially create the `state` object with the same constructor arguments.

We refer to this initial state as `S_0` and assume that any worker is able to create

`S*0` without needing any assistance from torchelastic. Essentially `S*0` is the **bootstrap**

state. This concept will become important in the next sections when talking about

state persistence (rollbacks and checkpoints).

## (optional) `capture*snapshot()` and `apply*snapshot()`
=========================================================

> You do not have to implement these methods if you do not want rollbacks

from failed `train_steps` 

torchelastic has the ability to rollback a state if a `train_step` fails to 

execute successfully, which may result in the `state` object being left partially

updated. It relies on a properly implemented `capture*snapshot()` and `apply*snapshot()`

methods of the `state` to ensure that the `state` is restored to before the

faulty `train_step`.

The `capture_snapshot()` method, as the name implies, takes a snapshot of the `state`

 and returns the necessary information to be able to restore
the `state` object. You may return **any** object from `capture_snapshot()` so long as you

can use it in the `apply_snapshot(snapshot)` method. A possible implementation of a 

rollback is:

```python

snapshot = state.capture_snapshot()

try:

	train\_step(state)

except RuntimeError:

	state.apply\_snapshot(snapshot)

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
 
 By default the `capture*snapshot()` method returns `None` and the `apply*snapshot()` method

 is a `pass`, which essentially means "rollback not supported".
 
 > IMPORTANT: The `apply_snapshot` object should make **no** assumptions about

 which `state` object it is called on (e.g. the values of the member variables).

 That is, applying a `snapshot`

 to **any** state followed by `state.sync()` should effectively restore the

 state object to when the corresponding `capture_snapshot` method was called. 

 A good rule of thumb is that the `apply_snapshot` should act more like a `set`

 method rather than an `update` method.  

## (optional) `save(stream)` and `load(stream)`
===============================================

> You do not have to implement these methods if you do not plan on using

checkpointing.

Much like the `capture*snapshot` and `apply*snapshot`, the `save` and `load` methods form a pair.

They are responsible for persisting and restoring the `state` object to and from 

a `stream` which is a *file-like* object 

that is compatible with `pytorch.save <https://pytorch.org/docs/stable/torch.html?highlight=save#torch.save>`_.

torchelastic relies on these methods to provide checkpoint functionality for your job.

> We encourage users to use `torch.save` and `torch.load` methods when implementing

`save` and `load` methods of their `state` class.

> NOTE: The default implementations of `save` and `load` use `capture_snapshot`

and `apply_snapshot`

# Implement `train_step`
========================

The `train_step` is a function that takes `state` as a single argument

and carries out a partition of the overall training job. 

This is your unit of work and it is up to you to define what

a *unit* is. When deciding what your unit of work should be, keep in mind the

following:

1. Rollbacks and checkpoints are done at `train_step` granularity. This means 

that torchelastic can only recover to the last successful `train_step` Any failures

**during** the train_step are not recoverable.

2. A `train*step` iteration in the `train*loop` has overhead due

to the work that goes in ensuring that your job is fault-tolerant and elastic. 

How much overhead depends on your configurations for rollbacks and checkpoints as well

as how expensive your `snapshot`, `apply`, `save` and `load` functions are.

> In most cases, your job naturally lends itself to an 

obvious `train_step`. The most canonical one for many training jobs is to map

the processing of a mini-batch of training data to a `train_step`.

There is a trade-off to be made between how much work you are 

willing to lose versus how much overhead you want to pay for that security.

# Write a `main.py`
===================

Now that you have `state` and `train_step` implementations all that remains

is to bring everything together and implement a `main` that will execute your 

training. Your script should initialize torchelastic's `coordinator`, create

your `state` object, and call the `train_loop`. Below is a simple example:


```python

import torchelastic

from torchelastic.p2p import CoordinatorP2P

if **name** == "**main**":

		min\_workers = 1

		max\_workers = 1

		run\_id = 1234

		etcd\_endpoint = "localhost:2379"

		state = MyState()

		coordinator = CoordinatorP2P(

			c10d\_backend="gloo",

			init\_method=f"etcd://{etcd\_endpoint}/{run\_id}?min\_workers={min\_workers}&max\_workers={max\_workers}",

			max\_num\_trainers=max\_workers,

			process\_group\_timeout=60000,

		)

		torchelastic.train(coordinator, train\_step, state)

```

# Configuring
=============

## Metrics
==========

See metrics `documentation <torchelastic/metrics/README.md>`_.

## Checkpoint and Rollback
==========================

See checkpoint `documentation <torchelastic/checkpoint/README.md>`_

## Rendezvous
=============

See rendezvous `documentation <torchelastic/rendezvous/README.md>`_

