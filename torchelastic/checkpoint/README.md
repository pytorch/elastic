# Checkpoint

Users can use torchelastic's checkpoint functionality to ensure that their 
jobs checkpoint the work done at different points in time. 

torchelastic checkpoints `state` objects and calls `state.save` and `state.load`
methods to save and load the checkpoints. It is assumed that all your work
(e.g. learned model weights) is encoded in the `state` object.

The `CheckpointManager` is responsible for saving and loading checkpoints to 
some persistent store. torchelastic ships with a basic `FileSystemCheckpointManager`
that writes the checkpoints to a file. If you have a specific storage solution
that the checkpoints should be written to you may provide your own `CheckpointManager`
implementation and configure your job to use this custom checkpoint manager. 

## Enabling Checkpoints
To enable checkpoints for your job you need to:

1. Set a checkpoint manager for your job
   ```python
   import torchelastic
   import torchelastic.checkpoint.api as checkpoint

   if __name__ == "__main__":
        # ... other setup <omitted> ...
        checkpoint_manager = checkpoint.FileSystemCheckpointManager(checkpoint_dir)
        checkpoint.set_checkpoint_manager(checkpoint_manager)
        
        torchelastic.train(coordinator, train_step, state)
   ```
   
2. Provide correct "serialization" methods in your state. Either,
   1. `snapshot` and `apply` methods in your state
   2. `save` and `load` methods in your state
   > NOTE: If non-trivial `snapshot` and `apply` methods are implemented, then
   rollbacks are enabled by default. If you want only use checkpoints then
   implement the save and load methods directly.
3. Implement `state.should_save_checkpoint` method that returns a boolean value
to indicate whether or not checkpoint should be saved.
   > NOTE: torchelastic asks the state whether a checkpoint should be saved
   at the end of each `train_step`. The `should_save_checkpoint` method can be
   implemented to only save a checkpoint once every `n` train steps (e.g.
   once every epoch).
   
   > NOTE: If checkpointing is turned on each time a `state.sync()` is required,
    torchelastic loads the latest checkpoint for the job. This ensures that
    new members are "caught up" by loading the latest checkpoint.
 
 IMPORTANT: Currently saving and loading checkpoints are done from rank 0.
 
 WARNING: `should_save_checkpoint` may be removed from the `state` API and 
 a different way to customize checkpointing behavior might be provided in future
 releases.
 
 ## Rollback versus Checkpoint
 
 The difference between rollbacks and checkpoints can be confusing. This 
 section clarifies the difference and similarities between them. 
 
 A rollback restores the state object when there is a **recoverable** exception 
 in the `train_step`. This means that the `train_step` threw an exception that
 could be caught and the *worker process has not been terminated*. 
 The `snapshot` that was taken prior to executing the `train_step`
 is applied to the (possibly) corrupt `state` object to restore it 
 to a previously well-defined state. Then `state.sync()` is called to ensure
 that all workers are in consensus regarding the `state`.
 
 > IMPORTANT: On a rollback, the snapshot is applied to the EXISTING
  state object NOT a newly created one.
 
 Depending on how the `sync()` method is implemented, rollbacks also help 
 recover from worker losses. When workers are lost, the `train_step` on the
 surviving workers would eventually fail on a collectives operation (e.g. `all_reduce`).
 If the `state` has a non-trivial `snapshot` then these workers will rollback
 their states to the `snapshot` and enter the re-rendezvous barrier. When
 the lost workers are replaced or the minimum number of workers have joined,
 torchelastic will call `state.sync()` before resuming execution. The `sync`
 method can be implemented in such a way that the `state` in the new workers
 can be restored from the surviving workers.
 
 > IMPORTANT: Currently torchelastic's checkpoint loading logic only loads the
 checkpoint from the worker with rank 0 if the checkpoint has not been loaded
 already on this worker. This implies that as long as rank 0 is a surviving worker,
 the state on other workers are restored from snapshots rather than checkpoints.
  
 On the other hand, checkpoints help recover from a total loss
 of the job (e.g all workers are lost). Checkpoints are persisted to a persistent
 store, such as a shared filesystem or S3, whereas rollbacks exist only in memory.
 When the job is replaced or retried, the workers load the latest checkpoint.
 
 The following table summarizes the similarities and differences between rollbacks
 and checkpoints.
 
| Item                              | Rollback            | Checkpoint      |
|-----------------------------------|:--------------------|:----------------|
| Persistence                       |in memory            | store           |
| Failure Recovery                  |non-process terminating exceptions | total loss of all workers |
| Relevant `state` methods          |`snapshot`, `apply`  |`save`, `load`   |
| Applied/Loaded onto               | existing `state` object| Initial `state` object|
| Enabled by                        | providing non-trivial `snapshot` and `apply` implementations| `set_checkpoint_manager` and return `should_save_checkpoint == True`/
| Performance expectation           | light weight        | writing to store can take long |
| `snapshot`/`save` called on       | before each `train_step` | before each `train_step` **if** `should_save_checkpoint` is `True`|
| `apply`/`load` called on          | on `train_step` exception| beginning of the *outer_loop* in `train_loop`|
  
