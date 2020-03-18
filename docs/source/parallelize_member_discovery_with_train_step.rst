Motivation
##########

Current elastic executes **rendezvous_barrier** and **train_step** in a sequential order.

From elastic performance test on Azure Kubernetes Service (each VM: 1 gpu, 6 vcpus, 56GiB), the metics show as follows,

*I write a very simple controller to  scale out 1 worker || stay || scale in 1 worker in every 60 sec.*

Min Workers: 2

Max Workers: 4

Avarage train_step time cost with 2 workers: 7627ms

Avarage rendezvous_barrier time cost with 2 workers: 12202ms

Avarage train_step time cost with 3 workers: 7962ms

Avarage rendezvous_barrier time cost with 3 workers:14848ms

Avarage train_step time cost with 4 workers: 8137ms 

Avarage rendezvous_barrier time cost with 4 workers: 16394ms


The test shows the overhead of rendezvous_barrier is non-negligible. With so much overhead, it is impractical for cluster scheduler(e.g. kubernetes) to scale **in miniutes**.


Proposal
########
Our goal is to parallelize **rendezvous_barrier** and **train_step** .

Currently, when existing workers discover new workers waiting, they stop **train_step** and wait until **rendezvous_barrier** completes.

However, it can be parallelized. 

    1. If a new worker joins in, existing workers do not need to stop immediately. Instead, they continue **train_step** until **rendezvous_barrier** completes. 
    2. If a existing worker leaves, there are two different situations, 

        * If it is requested by the scheduler, whether due to preemption or stragger detection, the leaving worker could **exit gracefully**.
          It notifies the peers to start the next rendezvous, **at the same time, it continues train_step.** when the next **rendezvous_barrier** completes, other workers will go on in the new process group.
        
        * If it is due to the node failure, either the **train_step** will timeouts or the peers will discover(the failed worker won't renew its lease). In this case, the **train_step** has to stop and wait until **rendezvous_barrier** completes.

High Level Design
#################
Modify ``CoordinatorP2P`` or add a new coordinator called ``CoordinatorParallel``, but most part of them are the same. Let's take the ``CoordinatorP2P`` as an example.

1. Add Three functions,
    
    * _rendezvous_barrier(): It is the same with current rendezvous_barrier() logic.

    * _should_rendezvous(): it contains current should_rendezvous() and check the un-renewed lease logic.

    * graceful_exit(): it will delete its lease in the etcd.


2. Add a background function called ``memebership_discovery()``.

.. code:: python

    def memebership_discovery():
        while True:
            if self.stop_training:
               break
            # it has to rendezvous at the first time.
            if not self._is_initiliazed or self._should_rendezvous():
                store, rank, world_size = self._rendezvous_barrier()
                with self.lock:
                    self.next_store, self.next_rank, self.next_world_size = store, rank, world_size
                    self._should_rendezvous_flag = True
                     self._is_initiliazed = True
    

3. Reimplement rendezvous_barrier() and should_rendezvous()

.. code:: python
    
    def rendezvous_barrier(self):

        # it has to wait at the first time
        while not self._is_initiliazed:
            cas_delay()
        with self.lock:
            self.store, self.rank, self.world_size = self.next_store, self.next_rank, self.next_world_size
            self._should_rendezvous_flag = False
            self.destroy_group()

    def should_rendezvous(self, state):
        return self._should_rendezvous_flag



Future
######
1. Although thhe above design could achieve our goal, but somehow not so straight-forward. Maybe it is better to change the train_loop directly.
2. From performance test, state.sync() also has non-negligible overhead(as much as train_step). Is there a way to make sync() more fine-grained and make it in parallel with train_step?

