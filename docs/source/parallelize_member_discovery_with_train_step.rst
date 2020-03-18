Motivation
##########

Current elastic executes ``memebership discovery`` and ``train step`` in a sequential order.

From elastic performance test on Azure Kubernetes Service (each VM: 1 gpu, 6 vcpus, 56GiB), the metics show as follows,

*I write a very simple controller to || scale out 1 worker || stay || scale in 1 worker || every 60 sec.*

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
Our goal is to parallelize ``memebership discovery`` and ``train step``.

Currently, when existing workers discover there are new workers waiting, they stop ``train step`` and wait until ``rendezvous_barrier`` completes.

However, it can be parallelized. 

    1. If a new worker joins in, existing workers do not need to stop immediately. Instead, they continue ``train step`` until ``rendezvous_barrier`` completes. 
    2. If a existing worker leaves, there are two different situations, 

        * If it is requested by the scheduler, whether due to preemption or stragger detection, the leaving worker could **exit gracefully**.
          It notifies the peers to start the next rendezvous, **at the same time, it continues train step.** when the next ``rendezvous_barrier`` completes, other workers will go on in the new process group.
        
        * If it is due to the node failure, either the ``train step`` will timeout or the peers will discover(the failed worker won't renew its lease). In such case, the ``train step`` has to stop and wait until ``rendezvous_barrier`` completes.

High level design
#################
We only need to modify CoordinatorP2P or add a new coordinator called CoordinatorParallel, but most part of them are the same. Let's take the CoordinatorP2P as an example.

1. Add two functions, ``_should_rendezvous()`` and ``_rendezvous_barrier()``. Actually they are the same with current ``should_rendezvous()`` and ``rendezvous_barrier()``.

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
Although thhe above design could achieve our goal, but somehow not so straight-forward. Maybe it is better to change the train_loop directly.


