Customization
=============

Torchelastic provides abilities to extend or customize different components of the system. Currently,
the following components can be customized: agent launcher, rendezvous handler and metrics handler.
Below we describe the general principles how to customize each component. You can find API description
in the corresponding API section.

Agent Launcher Extension
------------------------

The main responsibility of the launcher is to instantiate and run agents. The launcher serves
as a start point for the workflow and if you need to customize any components, you would need
to implement your own launcher. The detailed documentation can be found here :ref:`launcher-api`.

In order to setup the agent one needs to configure WorkerSpec and RendezvousHandler.
You can learn more about Rendezvous here :ref:`rendezvous-api`,
and about custom Rendezvous implementation in the section below.
The example of the custom launcher is provided below:

.. code-block:: python

    # mylauncher.py

    spec = WorkerSpec(
        local_world_size=nproc_per_node,
        fn=trainer_entrypoint_fn,
        args=(trainer_entrypoint_fn args,...),
        rdzv_handler=rdzv_handler,
        max_restarts=max_restarts,
        monitor_interval=monitor_interval,
    )
    metrics.initialize_metrics()

    elastic_agent = LocalElasticAgent(spec, start_method=start_method)
    try:
        elastic_agent.run(spec.role)
    except Exception ex:
        # handle exception

After that, you can run training jobs via:

.. code-block:: bash

    >>> python -m mylauncher ...


Rendezvous Extension
------------------------

One of the main concepts of torchelastic is Rendezvous. You can read more about it here :ref:`rendezvous-api`.
This section describes the main components to write your own Rendezvous.

.. note:: Writing rendezvous handlers can be very difficult
          and error-prone due to the complexity of the operation.

The rendezvous consists of two main components: rendezvous handler and store. The rendezvous handler
needs to implement the RendezvousHandler interface described here :ref:`rendezvous-api`.
The rendezvous store is the implementation of the torch.distributed.Store interface.

Rendezvous Handler
-------------------

The responsibility of rendezvous handler is to synchronize state between nodes and agree on the rank
of the agents for the current run. It is executed by each agent. The implementation of the handler
can be found here :ref:`rendezvous-api`.
This class implements rendezvous handler using etcd as underlying storage for communications between agents.

In order to implement rendezvous handler one must implement the *RendezvousHandler* interface.
The main method that needs to be implemented is the *next_rendezvous* method. It is executed by each agent.
Agent implements a monitoring procedure to capture events of new instances joining the run.
This is done by implementing *num_nodes_waiting* method. The agent will be constantly invoking
this method to determine whether new instances have joined the run or not. The example below shows the
skeleton of custom rendezvous handler implementation:

.. code-block:: python

    # filename: myrdzv.py

    class MyRendezvousHandler(RendezvousHandler):

        def next_rendezvous(self) -> Tuple["torch.distributed.Store", int, int]:
            # Using the synchronization procedure that is common among all the agents
            # agree on the ranks and return the common storage
        ...


The custom rendezvous handler can be integrated with the custom launcher in the following manner:

.. code-block:: python

    spec = WorkerSpec(
        rdzv_handler=MyRendezvousHandler(params),
        ...
    )
    elastic_agent = LocalElasticAgent(spec, start_method=start_method)
    elastic_agent.run(spec.role)

Rendezvous Store
-------------------

One of the outputs that Rendezvous Handler provides is the store. Store is a shared repository that can be used
to exchange data between agents.

Metrics Handler Extension
--------------------------

Torchelastic supports the ability to emit and record metrics. The metrics documentation
can be found at :ref:`metrics-api`. This section shows how to implement a custom metrics handler.
The implementation of the metrics handler consists of several things: handler implementation and
configuration.

The first thing that needs to be done is to implement the MetricsHandler interface. The
code below implements a simple metrics handler that stores metrics in memory
and allows to retrieve metrics. Lets create file mem_metrics.py with the following content:

.. code-block:: python

    # filename mymetrics.py

    import torchelastic.metrics as metrics


    class MyMetricHandler(metrics.MetricHandler):
        def emit(self, metric_data: metrics.MetricData):
            # Emit metrics

After the metrics handler is implemented, it can be integrated with the custom launcher in the following manner:

.. code-block:: python

    spec = # init spec
    metrics.initialize_metrics()
    metrics.configure(mymetrics.MyMetricHandler())
    # run elastic agent
