# Usage

## Implement `state`

```python
import torchelastic

class MyState(torchelastic.State):
    # TODO fill out once API is fixed on State
    pass

```
## Implement `train_step`

```python
import torchelastic

def train_step(state: torchelastic.State):
    # TODO fill out once API is fixed on State
    pass

```
## Write a `main.py`

```python
import torchelastic
from torchelastic.p2p import CoordinatorP2P

if __name__ == "__main__":
        min_workers = 1
        max_workers = 1
        run_id = 1234
        etcd_endpoint = "localhost:2379"

        coordinator = CoordinatorP2P(
            c10d_backend="gloo",
            init_method=f"etcd://{etcd_endpoint}/{run_id}?min_workers={min_workers}&max_workers={max_workers}",
            max_num_trainers=max_workers,
            process_group_timeout=60000,
        )

        torchelastic.train(coordinator, train_step, MyState())
```

## Configuration

### Metrics

The metrics API in torchelastic enables users to publish telemetry metrics of
their jobs. torchelastic also publishes platform level metrics such as latencies
for certain stages of work (e.g. re-rendezvous). A `metric` can be thought of
 as timeseries data and is uniquely identified
by the string-valued tuple `(metric_group, metric_name)`. 

> torchelastic makes no assumptions about what a `metric_group` is and 
what relationship it has with `metric_name`. It is totally up to the user
to use these two fields to uniquely identify a metric.

> A sensible way to use metric groups is to map them to a stage or module
in your job. You may also encode certain high level properties of the job
such as the region or stage (dev vs prod).

The metric group `torchelastic` is used by torchelastic for platform
level metrics that it produces. For instance torchelastic may output the 
latency (in milliseconds) of a checkpoint operation by creating the metric

```
(torchelastic, checkpoint.write_latency_ms)
```

#### Add Metric Data
Using torchelastic's metrics API is similar to using python's logging framework.
You will first have to get a handle to the metric stream and add metric values
to the stream. The example below measures the latency for the `calculate()` function.

```python
import time
import torchelastic.metrics as metrics

def my_method():
    ms = metrics.getStream(group="my_app")
    start = time.time()
    calculate()
    end = time.time()

    ms.add_value("calculate_latency", int(end - start))
```

#### Publish Metrics
The `MetricHandler` is responsible for emitting the added metric values
to a particular destination. Metric groups can be configured with different
metric handlers. By default torchelastic emits all metrics to `/dev/null`.
By adding the following configuration metrics in the `torchelastic` and
`my_app` metric groups will be printed out to console. 

```python
import torchelastic.metrics as metrics

metrics.configure(metrics.ConsoleMetricsHandler(), group = "torchelastic")
metrics.configure(metrics.ConsoleMetricsHandler(), group = "my_app")
``` 

#### Implementing a Custom Metric Handler
If you want your metrics to be emitted to a custom location, implement the
`MetricHandler` interface and configure your job to use your custom metric handler.

Below is a toy example that prints the metrics to `stdout` 

```python
import torchelastic.metrics as metrics

class StdoutMetricHandler(metrics.MetricHandler):
    def emit(self, metric_data):
        print(
            f"[{metric_data.timestamp}][{metric_data.group_name}]: {metric_data.name}={metric_data.value}"
        )
        
metrics.configure(StdoutMetricHandler(), group="my_app")
```

Now all metrics in the group `my_app` will be printed to stdout as:

```
[1574213883.4182858][my_app]: my_metric=<value>
[1574213940.5237644][my_app]: my_metric=<value>
```


### Checkpoint
> TODO fill out

### Rendezvous
> TODO explain how rendezvous handlers are registered

> TODO explain how to add custom rendezvous

> TODO explain etcd rendezvous parameters (this can be a refernce to the etcd rdzv README)