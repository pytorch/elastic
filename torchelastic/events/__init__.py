#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Events API

**Overview**:

The event API is used to publish and process events. The module can be used
to capture state transitions to debug or give insight into the execution
flow. The :py:class:`torchelastic.events.Event` is an object containing information
about an occurrence during the execution of the program. The destination handler
for events can be configured by the event type parameter.


.. note:: The event type ``torchelastic`` is reserved by torchelastic for
          platform level events that can be produced by the agent or worker.


**Record Events**:

The event module resembles python's logging framework in terms
of usage and consists of two parts: handler configuration and event publishing.

The example below shows the simple event publishing mechanism
via :py:meth:`torchelastic.events.record_event` method.

::

  from torchelastic.events import Event, record_event

  event.configure(event.NullEventHandler()) # uses event_type = "torchelastic"
  event.configure(event.ConsoleEventHandler(), event_type = "foo")

  def execute():
    event = Event(name="test_event", event_type="foo")
    # The code below will be processed by the ConsoleEventHandler
    record_event(event)

Another way of using the module is via
:py:meth:`torchelastic.events.record`.

::

  from torchelastic.events import record

  def execute():
    metadata = {'key':'value'}
    record(event_name="test", event_type="console", metadata=metadata)


**Writing a Custom Event Handler**:

The custom event handler can be implemented by extending
:py:class:`torchelastic.events.EventHandler` class.

Example

::

  from torchelastic.events import EventHandler

  class StdoutEventHandler(EventHandler):
     def record(self, event):
         print(f"[{event.event_type}]: event_name: {event.name}")

  event.configure(StdoutEventHandler(), event_type="stdout_events")

Now all events with event_type 'stdout_events' will be printed to stdout.

"""

from .api import (  # noqa F401
    ConsoleEventHandler,
    Event,
    EventHandler,
    configure,
    record,
    record_event,
)
